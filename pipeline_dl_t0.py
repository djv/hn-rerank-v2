"""T0 attention-pooled MLP baseline (single-head, elementwise features)."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

logger = logging.getLogger(__name__)

EMB_DIM = 384


class AttentionMLPT0(nn.Module):
    """Single-head attention-pooled user profile + MLP head (T0 baseline).

    Architecture matches the original pre-Tier-1 implementation:
        1. Single-head dot-product attention (W_q, W_k), no W_v
        2. Profiles = raw-embedding weighted sums (emb_dim-d)
        3. Feature vector: [cand_emb, pu, pd, cand_emb*pu, |cand_emb-pu|, meta, *extra]
    """

    def __init__(
        self,
        emb_dim: int = EMB_DIM,
        d_attn: int = 64,
        meta_dim: int = 5,
        hidden_dim: int = 64,
        n_classes: int = 3,
        dropout: float = 0.3,
        extra_dim: int = 0,
    ):
        super().__init__()
        self.W_q = nn.Linear(emb_dim, d_attn, bias=False)
        self.W_k = nn.Linear(emb_dim, d_attn, bias=False)
        feat_dim = emb_dim * 5 + meta_dim + extra_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "mlp.0.weight" in name:
                nn.init.kaiming_uniform_(p, a=0, mode="fan_in", nonlinearity="relu")
            elif "mlp.3.weight" in name:
                nn.init.xavier_uniform_(p)
            elif "weight" in name:
                nn.init.xavier_uniform_(p)

    def _compute_profiles(
        self,
        q: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        labels: torch.Tensor,
        loocv_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = q.shape[0]
        device = q.device
        d = q.shape[1]

        logits = q @ k_all.T / (d**0.5)

        up_mask = labels == 2
        down_mask = labels == 0

        logits_up = logits.clone()
        logits_up[:, ~up_mask] = -float("inf")
        logits_down = logits.clone()
        logits_down[:, ~down_mask] = -float("inf")

        if loocv_idx is not None:
            idx = torch.arange(B, device=device)
            is_up_loocv = labels[loocv_idx] == 2
            logits_up[idx[is_up_loocv], loocv_idx[is_up_loocv]] = -float("inf")
            is_down_loocv = labels[loocv_idx] == 0
            logits_down[idx[is_down_loocv], loocv_idx[is_down_loocv]] = -float("inf")

        attn_up = F.softmax(logits_up, dim=1)
        attn_down = F.softmax(logits_down, dim=1)
        attn_up = torch.nan_to_num(attn_up, nan=0.0)
        attn_down = torch.nan_to_num(attn_down, nan=0.0)

        profile_up = attn_up @ v_all
        profile_down = attn_down @ v_all
        return profile_up, profile_down

    def forward(
        self,
        cand_emb: torch.Tensor,
        fb_embs: torch.Tensor,
        fb_labels: torch.Tensor,
        cand_meta: torch.Tensor,
        loocv_idx: Optional[torch.Tensor] = None,
        extra: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.W_q(cand_emb)
        k = self.W_k(fb_embs)
        pu, pd = self._compute_profiles(q, k, fb_embs, fb_labels, loocv_idx)

        parts = [cand_emb, pu, pd, cand_emb * pu, (cand_emb - pu).abs(), cand_meta]
        if extra is not None:
            parts.append(extra)
        feat = torch.cat(parts, dim=1)
        return self.mlp(feat)


def fit_attention_mlp_t0(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    train_meta: np.ndarray,
    *,
    train_extra: Optional[np.ndarray] = None,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
    val_frac: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
) -> Optional[AttentionMLPT0]:
    N = len(train_emb)
    if N < 10:
        return None

    torch.manual_seed(seed)

    embs = torch.from_numpy(train_emb).float()
    labels = torch.from_numpy(train_labels).long()
    meta = torch.from_numpy(train_meta).float()
    extra_t = torch.from_numpy(train_extra).float() if train_extra is not None else None
    n_extra = train_extra.shape[1] if train_extra is not None else 0

    counts = torch.bincount(labels, minlength=3).float()
    class_weights_t = N / (3.0 * counts)
    class_weights_t[counts == 0] = 1.0

    perm = torch.randperm(N)
    n_val = max(1, int(N * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    val_emb, val_labels, val_meta = embs[val_idx], labels[val_idx], meta[val_idx]
    tr_emb, tr_labels, tr_meta = embs[tr_idx], labels[tr_idx], meta[tr_idx]

    if extra_t is not None:
        tr_extra = extra_t[tr_idx]
        val_extra = extra_t[val_idx]
    else:
        tr_extra = None
        val_extra = None

    model = AttentionMLPT0(meta_dim=train_meta.shape[1], extra_dim=n_extra)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t.to(device))

    best_val_loss = torch.inf
    best_state: Optional[dict] = None
    patience_ct = 0

    for epoch in range(n_epochs):
        model.train()

        k_all = model.W_k(tr_emb)
        q_all = model.W_q(tr_emb)

        logits = q_all @ k_all.T / (model.W_q.out_features**0.5)

        up_mask = tr_labels == 2
        down_mask = tr_labels == 0

        logits_up = logits.clone()
        logits_up[:, ~up_mask] = -float("inf")
        logits_down = logits.clone()
        logits_down[:, ~down_mask] = -float("inf")

        ar = torch.arange(len(tr_emb), device=device)
        logits_up[ar[up_mask], ar[up_mask]] = -float("inf")
        logits_down[ar[down_mask], ar[down_mask]] = -float("inf")

        attn_up = torch.nan_to_num(F.softmax(logits_up, dim=1), nan=0.0)
        attn_down = torch.nan_to_num(F.softmax(logits_down, dim=1), nan=0.0)

        profile_up = attn_up @ tr_emb
        profile_down = attn_down @ tr_emb

        parts = [
            tr_emb,
            profile_up,
            profile_down,
            tr_emb * profile_up,
            (tr_emb - profile_up).abs(),
            tr_meta,
        ]
        if tr_extra is not None:
            parts.append(tr_extra)
        feat = torch.cat(parts, dim=1)
        logits_mlp = model.mlp(feat)

        opt.zero_grad()
        loss = criterion(logits_mlp, tr_labels)
        if loss.isnan():
            logger.warning("NaN loss at epoch %d, skipping", epoch)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            k_all = model.W_k(tr_emb)
            q_val = model.W_q(val_emb)
            logits_val = q_val @ k_all.T / (model.W_q.out_features**0.5)

            logits_val_up = logits_val.clone()
            logits_val_up[:, ~up_mask] = -float("inf")
            logits_val_down = logits_val.clone()
            logits_val_down[:, ~down_mask] = -float("inf")

            attn_val_up = torch.nan_to_num(F.softmax(logits_val_up, dim=1), nan=0.0)
            attn_val_down = torch.nan_to_num(F.softmax(logits_val_down, dim=1), nan=0.0)

            profile_val_up = attn_val_up @ tr_emb
            profile_val_down = attn_val_down @ tr_emb

            parts_val = [
                val_emb,
                profile_val_up,
                profile_val_down,
                val_emb * profile_val_up,
                (val_emb - profile_val_up).abs(),
                val_meta,
            ]
            if val_extra is not None:
                parts_val.append(val_extra)
            feat_val = torch.cat(parts_val, dim=1)
            val_logits = model.mlp(feat_val)
            val_loss = criterion(val_logits, val_labels)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                logger.debug("Early stopping at epoch %d", epoch)
                break

    if best_state is None:
        return None

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_attention_mlp_t0(
    model: AttentionMLPT0,
    cand_emb: np.ndarray,
    cand_meta: np.ndarray,
    fb_emb: np.ndarray,
    fb_labels: np.ndarray,
    *,
    cand_extra: Optional[np.ndarray] = None,
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)

    c_t = torch.from_numpy(cand_emb).float().to(device)
    m_t = torch.from_numpy(cand_meta).float().to(device)
    f_t = torch.from_numpy(fb_emb).float().to(device)
    l_t = torch.from_numpy(fb_labels).long().to(device)
    e_t = (
        torch.from_numpy(cand_extra).float().to(device)
        if cand_extra is not None
        else None
    )

    logits_list = []
    for i in range(0, len(cand_emb), batch_size):
        logits = model(
            c_t[i : i + batch_size],
            f_t,
            l_t,
            m_t[i : i + batch_size],
            extra=e_t[i : i + batch_size] if e_t is not None else None,
        )
        logits_list.append(logits.cpu())

    logits = torch.cat(logits_list, dim=0)
    probs = F.softmax(logits, dim=1).numpy().astype(np.float32)
    scores = probs[:, 2].copy()
    return scores, probs
