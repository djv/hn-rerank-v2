"""PyTorch attention-pooled user profile MLP for personalization."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EMB_DIM = 384


class AttentionMLP(nn.Module):
    """Multi-head attention-pooled user profile + MLP head.

    Architecture:
        1. Learned W_q, W_k, W_v projections for multi-head dot-product attention
        2. Attention-pooled up profile: weighted mean of upvoted feedback
        3. Attention-pooled down profile: weighted mean of downvoted feedback
        4. Feature vector: [cand_emb, up_profile, down_profile, meta, extra]
        5. MLP: feat_dim -> hidden_dim -> 3 logits
    """

    def __init__(
        self,
        emb_dim: int = EMB_DIM,
        d_head: int = 32,
        n_heads: int = 4,
        meta_dim: int = 5,
        hidden_dim: int = 256,
        n_classes: int = 3,
        dropout: float = 0.2,
        extra_dim: int = 0,
        meta_per_class_dim: int = 0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_head = d_head
        self.n_heads = n_heads
        d_attn = n_heads * d_head

        self.W_q = nn.Linear(emb_dim, d_attn, bias=False)
        self.W_k = nn.Linear(emb_dim, d_attn, bias=False)
        self.W_v = nn.Linear(emb_dim, d_attn, bias=False)

        feat_dim = emb_dim + d_attn + d_attn + meta_dim + extra_dim + meta_per_class_dim
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
        """Multi-head attention-pooled up and down user profiles.

        Args:
            q: (B, n_heads * d_head) — projected candidate queries
            k_all: (N, n_heads * d_head) — projected feedback keys
            v_all: (N, emb_dim) — raw feedback embeddings
            labels: (N,) — class labels {0: down, 1: neutral, 2: up}
            loocv_idx: (B,) or None — position of each batch item in k_all

        Returns:
            (profile_up, profile_down): both (B, n_heads * d_head)
        """
        B = q.shape[0]
        device = q.device
        n_heads = self.n_heads
        d_head = self.d_head

        q_mh = q.view(B, n_heads, d_head)
        k_mh = k_all.view(-1, n_heads, d_head)
        v_mh = self.W_v(v_all).view(-1, n_heads, d_head)

        logits = torch.stack(
            [q_mh[:, h] @ k_mh[:, h].T / (d_head**0.5) for h in range(n_heads)], dim=1
        )

        up_mask = labels == 2
        down_mask = labels == 0

        logits_up = logits.clone()
        logits_up[:, :, ~up_mask] = -float("inf")
        logits_down = logits.clone()
        logits_down[:, :, ~down_mask] = -float("inf")

        if loocv_idx is not None:
            idx = torch.arange(B, device=device)
            is_up_loocv = labels[loocv_idx] == 2
            is_down_loocv = labels[loocv_idx] == 0
            if is_up_loocv.any():
                logits_up[idx[is_up_loocv], :, loocv_idx[is_up_loocv]] = -float("inf")
            if is_down_loocv.any():
                logits_down[idx[is_down_loocv], :, loocv_idx[is_down_loocv]] = -float(
                    "inf"
                )

        attn_up = F.softmax(logits_up, dim=-1)
        attn_down = F.softmax(logits_down, dim=-1)
        attn_up = torch.nan_to_num(attn_up, nan=0.0)
        attn_down = torch.nan_to_num(attn_down, nan=0.0)

        profile_up = torch.einsum("bhn,nhd->bhd", attn_up, v_mh)
        profile_down = torch.einsum("bhn,nhd->bhd", attn_down, v_mh)
        profile_up = profile_up.reshape(B, -1)
        profile_down = profile_down.reshape(B, -1)
        return profile_up, profile_down

    def forward(
        self,
        cand_emb: torch.Tensor,
        fb_embs: torch.Tensor,
        fb_labels: torch.Tensor,
        cand_meta: torch.Tensor,
        loocv_idx: Optional[torch.Tensor] = None,
        extra: Optional[torch.Tensor] = None,
        meta_per_class: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.W_q(cand_emb)
        k = self.W_k(fb_embs)
        pu, pd = self._compute_profiles(q, k, fb_embs, fb_labels, loocv_idx)

        parts: list[torch.Tensor] = [cand_emb, pu, pd, cand_meta]
        if extra is not None:
            parts.append(extra)
        if meta_per_class is not None:
            parts.append(meta_per_class)
        feat = torch.cat(parts, dim=1)
        return self.mlp(feat)


def _ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.5,
    k_pairs: int = 256,
) -> torch.Tensor:
    """Pairwise hinge loss on (up, down) pairs.

    Samples up to k_pairs random (up, down) pairs and computes
    max(0, margin - (logit_up - logit_down)).

    Returns a scalar tensor (0 if either class is missing).
    """
    up_idx = torch.where(labels == 2)[0]
    down_idx = torch.where(labels == 0)[0]
    n_up = len(up_idx)
    n_down = len(down_idx)
    if n_up == 0 or n_down == 0:
        return torch.tensor(0.0, device=logits.device)

    n_pairs = min(k_pairs, n_up * n_down)
    up_sampled = up_idx[torch.randint(0, n_up, (n_pairs,), device=logits.device)]
    down_sampled = down_idx[torch.randint(0, n_down, (n_pairs,), device=logits.device)]

    scores_up = logits[up_sampled, 2]
    scores_down = logits[down_sampled, 0]
    return F.relu(margin - (scores_up - scores_down)).mean()


def fit_attention_mlp(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    train_meta: np.ndarray,
    *,
    train_extra: Optional[np.ndarray] = None,
    train_meta_per_class: Optional[np.ndarray] = None,
    n_epochs: int = 100,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    val_frac: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
    mixup_alpha: float = 0.0,
    ranking_lambda: float = 0.0,
    ranking_margin: float = 0.5,
    ranking_pairs: int = 256,
) -> Optional[AttentionMLP]:
    """Train attention-pooled MLP per user.

    Args:
        train_emb: (N, emb_dim) L2-normalized MiniLM embeddings
        train_labels: (N,) class labels {0: down, 1: neutral, 2: up}
        train_meta: (N, meta_dim) scaled meta features
        train_extra: (N, 5) or None — cosine-similarity features
        hidden_dim: width of MLP hidden layer (default 256)

    Returns:
        fitted model in eval mode, or None if training data is too small.
    """
    N = len(train_emb)
    if N < 10:
        return None

    torch.manual_seed(seed)

    embs = torch.from_numpy(train_emb).float()
    labels = torch.from_numpy(train_labels).long()
    meta = torch.from_numpy(train_meta).float()
    extra_t = torch.from_numpy(train_extra).float() if train_extra is not None else None
    meta_pc_t = (
        torch.from_numpy(train_meta_per_class).float()
        if train_meta_per_class is not None
        else None
    )

    n_extra = train_extra.shape[1] if train_extra is not None else 0
    n_meta_pc = train_meta_per_class.shape[1] if train_meta_per_class is not None else 0

    counts = torch.bincount(labels, minlength=3).float()
    class_weights_t = N / (3.0 * counts)
    class_weights_t[counts == 0] = 1.0

    perm = torch.randperm(N)
    n_val = max(1, int(N * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    val_emb = embs[val_idx]
    val_labels = labels[val_idx]
    val_meta = meta[val_idx]
    tr_emb = embs[tr_idx]
    tr_labels = labels[tr_idx]
    tr_meta = meta[tr_idx]

    if extra_t is not None:
        tr_extra = extra_t[tr_idx]
        val_extra = extra_t[val_idx]
    else:
        tr_extra = None
        val_extra = None

    if meta_pc_t is not None:
        tr_meta_pc = meta_pc_t[tr_idx]
        val_meta_pc = meta_pc_t[val_idx]
    else:
        tr_meta_pc = None
        val_meta_pc = None

    model = AttentionMLP(
        meta_dim=train_meta.shape[1],
        extra_dim=n_extra,
        hidden_dim=hidden_dim,
        meta_per_class_dim=n_meta_pc,
    )
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t.to(device))

    best_val_loss = torch.inf
    best_state: Optional[dict] = None
    patience_ct = 0

    for epoch in range(n_epochs):
        model.train()

        if mixup_alpha > 0 and len(tr_emb) >= 2:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm_mix = torch.randperm(len(tr_emb), device=device)
            tr_emb_feat = lam * tr_emb + (1 - lam) * tr_emb[perm_mix]
            tr_meta_feat = lam * tr_meta + (1 - lam) * tr_meta[perm_mix]
            tr_labels_mix = (
                lam * F.one_hot(tr_labels, num_classes=3).float()
                + (1 - lam) * F.one_hot(tr_labels[perm_mix], num_classes=3).float()
            )
            use_mixup = True
        else:
            tr_emb_feat = tr_emb
            tr_meta_feat = tr_meta
            tr_labels_mix = None
            use_mixup = False

        k_all = model.W_k(tr_emb)
        q_all = model.W_q(tr_emb_feat)

        pu, pd = model._compute_profiles(
            q_all,
            k_all,
            tr_emb,
            tr_labels,
            loocv_idx=torch.arange(len(tr_emb_feat), device=device),
        )

        parts: list[torch.Tensor] = [tr_emb_feat, pu, pd, tr_meta_feat]
        if tr_extra is not None:
            parts.append(tr_extra)
        if tr_meta_pc is not None:
            parts.append(tr_meta_pc)
        feat = torch.cat(parts, dim=1)
        logits_mlp = model.mlp(feat)

        opt.zero_grad()
        if use_mixup:
            log_probs = F.log_softmax(logits_mlp, dim=1)
            ce_loss = -(log_probs * tr_labels_mix.to(device)).sum(dim=1).mean()
        else:
            ce_loss = criterion(logits_mlp, tr_labels)

        ranking_loss = _ranking_loss(
            logits_mlp,
            tr_labels,
            margin=ranking_margin,
            k_pairs=ranking_pairs,
        )
        loss = ce_loss + ranking_lambda * ranking_loss

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
            pu_val, pd_val = model._compute_profiles(
                q_val,
                k_all,
                tr_emb,
                tr_labels,
            )

            parts_val: list[torch.Tensor] = [val_emb, pu_val, pd_val, val_meta]
            if val_extra is not None:
                parts_val.append(val_extra)
            if val_meta_pc is not None:
                parts_val.append(val_meta_pc)
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
def predict_attention_mlp(
    model: AttentionMLP,
    cand_emb: np.ndarray,
    cand_meta: np.ndarray,
    fb_emb: np.ndarray,
    fb_labels: np.ndarray,
    *,
    cand_extra: Optional[np.ndarray] = None,
    cand_meta_per_class: Optional[np.ndarray] = None,
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch predict for candidates.

    Returns:
        scores: (M,) — prob_up (class 2)
        probs: (M, 3) — softmax probabilities [down, neutral, up]
    """
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
    mp_t = (
        torch.from_numpy(cand_meta_per_class).float().to(device)
        if cand_meta_per_class is not None
        else None
    )

    logits_list = []
    for i in range(0, len(cand_emb), batch_size):
        kw = dict(extra=e_t[i : i + batch_size] if e_t is not None else None)
        if mp_t is not None:
            kw["meta_per_class"] = mp_t[i : i + batch_size]
        logits = model(
            c_t[i : i + batch_size],
            f_t,
            l_t,
            m_t[i : i + batch_size],
            **kw,
        )
        logits_list.append(logits.cpu())

    logits = torch.cat(logits_list, dim=0)
    probs = F.softmax(logits, dim=1).numpy().astype(np.float32)
    scores = probs[:, 2].copy()
    return scores, probs
