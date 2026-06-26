"""Tests for pipeline_dl (AttentionMLP)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

import numpy as np  # noqa: E402
from pipeline_dl import AttentionMLP, fit_attention_mlp, predict_attention_mlp  # noqa: E402


def test_forward_shape():
    model = AttentionMLP()
    cand_emb = torch.randn(4, 384)
    fb_embs = torch.randn(20, 384)
    fb_labels = torch.tensor([2] * 8 + [0] * 8 + [1] * 4)
    cand_meta = torch.randn(4, 5)
    logits = model(cand_emb, fb_embs, fb_labels, cand_meta)
    assert logits.shape == (4, 3), f"Expected (4, 3), got {logits.shape}"
    assert not logits.isnan().any(), "NaNs in forward output"
    assert not logits.isinf().any(), "Infs in forward output"


def test_forward_with_extra():
    model = AttentionMLP(extra_dim=5)
    cand_emb = torch.randn(4, 384)
    fb_embs = torch.randn(20, 384)
    fb_labels = torch.tensor([2] * 8 + [0] * 8 + [1] * 4)
    cand_meta = torch.randn(4, 5)
    extra = torch.randn(4, 5)
    logits = model(cand_emb, fb_embs, fb_labels, cand_meta, extra=extra)
    assert logits.shape == (4, 3), f"Expected (4, 3), got {logits.shape}"
    assert not logits.isnan().any(), "NaNs with extra features"


def test_up_profile_zero_when_no_upvotes():
    model = AttentionMLP()
    logits = model(
        torch.randn(2, 384),
        torch.randn(10, 384),
        torch.zeros(10, dtype=torch.long),
        torch.randn(2, 5),
    )
    assert logits.shape == (2, 3)
    assert not logits.isnan().any(), "NaNs with no upvotes"


def test_down_profile_zero_when_no_downvotes():
    model = AttentionMLP()
    logits = model(
        torch.randn(2, 384),
        torch.randn(10, 384),
        torch.full((10,), 2, dtype=torch.long),
        torch.randn(2, 5),
    )
    assert logits.shape == (2, 3)
    assert not logits.isnan().any(), "NaNs with no downvotes"


def test_all_same_class():
    """All feedback is neutral — both profiles should be finite."""
    model = AttentionMLP()
    logits = model(
        torch.randn(2, 384),
        torch.randn(10, 384),
        torch.full((10,), 1, dtype=torch.long),
        torch.randn(2, 5),
    )
    assert logits.shape == (2, 3)
    assert not logits.isnan().any(), "NaNs with all neutral feedback"


def test_handles_single_feedback():
    model = AttentionMLP()
    cand_emb = torch.randn(1, 384)
    fb_embs = torch.randn(1, 384)
    fb_labels = torch.tensor([2])
    cand_meta = torch.randn(1, 5)
    logits = model(cand_emb, fb_embs, fb_labels, cand_meta)
    assert logits.shape == (1, 3)
    assert not logits.isnan().any(), "NaNs with single feedback"


def test_loocv_excludes_self():
    model = AttentionMLP()
    cand_emb = torch.randn(1, 384)
    fb_embs = cand_emb.clone()
    fb_labels = torch.tensor([2])
    cand_meta = torch.randn(1, 5)

    logits_no_loocv = model(cand_emb, fb_embs, fb_labels, cand_meta)
    logits_loocv = model(
        cand_emb, fb_embs, fb_labels, cand_meta, loocv_idx=torch.tensor([0])
    )
    assert not logits_loocv.isnan().any(), "NaNs with LOOCV exclusion"
    assert not torch.allclose(logits_no_loocv, logits_loocv), (
        "LOOCV should change output"
    )


def test_multi_head_output_shape():
    """Output profile dim = n_heads * d_head = 128."""
    model = AttentionMLP(n_heads=4, d_head=32)
    _, pu, _, _ = _run_forward(model)
    assert pu.shape == (4, 128), f"Expected (4, 128), got {pu.shape}"


def test_n_heads_1_matches_single_head():
    """n_heads=1, d_head=64 should be equivalent to the previous single-head
    architecture with d_attn=64."""
    model = AttentionMLP(n_heads=1, d_head=64)
    cand_emb = torch.randn(4, 384)
    fb_embs = torch.randn(20, 384)
    fb_labels = torch.tensor([2] * 8 + [0] * 8 + [1] * 4)
    cand_meta = torch.randn(4, 5)
    logits = model(cand_emb, fb_embs, fb_labels, cand_meta)
    assert logits.shape == (4, 3)
    assert not logits.isnan().any()


def test_multi_head_loocv_still_excludes_self():
    """LOOCV with 2 heads: self-item should still be excluded."""
    model = AttentionMLP(n_heads=2, d_head=32)
    cand_emb = torch.randn(1, 384)
    fb_embs = cand_emb.clone()
    fb_labels = torch.tensor([2])
    cand_meta = torch.randn(1, 5)

    logits_no_loocv = model(cand_emb, fb_embs, fb_labels, cand_meta)
    logits_loocv = model(
        cand_emb, fb_embs, fb_labels, cand_meta, loocv_idx=torch.tensor([0])
    )
    assert not logits_loocv.isnan().any()
    assert not torch.allclose(logits_no_loocv, logits_loocv)


def _run_forward(model):
    cand_emb = torch.randn(4, 384)
    fb_embs = torch.randn(20, 384)
    fb_labels = torch.tensor([2] * 8 + [0] * 8 + [1] * 4)
    cand_meta = torch.randn(4, 5)
    q = model.W_q(cand_emb)
    k = model.W_k(fb_embs)
    pu, pd = model._compute_profiles(q, k, fb_embs, fb_labels)
    return cand_emb, pu, pd, cand_meta


def test_fit_runs_on_small_data():
    rng = np.random.default_rng(42)
    N = 40
    emb = rng.normal(size=(N, 384)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([2] * 15 + [0] * 15 + [1] * 10)
    rng.shuffle(labels)
    meta = np.random.uniform(0, 1, (N, 5)).astype(np.float32)
    extra = np.random.uniform(-1, 1, (N, 5)).astype(np.float32)

    model = fit_attention_mlp(
        emb,
        labels,
        meta,
        train_extra=extra,
        n_epochs=10,
        val_frac=0.2,
        seed=42,
    )
    assert model is not None, "Model should train successfully"

    scores, probs = predict_attention_mlp(
        model,
        emb,
        meta,
        emb,
        labels,
        cand_extra=extra,
    )
    assert scores.shape == (N,), f"Expected ({N},), got {scores.shape}"
    assert probs.shape == (N, 3), f"Expected ({N}, 3), got {probs.shape}"
    assert np.all(scores >= 0) and np.all(scores <= 1), "Scores outside [0, 1]"
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_fit_runs_without_extra():
    """Training without extra features should still work."""
    rng = np.random.default_rng(42)
    N = 40
    emb = rng.normal(size=(N, 384)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([2] * 15 + [0] * 15 + [1] * 10)
    rng.shuffle(labels)
    meta = np.random.uniform(0, 1, (N, 5)).astype(np.float32)

    model = fit_attention_mlp(
        emb,
        labels,
        meta,
        n_epochs=10,
        val_frac=0.2,
        seed=42,
    )
    assert model is not None, "Model should train without extra"
    scores, probs = predict_attention_mlp(model, emb, meta, emb, labels)
    assert scores.shape == (N,)
    assert probs.shape == (N, 3)


def test_fit_returns_none_on_insufficient_data():
    emb = np.random.randn(5, 384).astype(np.float32)
    labels = np.array([2, 2, 0, 0, 1])
    meta = np.random.randn(5, 5).astype(np.float32)
    model = fit_attention_mlp(emb, labels, meta, n_epochs=5)
    assert model is None, "Should return None for N < 10"


def test_predict_probs_sum_to_one():
    model = AttentionMLP()
    cand = np.random.randn(10, 384).astype(np.float32)
    meta = np.random.randn(10, 5).astype(np.float32)
    fb = np.random.randn(30, 384).astype(np.float32)
    lbl = np.array([2] * 15 + [0] * 15)
    _, probs = predict_attention_mlp(model, cand, meta, fb, lbl)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_weight_decay_supported():
    """fit_attention_mlp should accept weight_decay kwarg."""
    rng = np.random.default_rng(42)
    N = 20
    emb = rng.normal(size=(N, 384)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([2] * 8 + [0] * 7 + [1] * 5)
    rng.shuffle(labels)
    meta = np.random.uniform(0, 1, (N, 5)).astype(np.float32)
    model = fit_attention_mlp(
        emb,
        labels,
        meta,
        n_epochs=5,
        val_frac=0.2,
        seed=42,
        weight_decay=1e-3,
    )
    assert model is not None


def test_ranking_loss_basic():
    """Ranking loss should be > 0 when up items score below down items."""
    from pipeline_dl import _ranking_loss

    logits = torch.zeros((10, 3))
    logits[:3, 2] = 0.3  # up items, low up logit
    logits[3:6, 0] = 0.8  # down items, high down logit
    logits[6:, :] = 0.5
    labels = torch.tensor([2] * 3 + [0] * 3 + [1] * 4)
    loss = _ranking_loss(logits, labels, margin=0.5, k_pairs=256)
    assert loss > 0.0, f"Expected positive loss, got {loss}"


def test_ranking_loss_no_pairs():
    """Ranking loss returns 0 when only one class is present."""
    from pipeline_dl import _ranking_loss

    logits = torch.randn(10, 3)
    labels_all_up = torch.full((10,), 2)
    loss = _ranking_loss(logits, labels_all_up, margin=0.5)
    assert loss.item() == 0.0, "_ranking_loss should return 0 with no down items"

    labels_all_down = torch.full((10,), 0)
    loss = _ranking_loss(logits, labels_all_down, margin=0.5)
    assert loss.item() == 0.0, "_ranking_loss should return 0 with no up items"


def test_mixup_integration():
    """fit_attention_mlp with mixup_alpha > 0 should train successfully."""
    rng = np.random.default_rng(42)
    N = 40
    emb = rng.normal(size=(N, 384)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([2] * 15 + [0] * 15 + [1] * 10)
    rng.shuffle(labels)
    meta = np.random.uniform(0, 1, (N, 5)).astype(np.float32)

    model = fit_attention_mlp(
        emb,
        labels,
        meta,
        n_epochs=10,
        val_frac=0.2,
        seed=42,
        mixup_alpha=0.4,
    )
    assert model is not None, "Model should train with mixup"

    scores, probs = predict_attention_mlp(model, emb, meta, emb, labels)
    assert scores.shape == (N,)
    assert probs.shape == (N, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_meta_per_class_forward():
    """Forward with meta_per_class should work."""
    model = AttentionMLP(meta_per_class_dim=10)
    cand = torch.randn(4, 384)
    fb = torch.randn(20, 384)
    lbl = torch.tensor([2] * 8 + [0] * 8 + [1] * 4)
    meta = torch.randn(4, 5)
    mpc = torch.randn(4, 10)
    logits = model(cand, fb, lbl, meta, meta_per_class=mpc)
    assert logits.shape == (4, 3), f"Expected (4, 3), got {logits.shape}"
    assert not logits.isnan().any()


def test_ranking_loss_integration():
    """fit_attention_mlp with ranking loss should train successfully."""
    rng = np.random.default_rng(42)
    N = 40
    emb = rng.normal(size=(N, 384)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([2] * 15 + [0] * 15 + [1] * 10)
    rng.shuffle(labels)
    meta = np.random.uniform(0, 1, (N, 5)).astype(np.float32)

    model = fit_attention_mlp(
        emb,
        labels,
        meta,
        n_epochs=10,
        val_frac=0.2,
        seed=42,
        ranking_lambda=0.5,
        ranking_margin=0.5,
        ranking_pairs=64,
    )
    assert model is not None, "Model should train with ranking loss"

    scores, probs = predict_attention_mlp(model, emb, meta, emb, labels)
    assert scores.shape == (N,)
    assert probs.shape == (N, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_all_tier2_params():
    """fit_attention_mlp with mixup + ranking loss + meta_per_class."""
    rng = np.random.default_rng(42)
    N = 50
    emb = rng.normal(size=(N, 384)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([2] * 18 + [0] * 18 + [1] * 14)
    rng.shuffle(labels)
    meta = np.random.uniform(0, 1, (N, 5)).astype(np.float32)
    meta_pc = np.random.uniform(0, 1, (N, 10)).astype(np.float32)

    model = fit_attention_mlp(
        emb,
        labels,
        meta,
        train_meta_per_class=meta_pc,
        n_epochs=10,
        val_frac=0.2,
        seed=42,
        mixup_alpha=0.4,
        ranking_lambda=0.5,
        ranking_pairs=64,
    )
    assert model is not None, "Model should train with all Tier 2 params"

    scores, probs = predict_attention_mlp(
        model,
        emb,
        meta,
        emb,
        labels,
        cand_meta_per_class=meta_pc,
    )
    assert scores.shape == (N,)
    assert probs.shape == (N, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
