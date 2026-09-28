"""Pooling and vote sampling used by the embedding-model comparison scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hypothesis import given, settings, strategies as st

from scripts.bakeoff_embedding_models import _model_inputs, _pool
from scripts.eval_ranker_variants import sample_feedback_positions


@settings(max_examples=60)
@given(
    lengths=st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=6),
    left_padded=st.booleans(),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_last_token_pooling_picks_final_real_token(
    lengths: list[int], left_padded: bool, seed: int
) -> None:
    width = max(lengths)
    mask = np.zeros((len(lengths), width), dtype=np.int64)
    for row, length in enumerate(lengths):
        if left_padded:
            mask[row, width - length :] = 1
        else:
            mask[row, :length] = 1
    hidden = np.random.default_rng(seed).normal(size=(len(lengths), width, 5))
    pooled = _pool({"last_hidden_state": hidden}, mask, "last")
    for row, length in enumerate(lengths):
        last = width - 1 if left_padded else length - 1
        expected = hidden[row, last] / np.linalg.norm(hidden[row, last])
        np.testing.assert_allclose(pooled[row], expected, rtol=1e-5)

    feed = _model_inputs(
        {"input_ids": mask, "attention_mask": mask}, {"input_ids", "position_ids"}
    )
    # Real tokens are numbered 0..length-1 whichever side is padded.
    for row, length in enumerate(lengths):
        np.testing.assert_array_equal(
            feed["position_ids"][row][mask[row] == 1], np.arange(length)
        )


def test_sentence_pooling_uses_the_model_output() -> None:
    outputs = {
        "last_hidden_state": np.ones((2, 3, 4)),
        "sentence_embedding": np.array([[3.0, 4.0], [0.0, 2.0]]),
    }
    pooled = _pool(outputs, np.ones((2, 3), dtype=np.int64), "sentence")
    np.testing.assert_allclose(pooled, [[0.6, 0.8], [0.0, 1.0]])


@settings(max_examples=40)
@given(
    labels=st.lists(st.integers(min_value=0, max_value=2), max_size=80),
    per_class=st.integers(min_value=1, max_value=30),
)
def test_feedback_sample_caps_each_class_and_is_repeatable(
    labels: list[int], per_class: int
) -> None:
    y = np.array(labels, dtype=int)
    kept = sample_feedback_positions(y, per_class)
    assert list(kept) == sorted(set(kept.tolist()))
    for label in (0, 1, 2):
        assert (y[kept] == label).sum() == min(per_class, (y == label).sum())
    np.testing.assert_array_equal(kept, sample_feedback_positions(y, per_class))


@settings(max_examples=40)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    body_share=st.floats(min_value=0.05, max_value=0.95),
)
def test_combined_vectors_are_unit_and_weighted(seed: int, body_share: float) -> None:
    from scripts.combine_replay_embeddings import combine

    rng = np.random.default_rng(seed)
    body, comments = (rng.normal(size=(5, 8)).astype(np.float32) for _ in range(2))
    body /= np.linalg.norm(body, axis=1, keepdims=True)
    comments /= np.linalg.norm(comments, axis=1, keepdims=True)
    weights = [body_share, 1 - body_share]
    joined = combine([body, comments], weights, "concat")
    np.testing.assert_allclose(np.linalg.norm(joined, axis=1), 1.0, rtol=1e-5)
    # Each part keeps its share of the squared norm.
    np.testing.assert_allclose((joined[:, :8] ** 2).sum(axis=1), body_share, rtol=1e-4)
    averaged = combine([body, comments], weights, "mean")
    expected = body_share * body + (1 - body_share) * comments
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(averaged, expected, rtol=1e-4, atol=1e-6)


class _WordTokenizer:
    """One token per whitespace-separated word."""

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        return {"input_ids": text.split()}

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


@settings(max_examples=40)
@given(
    words=st.integers(min_value=0, max_value=60),
    size=st.integers(min_value=3, max_value=12),
    max_chunks=st.integers(min_value=1, max_value=6),
)
def test_token_chunks_cover_text_in_order(
    words: int, size: int, max_chunks: int
) -> None:
    from scripts.encode_replay_embeddings import token_chunks

    text = " ".join(f"w{i}" for i in range(words))
    pieces = token_chunks(text, _WordTokenizer(), size, max_chunks)
    assert 1 <= len(pieces) <= max_chunks
    assert all(len(p.split()) <= size - 2 for p in pieces)
    covered = " ".join(pieces).split()
    assert covered == text.split()[: len(covered)]
    if words <= (size - 2) * max_chunks:
        assert covered == text.split()


def test_split_budget_text_keeps_comments_after_capped_body() -> None:
    from database import Story
    from scripts.encode_replay_embeddings import split_budget_text

    story = Story(
        id=1,
        title="Title",
        url="",
        score=1,
        time=0,
        text_content="",
        article_body=" ".join(f"a{i}" for i in range(50)),
        top_comments="first comment",
    )
    text = split_budget_text(story, _WordTokenizer(), 5)
    assert text.split()[:5] == ["Title.", "a0", "a1", "a2", "a3"]
    assert text.endswith("first comment")
    assert "a4" not in text.split()


def test_reuse_keeps_only_same_settings_and_text(tmp_path: Path) -> None:
    from scripts.encode_replay_embeddings import load_reusable

    def write(name: str, model: str, ids: list[int], hashes: list[str]) -> Path:
        path = tmp_path / name
        np.savez(
            path,
            story_ids=np.array(ids, dtype=np.int64),
            text_hashes=np.array(hashes, dtype="<U64"),
            embeddings=np.eye(len(ids), 3, dtype=np.float32),
            model=np.array(model),
        )
        return path

    same = write("same.npz", "m", [1, 2], ["a", "b"])
    other = write("other.npz", "other settings", [3], ["c"])
    found = load_reusable([same, other, tmp_path / "missing.npz"], "m")
    assert set(found) == {(1, "a"), (2, "b")}
    np.testing.assert_array_equal(found[(2, "b")], [0, 1, 0])
