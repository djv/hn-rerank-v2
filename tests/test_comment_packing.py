from __future__ import annotations

from itertools import product

from hypothesis import example, given, strategies as st

from pipeline.ranking import join_top_comments


@given(
    texts=st.lists(st.text(alphabet="abc012 \t", max_size=40), max_size=8),
    limit=st.integers(0, 120),
)
@example(texts=["too large", "x"], limit=1)
@example(texts=["a", "b"], limit=9)
@example(texts=["a", "b"], limit=8)
@example(texts=[" ", "\t", ""], limit=0)
def test_comment_packing_matches_exhaustive_selection(
    texts: list[str], limit: int
) -> None:
    # The contract prefers earlier whole comments, not maximum character
    # utilization. Enumerate every feasible inclusion vector; lexicographic
    # maximum expresses that priority independently of the greedy algorithm.
    comments = [text for text in texts if text.strip()]
    separator = "\n\n---\n\n"
    feasible = {
        mask: separator.join(text for text, include in zip(comments, mask) if include)
        for mask in product((False, True), repeat=len(comments))
    }
    expected_mask = max(
        mask for mask, packed in feasible.items() if len(packed) <= limit
    )
    actual = join_top_comments(texts, limit=limit)
    assert actual == feasible[expected_mask]
    assert len(actual) <= limit
