"""Behavioral regressions found in the client review; no live server access."""

from __future__ import annotations

import json
from dataclasses import dataclass

from hypothesis import example, given, settings, strategies as st

from tests.test_client_js import _client_harness


class ClientStateMismatch(AssertionError):
    """The client state diverges from acknowledged votes or fetched metadata."""


@dataclass(frozen=True)
class ReratingSequence:
    actions: tuple[str, ...]
    failed_request: int


@st.composite
def rerating_sequences(draw: st.DrawFn) -> ReratingSequence:
    """Reach overlapping operations on one story, with a shrinkable failure.

    Every vote except the final one is undone before any acknowledgement.
    Varying chain length, actions, and which earlier request fails exercises
    both vote and undo rollback against subsequent successful operations.
    """
    action_indices = draw(st.lists(st.integers(0, 2), min_size=2, max_size=6))
    actions = tuple(("up", "neutral", "down")[index] for index in action_indices)
    failed_request = draw(st.integers(0, 2 * (len(actions) - 1) - 1))
    return ReratingSequence(actions, failed_request)


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(sequence=rerating_sequences())
@example(sequence=ReratingSequence(("up", "down"), 0))
@example(sequence=ReratingSequence(("up", "down"), 1))
def test_later_successful_vote_survives_earlier_request_rollback(
    sequence: ReratingSequence,
) -> None:
    """The final acknowledged action wins regardless of an earlier failure."""
    result = _client_harness(
        f"const actions = {json.dumps(sequence.actions)};\n"
        f"const failedRequest = {sequence.failed_request};\n"
        + r"""
        setFeed(makeFeed([1, 2, 3]));
        actions.forEach((action, index) => {
          vote(action);
          if (index < actions.length - 1) undo();
        });
        await flush();
        const requestCount = actions.length * 2 - 1;
        let savedAction = null;
        for (let index = 0; index < requestCount; index++) {
          const action = pending('/api/feedback')[0].body.action;
          if (index === failedRequest) {
            await fail('/api/feedback');
          } else {
            savedAction = action === 'clear' ? null : action;
            await answer('/api/feedback', 200, {
              ok: true, target_version: 6 + index,
            });
          }
        }
        console.log(JSON.stringify({
          savedAction, state: state(),
          history: history.map(entry => ({id: entry.story.id, action: entry.action})),
          sent: calls.filter(call => call.url.includes('/api/feedback')).map(call => call.body),
        }));
        """
    )
    # Independent server model: successful writes replace the prior vote;
    # the final request always succeeds and supersedes any earlier failure.
    assert result["savedAction"] == sequence.actions[-1]
    expected_requests = []
    for i, action in enumerate(sequence.actions):
        expected_requests.append({"story_id": 1, "action": action})
        if i < len(sequence.actions) - 1:
            expected_requests.append({"story_id": 1, "action": "clear"})
    assert result["sent"] == expected_requests
    observed = (
        result["state"]["rated"],
        1 in result["state"]["visible"],
        result["history"],
        result["state"]["counts"],
    )
    expected = (
        [1],
        False,
        [{"id": 1, "action": sequence.actions[-1]}],
        [int(action == sequence.actions[-1]) for action in ("up", "neutral", "down")],
    )
    if observed != expected:
        raise ClientStateMismatch(f"Expected {expected!r}, observed {observed!r}")


def test_manual_refresh_updates_counts_on_the_preserved_active_card() -> None:
    """Keeping the reading card must still apply freshly fetched metadata."""
    result = _client_harness(
        r"""
        const initial = makeFeed([1, 2]);
        initial.stories[0].comments_url = 'https://news.ycombinator.com/item?id=1';
        setFeed(initial);
        const beforeId = activeId;
        reload({manual: true});
        const refreshed = makeFeed([1, 2]);
        refreshed.stories[0] = story(1, {
          points: 444, comments: 218,
          comments_url: 'https://news.ycombinator.com/item?id=1',
        });
        await answer(FEED + '1w', 200, refreshed);
        console.log(JSON.stringify({
          beforeId, afterId: activeId,
          stored: [storiesById.get(1).points, storiesById.get(1).comments],
          header: activeCard().children[0].children.map(child => child.textContent),
        }));
        """
    )
    assert result["stored"] == [444, 218]
    assert result["afterId"] == result["beforeId"] == 1
    if "444 pts" not in result["header"] or "💬 218 comments" not in result["header"]:
        raise ClientStateMismatch(f"Fresh counts missing from {result['header']!r}")


def test_counts_only_change_refreshes_the_web_feed() -> None:
    """Hot-thread updates must reach the web without a ranking version bump."""
    result = _client_harness(
        r"""
        setFeed(makeFeed([1, 2]));
        const baseline = pollFeedVersion();
        await answer('/api/ranking-ready', 200, {
          ok: true, ready: true, current_version: 5, counts_version: 5,
        });
        await baseline;
        const changed = pollFeedVersion();
        await answer('/api/ranking-ready', 200, {
          ok: true, ready: true, current_version: 5, counts_version: 6,
        });
        await changed;
        console.log(JSON.stringify({refreshes: pending(FEED + '1w').length}));
        """
    )
    if result["refreshes"] != 1:
        raise ClientStateMismatch("A counts-only change did not refresh the feed")


def test_summary_reply_patches_counts_in_cards_and_cached_windows() -> None:
    result = _client_harness(
        r"""
        const initial = makeFeed([1, 2]);
        initial.stories[0].comments_url = 'https://news.ycombinator.com/item?id=1';
        setFeed(initial);
        const cached = makeFeed([1, 2], {window: '1d'});
        feeds.set('1d', cached);
        const before = activeCard();
        keepSummary(1, {tldr: 'Updated summary', points: 444, comments: 218});
        console.log(JSON.stringify({
          sameCard: before === activeCard(),
          cached: [cached.stories[0].points, cached.stories[0].comments],
          header: activeCard().children[0].children.map(child => child.textContent),
        }));
        """
    )
    assert result["sameCard"]
    assert result["cached"] == [444, 218]
    assert "444 pts" in result["header"]
    assert "💬 218 comments" in result["header"]
