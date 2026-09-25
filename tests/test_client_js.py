"""Executed tests for the dashboard's inline client script.

The script lives in ``templates/index.html``. These tests pull named
functions out of it and run them under Node against small DOM/fetch stubs,
so they check behavior (ordering, rollback, coalescing) rather than pinning
source text. Skipped when Node is unavailable.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "index.html"


def _inline_script() -> str:
    html = _TEMPLATE.read_text(encoding="utf-8")
    start = html.index("  <script>\n")
    return html[start : html.index("  </script>\n", start)]


def _matching(text: str, open_idx: int, pair: str) -> int:
    """Index just past the bracket closing ``text[open_idx]``, skipping quoted
    strings and comments (braces inside template-literal ``${...}`` stay
    balanced)."""
    opener, closer = pair
    depth = 0
    i = open_idx
    quote: str | None = None
    while i < len(text):
        ch = text[i]
        if quote is None and text.startswith("//", i):
            i = text.index("\n", i)
            continue
        if quote is None and text.startswith("/*", i):
            i = text.index("*/", i) + 2
            continue
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError(f"unbalanced {pair} from {open_idx}")


def js_functions(script: str, *names: str) -> str:
    """Source of the named top-level functions, in the given order."""
    chunks = []
    for name in names:
        match = re.search(rf"(?:async\s+)?function {name}\(", script)
        if match is None:
            raise AssertionError(f"function {name} not found in inline script")
        params_end = _matching(script, match.end() - 1, "()")
        body_start = script.index("{", params_end)
        chunks.append(script[match.start() : _matching(script, body_start, "{}")])
    return "\n\n".join(chunks)


def run_node(source: str) -> Any:
    """Run *source* under Node; it must print one JSON line last (parsed JSON
    is this boundary's only untyped value)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the client script")
    completed = subprocess.run(
        [node, "-e", source], capture_output=True, text=True, timeout=20
    )
    if completed.returncode != 0:
        raise AssertionError(f"node failed:\n{completed.stderr}\n{completed.stdout}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


_VOTE_STUBS = r"""
let activeCard = null, lastVote = null, nextVoteId = 1;
const votedStoryIds = new Set();
const feedbackChains = new Map();
const latestVoteIds = new Map();
const VOTED_STORAGE_LIMIT = 1000;
const storage = new Map();
const window = {
  localStorage: {
    getItem: k => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => storage.set(k, String(v)),
  },
  setTimeout: () => 0,
};
const counts = { up: { textContent: '0' }, down: { textContent: '0' }, neutral: { textContent: '0' } };
const document = {
  querySelector: sel => {
    const m = sel.match(/data-vote-count="(\w+)"/);
    return m ? counts[m[1]] : null;
  },
  querySelectorAll: () => [],
};
const storiesContainer = { dataset: { userId: '7' }, insertBefore() {}, firstChild: null };
const toasts = [];
function showToast(message, variant) { toasts.push(variant); }
const refreshes = [];
function scheduleVoteRefresh(data) { refreshes.push(data.target_version); }
function focusActiveCard() {}
function setActiveCard(card) { activeCard = card; }
function nextQueuedSibling() { return null; }
function showNextCard() {}
function queuedCards() { return []; }
const sent = [];
function sendFeedback(storyId, action) {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  sent.push({ storyId, action, resolve, reject });
  return promise;
}
function makeCard(id) {
  return {
    dataset: { storyId: String(id) },
    style: { setProperty() {} },
    classList: { add() {}, remove() {} },
    isConnected: true,
    remove() { this.isConnected = false; },
  };
}
const flush = async () => { for (let i = 0; i < 10; i++) await new Promise(r => setImmediate(r)); };
const count = action => Number(counts[action].textContent);
const stored = () => JSON.parse(storage.get('hnRewrite:votedStoryIds:7') || '[]');
console.error = () => {};
"""


def _vote_harness(scenario: str) -> dict[str, Any]:
    script = _inline_script()
    functions = js_functions(
        script,
        "votedStorageKey",
        "writeStoredVotedStoryIds",
        "rememberVotedStoryId",
        "forgetVotedStoryId",
        "enqueueFeedback",
        "adjustVoteCount",
        "incrementVoteCount",
        "decrementVoteCount",
        "submitVote",
        "undoLastVote",
    )
    source = (
        _VOTE_STUBS
        + functions
        + "\n(async () => {\n"
        + scenario
        + "\n})().catch(e => { console.log(JSON.stringify({error: String(e.stack)})); });\n"
    )
    result = run_node(source)
    assert isinstance(result, dict)
    assert "error" not in result, result.get("error")
    return result


def test_vote_undo_revote_on_one_story_is_serialized_and_counted_once() -> None:
    result = _vote_harness(
        r"""
  const card = makeCard(1);
  submitVote('up', card);
  const afterVote = { up: count('up'), stored: stored() };
  await flush();
  const inFlightAfterVote = sent.length;
  undoLastVote();
  submitVote('down', card);
  await flush();
  // clear and the revote wait for the first request on the same story
  const inFlightBeforeResolve = sent.length;
  for (let i = 0; i < 3; i++) {
    sent[i].resolve({ target_version: i + 2 });
    await flush();
  }
  console.log(JSON.stringify({
    afterVote, inFlightAfterVote, inFlightBeforeResolve,
    actions: sent.map(s => s.action),
    counts: { up: count('up'), down: count('down') },
    stored: stored(), refreshes, toasts, chains: feedbackChains.size,
  }));
"""
    )
    assert result["afterVote"] == {"up": 1, "stored": [1]}
    assert result["inFlightAfterVote"] == 1
    assert result["inFlightBeforeResolve"] == 1
    assert result["actions"] == ["up", "clear", "down"]
    assert result["counts"] == {"up": 0, "down": 1}
    assert result["stored"] == [1]
    assert result["refreshes"] == [2, 3, 4]
    assert result["toasts"] == []
    assert result["chains"] == 0


def test_failed_save_after_undo_does_not_decrement_twice() -> None:
    result = _vote_harness(
        r"""
  const card = makeCard(2);
  submitVote('up', card);
  undoLastVote();
  await flush();
  sent[0].reject(new Error('offline'));
  await flush();
  sent[1].resolve({ target_version: 2 });
  await flush();
  console.log(JSON.stringify({ up: count('up'), toasts, stored: stored() }));
"""
    )
    assert result == {"up": 0, "toasts": ["error"], "stored": []}


def test_failed_save_rolls_back_its_story_but_not_a_newer_vote() -> None:
    result = _vote_harness(
        r"""
  // Vote on story 3, then story 4; story 3's save fails.
  const a = makeCard(3), b = makeCard(4);
  submitVote('up', a);
  submitVote('down', b);
  await flush();
  sent[0].reject(new Error('offline'));
  await flush();
  const afterOtherStory = {
    up: count('up'), down: count('down'), stored: stored(),
    aVoted: a.dataset.voted ?? null, lastVoteStory: lastVote && lastVote.storyId,
  };
  // Same story: vote, undo, revote; the first request fails late and must
  // not clobber the revote.
  const c = makeCard(5);
  submitVote('up', c);
  undoLastVote();
  submitVote('neutral', c);
  await flush();
  sent[2].reject(new Error('offline'));
  await flush();
  console.log(JSON.stringify({
    afterOtherStory,
    cVoted: c.dataset.voted ?? null,
    storedAfterRevote: stored(),
    lastVoteStory: lastVote && lastVote.storyId,
  }));
"""
    )
    assert result["afterOtherStory"] == {
        "up": 0,
        "down": 1,
        # Story 3 was never saved, so it must not stay suppressed.
        "stored": [4],
        "aVoted": None,
        "lastVoteStory": 4,
    }
    assert result["cVoted"] == "neutral"
    assert result["storedAfterRevote"] == [4, 5]
    assert result["lastVoteStory"] == 5


_REFILL_STUBS = r"""
let activeCard = null;
let refillInFlight = false, queuedRefillAdvance = null;
let warmPollInFlight = false, warmMinVersion = null, latestWarmTargetVersion = null;
let lastScheduledWarmVersion = null, voteWarmIdleTimer = null, latestVoteTargetVersion = null;
let now = 0;
const Date = { now: () => now };
const window = {
  setTimeout: (fn, ms) => { setImmediate(() => { now += ms || 0; fn(); }); return 1; },
  clearTimeout: () => {},
  requestAnimationFrame: fn => setImmediate(fn),
};
function apiPath(p) { return p; }
function updateQueueLoading() {}
const refills = [];
let releaseRefill = null;
async function refillQueue({ advance }) {
  refills.push(advance);
  if (refills.length === 1 && holdFirstRefill) {
    await new Promise(r => { releaseRefill = r; });
  }
}
let holdFirstRefill = false;
const polls = [];
let readyScript = [];
async function fetch(url) {
  const q = new URL('http://x' + url).searchParams;
  polls.push(Number(q.get('min_version')));
  const readyVersion = readyScript.length ? readyScript.shift() : null;
  return { ok: true, json: async () => ({ ready: readyVersion !== null, ready_version: readyVersion }) };
}
const flush = async () => { for (let i = 0; i < 50; i++) await new Promise(r => setImmediate(r)); };
console.error = () => {};
"""


def _refill_harness(scenario: str) -> dict[str, Any]:
    functions = js_functions(
        _inline_script(),
        "rankingReadyPath",
        "waitForVoteRemoval",
        "scheduleDeckRefresh",
        "scheduleVoteRefresh",
        "queueRefill",
        "runRefillLoop",
        "runWarmPollLoop",
        "waitForRankingReady",
    )
    result = run_node(
        _REFILL_STUBS
        + functions
        + "\n(async () => {\n"
        + scenario
        + "\n})().catch(e => console.log(JSON.stringify({error: String(e.stack)})));\n"
    )
    assert isinstance(result, dict)
    assert "error" not in result, result.get("error")
    return result


def test_refill_lane_coalesces_requests_made_while_one_is_in_flight() -> None:
    result = _refill_harness(
        r"""
  holdFirstRefill = true;
  queueRefill(false);
  await flush();
  queueRefill(false);
  queueRefill(true);
  queueRefill(false);
  releaseRefill();
  await flush();
  console.log(JSON.stringify({ refills, inFlight: refillInFlight }));
"""
    )
    # One follow-up refill for the three queued requests; advance is sticky.
    assert result == {"refills": [False, True], "inFlight": False}


def test_warm_poll_loads_intermediate_version_then_keeps_polling_to_target() -> None:
    result = _refill_harness(
        r"""
  readyScript = [null, 3, 5];
  scheduleDeckRefresh({ waitForWarm: true, targetVersion: 3, advance: false });
  scheduleDeckRefresh({ waitForWarm: true, targetVersion: 5, advance: false });
  scheduleDeckRefresh({ waitForWarm: true, targetVersion: 4, advance: false }); // older: ignored
  await flush();
  console.log(JSON.stringify({ refills, polls, inFlight: warmPollInFlight,
    pending: [warmMinVersion, latestWarmTargetVersion] }));
"""
    )
    # Polls from the earliest useful version; a ready intermediate deck is
    # loaded without advancing the active card, then polling resumes above it.
    assert result == {
        "refills": [False, False],
        "polls": [3, 3, 4],
        "inFlight": False,
        "pending": [None, None],
    }


def test_warm_poll_gives_up_after_timeout_without_refilling() -> None:
    result = _refill_harness(
        r"""
  scheduleDeckRefresh({ waitForWarm: true, targetVersion: 2, advance: false });
  await flush();
  for (let i = 0; i < 40 && warmPollInFlight; i++) await flush();
  console.log(JSON.stringify({ refills, elapsed: now, inFlight: warmPollInFlight }));
"""
    )
    assert result["refills"] == []
    assert result["inFlight"] is False
    assert 30000 <= result["elapsed"] <= 33000


@pytest.mark.parametrize(
    ("page_version", "current_version", "polls"),
    [("0", "1", True), ("1", "1", False), ("3", "5", True), ("", "1", False)],
)
def test_page_load_polls_only_when_served_version_is_behind(
    page_version: str, current_version: str, polls: bool
) -> None:
    script = _inline_script()
    start = script.index("    // If the page was served from a stale cache")
    snippet = script[start:]
    result = run_node(
        r"""
const calls = [];
function scheduleDeckRefresh(opts) { calls.push(opts); }
const document = { getElementById: () => ({ dataset: {
  dashboardVersion: %s, currentVersion: %s } }) };
"""
        % (json.dumps(page_version), json.dumps(current_version))
        + snippet
        + "\nconsole.log(JSON.stringify(calls));\n"
    )
    # Version 0 is valid cold-deck data: finiteness, not truthiness, decides.
    expected = (
        [{"waitForWarm": True, "targetVersion": int(current_version), "advance": True}]
        if polls
        else []
    )
    assert result == expected


_REFILL_QUEUE_STUBS = r"""
const votedStoryIds = new Set([2]);
const calls = [];
const children = [];
const storiesContainer = { appendChild(card) { card.isConnected = true; children.push(card); } };
function makeCard(id) {
  return {
    dataset: { storyId: String(id) }, isConnected: true,
    remove() { this.isConnected = false; children.splice(children.indexOf(this), 1); },
    focus() {},
  };
}
function cards() { return children.slice(); }
const document = { activeElement: null };
let incoming = [];
async function fetchRefillCards() { return incoming; }
function applyGradient() { calls.push('gradient'); }
function orderForCurrentSort() { calls.push('order'); }
function showNextCard(opts) { calls.push('show:' + opts.allowRefresh); }
function prefetchUpcomingTldrs() { calls.push('prefetch'); }
console.debug = () => {};
"""


@pytest.mark.parametrize(
    ("has_active", "advance", "activates"),
    [(True, False, False), (False, False, True), (True, True, True)],
)
def test_refill_queue_filters_voted_and_activates_only_when_needed(
    has_active: bool, advance: bool, activates: bool
) -> None:
    functions = js_functions(_inline_script(), "refillQueue")
    result = run_node(
        _REFILL_QUEUE_STUBS
        + "let activeCard = null;\n"
        + functions
        + r"""
(async () => {
  const active = makeCard(1), stale = makeCard(9);
  children.push(active, stale);
  if (%s) activeCard = active; else { active.remove(); }
  // 1 duplicates the active card, 2 was voted by this browser.
  incoming = [makeCard(1), makeCard(2), makeCard(3)];
  await refillQueue({ advance: %s });
  console.log(JSON.stringify({ ids: children.map(c => Number(c.dataset.storyId)), calls }));
})();
"""
        % ("true" if has_active else "false", "true" if advance else "false")
    )
    assert isinstance(result, dict)
    # The stale card is dropped, the duplicate and the voted story skipped.
    assert result["ids"] == [1, 3]
    calls = result["calls"]
    assert calls[:2] == ["gradient", "order"]
    assert calls[2:] == (["show:false"] if activates else ["prefetch"])


def test_vote_refresh_refills_now_and_polls_for_warm_after_idle() -> None:
    result = _refill_harness(
        r"""
  readyScript = [2];
  scheduleVoteRefresh({ target_version: 2, ranking_refresh_queued: false, ranking_idle_seconds: 3 });
  const immediate = { refills: refills.slice(), polls: polls.slice() };
  await flush();
  console.log(JSON.stringify({ immediate, refills, polls, waited: now }));
"""
    )
    # The stale deck refill happens at once; the ready-gated poll only after
    # the server's idle window (3s), then a second non-advancing refill.
    assert result["immediate"] == {"refills": [False], "polls": []}
    assert result["refills"] == [False, False]
    assert result["polls"] == [2]
    assert result["waited"] >= 3000
