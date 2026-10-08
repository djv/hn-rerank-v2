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
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

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
        if quote is None and ch == "/" and _starts_regex(text, i):
            i = _skip_regex(text, i)
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


def _starts_regex(text: str, i: int) -> bool:
    """A ``/`` begins a regex literal when it follows an operator, an opening
    bracket or a keyword-like position rather than a value."""
    j = i - 1
    while j >= 0 and text[j] in " \t\n":
        j -= 1
    return (
        j < 0
        or text[j] in "(,=:[!&|?{};+-*%<>~^"
        or text[max(0, j - 5) : j + 1].endswith("return")
    )


def _skip_regex(text: str, i: int) -> int:
    """Index just past the regex literal starting at ``text[i] == "/"``."""
    i += 1
    in_class = False
    while text[i] != "\n":
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif ch == "/" and not in_class:
            i += 1
            while text[i].isalpha():  # flags
                i += 1
            return i
        i += 1
    raise ValueError(f"unterminated regex literal at {i}")


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


_HOSTILE_FRAGMENTS = [
    "<img src=x onerror=alert(1)>",
    "<script>alert(1)</script>",
    "<svg/onload=alert(1)>",
    '"><b onmouseover=alert(1)>',
    "[t](javascript:alert(1))",
    "[t]( JaVaScRiPt:alert(1))",
    "[t](&#106;avascript:alert(1))",
    "[t](data:text/html,<b>x</b>)",
    '[t](https://ok.example/" onmouseover="alert(1))',
    "![a](https://img.example/p.png)",
    "[ok](https://ok.example/a?b=1&c=2)",
    "**bold** _em_ `a < b` ~~s~~",
    "```js\n<b>x</b>\n```",
    "- item <i>x</i>\n- two",
    "### Heading <u>x</u>",
    "> quote",
    "Label: value & more",
    "&lt;already&gt;",
    "\n\n",
]
_ALLOWED_TAGS = {
    "a", "br", "code", "em", "h1", "h2", "h3", "h4", "h5", "h6",
    "hr", "li", "ol", "pre", "s", "strong", "ul",
}  # fmt: skip


class _TagAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.problems: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _ALLOWED_TAGS:
            self.problems.append(f"tag <{tag}>")
        for name, value in attrs:
            if tag == "a" and name == "href":
                if not re.match(r"https?://", value or "", re.IGNORECASE):
                    self.problems.append(f"href {value!r}")
            elif tag == "a" and name in {"target", "rel"}:
                continue
            elif tag in {"pre", "code"} and name == "class":
                continue
            else:
                self.problems.append(f"attr {name}= on <{tag}>")


@settings(max_examples=12, deadline=None)
@given(
    docs=st.lists(
        st.lists(st.sampled_from(_HOSTILE_FRAGMENTS), min_size=1, max_size=6).map(
            lambda parts: " ".join(parts)
        ),
        min_size=20,
        max_size=40,
    )
)
def test_tldr_markdown_output_only_contains_safe_markup(docs: list[str]) -> None:
    """Whatever an LLM echoes from a hostile page, the rendered TLDR holds
    only whitelisted tags, no event-handler attributes, and http(s) links."""
    script = _inline_script()
    start = script.index("    const TAGS=")
    end = script.index("    function styleTldrLabels(")
    outputs = run_node(
        script[start:end]
        + f"\nconsole.log(JSON.stringify({json.dumps(docs)}.map(parseSimpleMarkdown)));\n"
    )
    for doc, html in zip(docs, outputs, strict=True):
        audit = _TagAudit()
        audit.feed(html)
        assert audit.problems == [], (doc, html, audit.problems)


# The dashboard client: feed, views, votes, freshness, summaries -------------

_CLIENT_FUNCTIONS = (
    "parseFeed",
    "feedPath",
    "viewKey",
    "viewOrder",
    "restoreStory",
    "showFeed",
    "setFeed",
    "prefetchWindows",
    "render",
    "activeCard",
    "setActive",
    "move",
    "updateStatus",
    "setFilter",
    "cycleSort",
    "cycleWindow",
    "tintByRank",
    "feedCard",
    "patchCardMetadata",
    "patchCounts",
    "queueInteraction",
    "routePrefix",
    "apiPath",
    "request",
    "reload",
    "pollFeedVersion",
    "setVoteCounts",
    "adjustVoteCount",
    "applyVote",
    "applyUndo",
    "vote",
    "undo",
    "submit",
    "summaryFor",
    "track",
    "lookupSummary",
    "generateSummary",
    "keepSummary",
    "loadSummary",
    "withPrefetchSlot",
    "prefetchUpcoming",
)

_CLIENT_DOM = r"""
function el(tag) {
  const node = {
    tag, dataset: {}, children: [], className: '', textContent: '', isConnected: false,
    style: { setProperty() {} }, tabIndex: 0,
    classList: { set: new Set(), add(c) { this.set.add(c); }, remove(c) { this.set.delete(c); },
                 toggle(c, on) { on ? this.set.add(c) : this.set.delete(c); } },
    appendChild(child) { this.children.push(child); return child; },
    replaceChild(child, old) { const i = this.children.indexOf(old); if (i < 0) throw new Error('missing child'); this.children[i] = child; return old; },
    remove() { this.isConnected = false; },
    focus() {}, querySelector() { return null; },
    get innerHTML() { throw new Error('cards must not use innerHTML'); },
    set innerHTML(v) { throw new Error('cards must not use innerHTML'); },
  };
  return node;
}
const counts = { up: el('span'), neutral: el('span'), down: el('span') };
const queueLoadingEl = el('p'), statusEl = el('div');
const storiesContainer = el('div');
storiesContainer.insertBefore = card => { card.isConnected = true; };
const document = {
  createElement: el, visibilityState: 'visible',
  querySelector: sel => { const m = sel.match(/data-vote-count="(\w+)"/); return m ? counts[m[1]] : null; },
  getElementById: () => null,
};
const window = { location: { pathname: '/' }, setTimeout, clearTimeout,
                 sessionStorage: { getItem: () => 'session', setItem() {} } };
const voteBar = el('div'), sidePanel = el('aside');
const sortTabs = [], windowSelect = null;
const toasts = [];
function showToast(message, variant) { toasts.push(variant + ': ' + message); }
function showSummary() {}
function focusActiveCard() {}
function beginInteraction() {}
function endInteraction() {}
function enterFrom() {}
// fetch: every call is recorded and waits for the scenario to answer it.
const calls = [];
function fetch(url, init = {}) {
  return new Promise((resolve, reject) => {
    calls.push({ url, method: init.method || 'GET', body: init.body ? JSON.parse(init.body) : null,
                 resolve: (status, data) => resolve({ ok: status < 300, status, json: async () => data,
                                                      headers: { get: () => null } }),
                 reject });
  });
}
const flush = async () => { for (let i = 0; i < 20; i++) await new Promise(r => setImmediate(r)); };
const pending = path => calls.filter(c => !c.done && c.url.includes(path));
const answer = async (path, status, data) => {
  const call = pending(path)[0];
  if (!call) throw new Error('no pending ' + path + ' in ' + JSON.stringify(calls.map(c => c.url)));
  call.done = true; call.resolve(status, data); await flush();
};
const fail = async path => {
  const call = pending(path)[0]; call.done = true; call.reject(new TypeError('offline')); await flush();
};
function story(id, extra = {}) {
  return { id, title: `Story ${id}`, article_url: '', comments_url: '', source: 'hn', points: 1,
           comments: 0, time: 1000 + id, rank_score: 100 - id, badges: [], badge_details: [],
           best_match_title: '', source_label: 'HN', domain: '', enriched: false, ...extra };
}
function makeFeed(ids, extra = {}) {
  const orders = { recommended: ids.slice(), popular: ids.slice().reverse(), explore: ids.slice() };
  return { api_version: 2, window: '1w', stories: ids.map(id => story(id)), orders,
           feedback_counts: { up: 0, neutral: 0, down: 0 }, version: 5, target_version: 5,
           ready: true, ...extra };
}
const FEED = '/api/feed?window=';
const state = () => ({ active: activeId, visible: visible.map(s => s.id), rated: [...rated],
                       ready: feed.ready, target: feed.target_version,
                       counts: ['up', 'neutral', 'down'].map(a => Number(counts[a].textContent)) });
console.error = () => {}; console.debug = () => {};
"""


def _client_harness(scenario: str) -> Any:
    script = _inline_script()
    constants = script[
        script.index("    const VIEW_LIMIT = ") : script.index(
            "    let activeInteraction = null;\n"
        )
    ]
    first = "    const FIRST = Symbol('first');\n"
    assert first in script
    source = (
        _CLIENT_DOM
        + constants
        + "let activeInteraction = null;\n"
        + first
        + js_functions(script, *_CLIENT_FUNCTIONS)
        + "\n(async () => {\n"
        + scenario
        # Request deadlines and unanswered fetches would keep Node alive.
        + "\n})().catch(e => { console.log(JSON.stringify({error: String(e.stack)})); })"
        + ".finally(() => process.exit(0));\n"
    )
    result = run_node(source)
    assert not (isinstance(result, dict) and "error" in result), result
    return result


def test_votes_hide_at_once_go_out_in_order_and_a_failure_reverts_only_its_story() -> (
    None
):
    result = _client_harness(r"""
    setFeed(makeFeed([1, 2, 3, 4]));
    const start = state();
    vote('up'); vote('down');          // two stories, before the server answers
    await flush();
    const hidden = state();
    const sentFirst = pending('/api/feedback').map(c => c.body);
    await fail('/api/feedback');       // the first vote is lost
    const afterFail = state();
    const sentSecond = pending('/api/feedback').map(c => c.body);
    await answer('/api/feedback', 200, { ok: true, target_version: 7 });
    console.log(JSON.stringify({ start, hidden, sentFirst, afterFail, sentSecond,
                                 end: state(), toasts }));
    """)
    assert result["start"]["active"] == 1
    assert result["hidden"]["visible"] == [3, 4] and result["hidden"]["active"] == 3
    assert result["hidden"]["counts"] == [1, 0, 1]
    # One request at a time, in the order voted.
    assert result["sentFirst"] == [{"story_id": 1, "action": "up"}]
    assert result["sentSecond"] == [{"story_id": 2, "action": "down"}]
    # The failed story is back (and shown), its count undone; never retried.
    assert result["afterFail"]["visible"] == [1, 3, 4]
    assert result["afterFail"]["active"] == 1
    assert result["afterFail"]["counts"] == [0, 0, 1]
    assert len(result["toasts"]) == 1 and "not confirmed" in result["toasts"][0]
    # A saved vote marks the deck stale until the reranked one lands.
    assert result["end"]["ready"] is False and result["end"]["target"] == 7


def test_undo_puts_the_story_back_where_the_server_ranks_it() -> None:
    result = _client_harness(r"""
    setFeed(makeFeed([1, 2, 3, 4]));
    move(1);                           // on story 2
    vote('neutral');
    await flush();
    await answer('/api/feedback', 200, { ok: true, target_version: 6 });
    // A reload whose (stale) deck no longer has story 2.
    reload();
    await answer(FEED + '1w', 200, makeFeed([1, 3, 4], { version: 5, target_version: 6, ready: false }));
    const reloaded = state();
    undo();
    await flush();
    const undone = state();
    const sent = pending('/api/feedback').map(c => c.body);
    await fail('/api/feedback');       // the undo is lost: hidden again
    console.log(JSON.stringify({ reloaded, undone, sent, end: state() }));
    """)
    assert result["reloaded"]["visible"] == [1, 3, 4]
    assert result["undone"]["visible"] == [1, 2, 3, 4]
    assert result["undone"]["active"] == 2
    assert result["undone"]["counts"] == [0, 0, 0]
    assert result["sent"] == [{"story_id": 2, "action": "clear"}]
    assert result["end"]["visible"] == [1, 3, 4] and result["end"]["counts"] == [
        0,
        1,
        0,
    ]


@pytest.mark.parametrize(
    ("ready", "answer_ready", "current", "reloads"),
    [
        (True, True, 5, False),  # current deck, same version
        (True, True, 6, True),  # a regen or a vote elsewhere
        (False, False, 7, False),  # still ranking
        (False, True, 7, True),  # the reranked deck landed
    ],
)
def test_poller_reloads_only_for_a_newer_deck(
    ready: bool, answer_ready: bool, current: int, reloads: bool
) -> None:
    result = _client_harness(rf"""
    setFeed(makeFeed([1, 2], {{ ready: {str(ready).lower()}, target_version: {5 if ready else 7} }}));
    const poll = pollFeedVersion();
    const asked = pending('/api/ranking-ready').map(c => c.url);
    await answer('/api/ranking-ready', 200,
                 {{ ok: true, ready: {str(answer_ready).lower()}, current_version: {current} }});
    await poll;
    console.log(JSON.stringify({{ asked, reloading: pending(FEED + '1w').length }}));
    """)
    wanted = 5 if ready else 7
    assert result["asked"] == [f"/api/ranking-ready?min_version={wanted}"]
    assert result["reloading"] == (1 if reloads else 0)


def test_one_summary_request_per_story_and_lookahead_only_reads_the_cache() -> None:
    result = _client_harness(r"""
    setFeed(makeFeed([1, 2, 3, 4, 5, 6]));   // active 1: next 3 may generate, 5-6 cache only
    await flush();
    const first = calls.map(c => c.method + ' ' + c.url.replace(/\d+$/, 'N'));
    loadSummary(1);                           // a second ask joins the running request
    const joined = pending('/api/tldr-detail').length;
    await answer('/api/tldr-detail', 200, { ok: true, tldr: 'one' });
    await answer('/api/tldr-cache/', 204, null);   // story 5: not generated yet
    const missed = summaryFor(5, { generate: false });
    const lookedUpAgain = pending('/api/tldr-cache/5').length;
    await missed;
    const next = pending('/api/tldr-cache/').map(c => c.url);  // the freed slot's turn
    console.log(JSON.stringify({ first, joined, lookedUpAgain, next, one: summaries.get(1) }));
    """)
    # The open card, then 3 generations + 1 lookup: 4 prefetch slots.
    assert result["first"].count("POST /api/tldr-detail") == 4
    assert result["first"].count("GET /api/tldr-cache/N") == 1
    assert result["next"] == ["/api/tldr-cache/6"]
    assert result["joined"] == 4
    assert result["lookedUpAgain"] == 0  # a miss isn't asked again within a minute
    assert result["one"] == {"text": "one", "provisional": False}


def test_a_failed_background_generation_pauses_prefetch_generation() -> None:
    result = _client_harness(r"""
    setFeed(makeFeed([1, 2, 3]));
    await flush();
    await answer('/api/tldr-detail', 200, { ok: true, tldr: 'active' });
    const second = pending('/api/tldr-detail').find(c => c.body.story_id === 2);
    second.done = true;
    second.resolve(429, { error: 'Summaries are busy.', retry_after: 5 });
    await flush();
    const paused = prefetchPausedUntil > Date.now();
    const retry = retryAt.get(2) - Date.now();
    console.log(JSON.stringify({ paused, retry, message: summaryErrors.get(2) }));
    """)
    assert result["paused"] is True
    assert 4000 < result["retry"] <= 5000
    assert result["message"] == "Summaries are busy."


def test_views_cap_backfill_start_at_the_head_and_keep_the_explore_order() -> None:
    result = _client_harness(r"""
    const ids = Array.from({ length: 20 }, (_, i) => i + 1);
    setFeed(makeFeed(ids));
    const capped = visible.map(s => s.id);
    move(3);                                  // on story 4
    vote('up');
    const backfilled = { active: activeId, visible: visible.map(s => s.id) };
    setFilter('sort', 'popular');
    const popular = { active: activeId, first: visible[0].id };
    setFilter('sort', 'explore');
    const explore = viewOrder('1w:explore');
    reload();
    await answer(FEED + '1w', 200, makeFeed(ids.concat([21]), { version: 6, target_version: 6 }));
    const exploreAfter = viewOrder('1w:explore');
    console.log(JSON.stringify({ capped, backfilled, popular, explore, exploreAfter }));
    """)
    match = re.search(r"const VIEW_LIMIT = (\d+);", _inline_script())
    assert match is not None
    limit = int(match.group(1))
    assert result["capped"] == list(range(1, limit + 1))
    assert result["backfilled"]["active"] == 5
    assert result["backfilled"]["visible"] == [1, 2, 3, *range(5, limit + 2)]
    # A new view starts at its first story.
    assert result["popular"] == {"active": 20, "first": 20}
    # Explore keeps placed stories where they were; new ones go after.
    assert sorted(result["explore"]) == list(range(1, 21))
    assert result["exploreAfter"] == [*result["explore"], 21]


def test_cards_are_built_from_text_only() -> None:
    result = _client_harness(r"""
    const card = feedCard(story(9, { title: '<img src=x onerror=alert(1)>', best_match_title: '<b>x</b>',
                                     domain: '<i>d</i>', comments_url: 'https://news.ycombinator.com/item?id=9' }), 5);
    const text = node => (node.textContent || '') + node.children.map(text).join('');
    console.log(JSON.stringify({ text: text(card), id: card.dataset.storyId, version: card.dataset.dashboardVersion }));
    """)
    assert "<img src=x onerror=alert(1)>" in result["text"]
    assert "Because you upvoted: <b>x</b>" in result["text"]
    assert result["id"] == "9" and result["version"] == "5"


def test_cards_render_and_log_all_applicable_badges() -> None:
    result = _client_harness(r"""
    globalThis.scheduleInteractionFlush = () => {};
    setFeed(makeFeed([9, 10]));
    const badge = kind => ({ kind, icon: '*', label: kind, tooltip: '' });
    const kinds = ['interest', 'hot', 'top', 'talk'];
    const card = feedCard(story(9, { badge_details: kinds.map(badge) }), 5);
    queueInteraction('impression', card);
    queueInteraction('impression', feedCard(story(10), 5));
    const rendered = card.children[0].children.filter(n => n.className.startsWith('badge badge--'));
    console.log(JSON.stringify({ rendered: rendered.map(n => n.textContent),
                                events: interactionEvents.map(e => [e.story_id, e.badges]) }));
    """)
    assert result["rendered"] == ["* interest", "* hot", "* top", "* talk"]
    assert result["events"] == [[9, ["interest", "hot", "top", "talk"]], [10, []]]


def test_why_story_panel_uses_view_and_literal_evidence() -> None:
    result = _client_harness(r"""
    currentSort = 'popular';
    const card = feedCard(story(9, { related_upvotes: ['<b>Up A</b>', 'Up B'],
      badge_details: [{kind: 'hot', icon: '*', label: 'Hot', tooltip: 'Rising fast'}] }), 5);
    const panel = card.children.find(n => n.tag === 'details');
    const text = node => (node.textContent || '') + node.children.map(text).join(' ');
    const fallback = feedCard(story(10, {best_match_title: 'Legacy upvote'}), 5);
    console.log(JSON.stringify({text: text(panel), summary: panel.children[0].tag,
      fallback: text(fallback), empty: text(feedCard(story(11), 5))}));
    """)
    assert result["summary"] == "summary"
    assert "points and age" in result["text"]
    assert "<b>Up A</b> Up B" in result["text"]
    assert "Hot: Rising fast" in result["text"]
    assert "not a complete explanation" in result["text"]
    assert "Legacy upvote" in result["fallback"]
    assert "No close upvote match" in result["empty"]


def test_why_story_refreshes_factors_and_feed_reason_without_closing() -> None:
    result = _client_harness(r"""
    const s = story(9, {ranking_factors: ['Helped: Content', 'Hurt: Words']});
    const card = feedCard(s, 5); cardEls.set(9, card);
    const panel = () => card.children.find(n => n.tag === 'details');
    const text = node => (node.textContent || '') + node.children.map(text).join(' ');
    panel().open = true;
    const before = text(panel());
    currentSort = 'popular'; patchCardMetadata(s, 6);
    const popular = text(panel());
    currentSort = 'recommended';
    patchCardMetadata({...s, ranking_factors: ['Hurt: Content']}, 7);
    console.log(JSON.stringify({before, popular, after: text(panel()), open: panel().open}));
    """)
    assert "Helped: Content" in result["before"] and "Hurt: Words" in result["before"]
    assert (
        "points and age" in result["popular"] and "Hurt: Words" not in result["popular"]
    )
    assert (
        "Hurt: Content" in result["after"] and "Helped: Content" not in result["after"]
    )
    assert result["open"] is True


def test_keys_match_the_terminal_client() -> None:
    """The key map runs the same actions as the TUI's bindings (plus the
    web-only f)."""
    script = _inline_script()
    start = script.index("    const KEY_ACTIONS = {")
    keymap = script[start : script.index("    };\n", start) + len("    };\n")]
    result = run_node(
        r"""
const done = [];
const record = name => (...args) => done.push([name, ...args]);
const move = record('move'), vote = record('vote'), undo = record('undo');
const openStoryUrl = record('open'), copyLink = record('copy'), reload = record('reload');
const cycleSort = record('sort'), cycleWindow = record('window'), showHelp = record('help');
const togglePanel = record('fullscreen');
const sidePanel = { classList: { toggle: () => done.push(['panel']) } };
"""
        + keymap
        + r"""
const out = {};
for (const key of Object.keys(KEY_ACTIONS)) { done.length = 0; KEY_ACTIONS[key](); out[key] = done[0]; }
console.log(JSON.stringify(out));
"""
    )
    assert result == {
        "j": ["move", 1],
        "k": ["move", -1],
        "1": ["vote", "up"],
        "2": ["vote", "neutral"],
        "3": ["vote", "down"],
        "u": ["undo"],
        "o": ["open", "article"],
        "c": ["open", "comments"],
        "y": ["copy"],
        "r": ["reload", {"manual": True}],
        "s": ["sort", 1],
        "l": ["sort", 1],
        "h": ["sort", -1],
        "d": ["window", 1],
        "D": ["window", -1],
        "b": ["panel"],
        "?": ["help"],
        "f": ["fullscreen"],
    }


def test_window_switch_uses_the_cache_and_prefetches_neighbours_once_per_version() -> (
    None
):
    result = _client_harness(r"""
    setFeed(makeFeed([1, 2, 3]));
    await flush();
    // One background request at a time: 1w's neighbours, 1m then 1d.
    const firstAsk = pending(FEED).map(c => c.url);
    await answer(FEED + '1m', 200, makeFeed([4, 5], { window: '1m' }));
    const secondAsk = pending(FEED).map(c => c.url);
    await answer(FEED + '1d', 200, makeFeed([6], { window: '1d' }));
    const idle = pending(FEED).length;
    // A cached window shows at once, from its first story, without asking.
    setFilter('window', '1m');
    const shown = { window: currentWindow, visible: visible.map(s => s.id), active: activeId };
    // ... and its own missing neighbour (archive) is fetched in the background.
    const thirdAsk = pending(FEED).map(c => c.url);
    // Back to 1w, then 1m again: nothing refetched for this version.
    setFilter('window', '1w');
    setFilter('window', '1m');
    const noRefetch = pending(FEED).map(c => c.url);
    console.log(JSON.stringify({ firstAsk, secondAsk, idle, shown, thirdAsk, noRefetch,
                                 feedWindow: feed.window }));
    """)
    assert result["firstAsk"] == ["/api/feed?window=1m"]
    assert result["secondAsk"] == ["/api/feed?window=1d"]
    assert result["idle"] == 0
    assert result["shown"] == {"window": "1m", "visible": [4, 5], "active": 4}
    assert result["thirdAsk"] == ["/api/feed?window=archive"]
    assert result["noRefetch"] == ["/api/feed?window=archive"]
    assert result["feedWindow"] == "1m"


def test_an_uncached_window_loads_in_front_and_a_late_prefetch_never_replaces_it() -> (
    None
):
    result = _client_harness(r"""
    setFeed(makeFeed([1, 2, 3]));
    await flush();                             // background: 1m in flight
    setFilter('window', '1m');                 // not cached yet: load it in front
    const loading = { visible: visible.map(s => s.id), status: queueLoadingEl.textContent,
                      asks: pending(FEED).map(c => c.url) };
    // The background answer lands first: dropped, the window is still loading.
    await answer(FEED + '1m', 200, makeFeed([7], { window: '1m' }));
    const afterLate = { visible: visible.map(s => s.id), loading: Boolean(feed.loading) };
    await answer(FEED + '1m', 200, makeFeed([4, 5], { window: '1m' }));
    console.log(JSON.stringify({ loading, afterLate, end: state(), window: feed.window }));
    """)
    assert result["loading"]["visible"] == []
    assert result["loading"]["status"] == "Loading more stories…"
    assert result["loading"]["asks"] == ["/api/feed?window=1m"] * 2
    assert result["afterLate"] == {"visible": [], "loading": True}
    assert result["end"]["visible"] == [4, 5] and result["window"] == "1m"


def test_a_vote_or_a_newer_deck_drops_background_windows_from_before_it() -> None:
    result = _client_harness(r"""
    setFeed(makeFeed([1, 2, 3]));
    await flush();                             // background: 1m in flight
    vote('up');                                // story 1
    await flush();
    await answer('/api/feedback', 200, { ok: true, target_version: 7 });
    // The 1m answer was asked before the vote was saved: dropped.
    await answer(FEED + '1m', 200, makeFeed([1, 4], { window: '1m' }));
    const afterVote = { cached: [...feeds.keys()], ready: feed.ready, target: feed.target_version };
    // The poller's reranked deck (a newer version) replaces every window.
    const oldAsk = pending(FEED).map(c => c.url);   // the next neighbour, still version 5
    reload();
    await answer(FEED + '1w', 200, makeFeed([2, 3], { version: 7, target_version: 7 }));
    // An answer for an older version never comes back into the cache.
    await answer(FEED + '1d', 200, makeFeed([9], { window: '1d', version: 5, target_version: 5 }));
    const newAsk = pending(FEED).map(c => c.url);   // neighbours again, for version 7
    await answer(FEED + '1m', 200, makeFeed([4], { window: '1m', version: 7, target_version: 7 }));
    console.log(JSON.stringify({ afterVote, oldAsk, newAsk, cached: [...feeds.keys()].sort(),
                                 end: state() }));
    """)
    assert result["afterVote"] == {"cached": ["1w"], "ready": False, "target": 7}
    assert result["oldAsk"] == ["/api/feed?window=1d"]
    assert result["newAsk"] == ["/api/feed?window=1m"]
    assert result["cached"] == ["1m", "1w"]
    assert result["end"]["visible"] == [2, 3] and result["end"]["ready"] is True


def test_votes_and_undo_follow_a_story_across_windows() -> None:
    result = _client_harness(r"""
    setFeed(makeFeed([1, 2, 3]));
    await flush();
    await answer(FEED + '1m', 200, makeFeed([2, 4], { window: '1m' }));
    await answer(FEED + '1d', 200, makeFeed([5], { window: '1d' }));
    move(1);                                   // story 2, in 1w and 1m
    vote('up');
    vote('down');                              // story 3, right after
    setFilter('window', '1m');
    const inMonth = visible.map(s => s.id);    // voted stories are hidden everywhere
    undo();                                    // story 3 (voted in 1w) comes back there
    undo();                                    // story 2 comes back in 1m too
    const undone = visible.map(s => s.id);
    setFilter('window', '1w');
    const inWeek = visible.map(s => s.id);
    await flush();
    const sent = calls.filter(c => c.url.includes('/api/feedback')).map(c => c.body);
    console.log(JSON.stringify({ inMonth, undone, inWeek, sent, counts: state().counts }));
    """)
    assert result["inMonth"] == [4]
    assert result["undone"] == [2, 4]
    assert result["inWeek"] == [1, 2, 3]
    assert result["counts"] == [0, 0, 0]
    # Sent one at a time in the order made; the first is still in flight.
    assert result["sent"] == [{"story_id": 2, "action": "up"}]


def test_popular_gravity_clock_matches_the_server() -> None:
    from clients.tui.src.hn_rerank.models import GRAVITY_TIME_SCALE as TUI_SCALE
    from pipeline.ranking import GRAVITY_TIME_SCALE

    line = next(
        line
        for line in _inline_script().splitlines()
        if "const GRAVITY_TIME_SCALE" in line
    )
    web = run_node(line + "\nconsole.log(JSON.stringify(GRAVITY_TIME_SCALE));")
    assert web == GRAVITY_TIME_SCALE == TUI_SCALE
