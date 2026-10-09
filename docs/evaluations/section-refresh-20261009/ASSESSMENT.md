# Agent assessment of the comparison

Historical comparison before implementation. Statements below describe that
prototype stage; the approved production implementation and its checks are now
saved in PLAN.md and IMPLEMENTATION-CHECKS.txt.

The user requested automatic refresh when leaving sections with s/h/l,
asked to compare approaches, then explicitly selected **report comparison
first**. Production code is unchanged. [Muse's results](RESULTS.md) and
[reproducible prototypes](test_section_refresh.py) are preserved.

| Approach | Verified result | Tradeoff |
|---|---|---|
| Fetch existing feed on return | New counts/orders appear on return; one current-window GET per tested round trip | Request starts later, when the section is revisited |
| Background fetch on departure | New counts/orders appear after the background response; two current-window GETs per tested round trip, including the return switch | Starts earlier, but does more requests |
| Force a new ML rerank | No existing client operation for it; no live forced rerank was run | Would require server/API work; votes already request reranking |

Both prototypes keep the old deck usable while a response is held. The
synthetic cached-summary fixture stays consistent and sends no forced-summary
requests. It does not prove a quota guarantee when new stories/comments arrive:
normal selection/prefetch can then request summaries.

**Recommended fit for the requested behavior:** background GET on departure,
preserving local navigation and summary text, with one request in flight per
window and a bounded follow-up if needed. Coalescing is a proposed improvement,
not tested implementation. Fetch-on-return is the cheaper alternative.
Freshness here means the server's existing deck/counts, not new upstream data.

Corrections to the Muse recommendation:

- A does not universally dominate C: an earlier background request is a
  distinct benefit. Keeping the current behavior would not fulfill the user's
  stated auto-refresh request.
- The order/count-change test bumps A's deck after departure and C's deck
  before departure. This proves each can apply fresh data, not a fair timed
  performance comparison. No milliseconds-to-freshness were measured.
- Four instantaneous selector changes produced one GET in each prototype.
  That proves the queued-event guard for this fixture, not that real repeated
  keypresses or requests already in flight are coalesced. The existing refresh
  worker is exclusive and can cancel a prior fetch.
- The narrow/wide real-terminal-buffer checks are in FINDINGS.md; the
  prototypes use Textual Pilot and do not establish graphical paint latency,
  public HTTPS latency or physical keyboard responsiveness.

Eight controlled checks passed against exact laptop app.py source on the VPS;
Ruff, format and ty passed. No production behavior fix or deployment occurred.
