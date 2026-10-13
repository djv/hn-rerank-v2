# HN Rerank terminal reader

Run `uvx hn-rerank` (Python 3.12+). Choose an existing profile link or create a
profile with your server URL, for example `https://your-host/hn/`.
`--server URL` selects a deployment; credentials from another server are never reused.
The server retains its existing access policy: installing the client does not grant access.

Keys: j/k next/previous story (list and zoom view); arrows scroll the focused
pane; Space pages the TLDR down from either view; d / D cycle the time window
forward / back (the last one picked opens next time); Tab changes focus; 1/2/3 vote
positive/neutral/negative and advance to the next story at once (votes are sent
in order in the background; a failed one puts its story back with an error);
u undoes the latest vote; o/c open article/comments (in Chrome, or in the
command in `HN_RERANK_BROWSER`, e.g. `surf`);
y copies the comments link (or the article link if there is none); a opens
Claude Code in a tmux pane split beside the reader (in your home directory) with the article and
comments links and a "dig deeper" prompt (`HN_RERANK_AGENT` overrides the
`claude` command; outside tmux the prompt is copied instead); r regenerates
the selected summary and checks its live points/comments, automatically
reloading if the discussion outgrew it; ? shows help; q quits. Both q and Ctrl+q
always exit within about three seconds: the first q blocks every key except q
and shows "Saving N votes… press q again to quit now" while votes finish,
with telemetry capped at about one second; a second q exits at once. Any vote
not confirmed is listed after the reader closes ("Vote not saved: …").
Ambiguous requests are never automatically retried.
Enter hides the article list and zooms the TLDR pane; Enter or Escape returns
to the list. Escape also closes help. Use the sort and window selectors (or h/l and d) to change
filters. Below 100 columns the headline list stacks above the summary.
Non-HN stories show their feed's name (AINews, r/sub) and no score or comment
count when the feed has none; the summary heading says which upvote a story
resembles ("Because you upvoted: …"). Press `w` for **Why this story**:
model signals that helped or hurt relative to a middle-of-pool rating,
related upvotes, feed ordering, and badge signals. Escape returns to the
summary. Model details require a freshly ranked deck from an updated server;
older servers still provide the original single-upvote attribution.

Summaries are cached for the session across sort changes. Up to four background
requests warm the next 20 stories, the previous three, and the first three
stories in each other sort for the current time window. Prefetch starts alongside
the selected summary and refills as you navigate or vote.

Missing TLDRs are generated for the next three stories, the previous three,
and the first three in each other sort. This uses the normal server generation
path and may incur provider usage; deeper stories only read existing caches.
`--prefetch N` adjusts forward cache depth (`0` disables all prefetch).
`--prefetch-generate N` adjusts forward generation depth and the other-sort
entry window (up to three); `0` keeps all background work cache-only.
Navigation reuses an in-flight prefetch instead of issuing a duplicate request.
Errors pause new background requests for a minute; requests already running may
finish. Empty, stale or partial responses are never cached. Rapid navigation
can still outrun generation, and changing the time window warms its own targets.

Selected-story transitions that remain selected for at least one second send
best-effort impression events to your configured server's existing interaction
ledger: story ID, zero-based list position, filters, feed version, time and a
random session ID. The reader also logs article/comment opens (`o`/`c`) at
once and dwell intervals per selection, with the same 1 s minimum and a
90 s idle cap (input idle ends the segment). Prefetches do not count;
failures never block reading. Dwell also ends when the terminal reports a
focus loss (in tmux this needs `focus-events on`); without such reports only
the idle cap bounds it, so it measures selection time, not attention. Events
are not sent to a separate analytics service and are not retried.

Configuration lives in platformdirs' user config directory (`hn-rerank/profile.json`),
outside uv's cache. The profile token is a credential: keep that file and profile links private.
Requests do not follow redirects or retry ambiguous votes. Refresh after a connection failure.
HTTP 5xx summary failures keep the story available. Press r, or move away and
back, to retry; an existing summary remains visible when its refresh fails.
Reading requires an online server; no local ML.

Local build: `uv build --project clients/tui`.
Local launch: `uvx --from /absolute/path/to/hn_rerank-0.1.0-py3-none-any.whl hn-rerank`.
If the package name cannot be registered, publish as `hn-rerank-tui` and run
`uvx --from hn-rerank-tui hn-rerank`.

## Reader appearance

The charcoal and ivory reader uses orange for the brand, active filters and
focus, with restrained blue/sage/amber accents in metadata and vote counts.
Bold key terms in a summary take the same orange, and single-marker emphasis
(italic) does too, so emphasis reads without extra chrome. A `>` marker
identifies the selected headline without relying on color. At 100 columns and
above, filter tabs sit over a one-third headline/two-thirds summary layout.
Smaller terminals use dropdowns and stack the list above the summary; both
panes stay visible and the summary scrolls in place. Enter zooms the TLDR pane
to full height at every terminal size,
even for short summaries. On wide terminals the pane is centered and capped
at 100 columns for comfortable reading. Enter or Escape restores the article
list. The footer shows the zoom and return keys for the current mode.
Press `r` to regenerate only the selected summary and background-check its
real points/comments. Visible counts update at once; when live comments
still exceed the refreshed summary's snapshot, `r` automatically reloads
that summary once per observed comment count, without a confirmation.
Background feed checks update counts silently; press `r` to refresh the
summary with the newer discussion. Nothing
else is cleared, refetched or reordered; hidden stories stay hidden.
Automatic refreshes keep using the server cache:
once a minute the reader checks for a newer deck (a regen, a vote from another
device, or the reranked deck after your vote) and reloads it; switching
sorts (`s`/`h`/`l`, selectors and tabs alike) navigates the shared window
feed immediately and fetches its latest existing version in the background
(one request in flight, at most one queued follow-up), keeping the open
story, its text and scroll usable while waiting.
Regeneration still respects provider limits and does not guarantee a longer summary.
Press `s` to cycle Recommended → Popular → Explore → Date; `h` / `l` step to
the previous / next sort.

Resize and focus changes preserve reading position. Run the offline preview from
the repository root with `uv run python -m clients.tui.tests.preview --headless`.

The reader follows the local clock: a light paper theme from 06:00 to 19:59,
the dark editorial theme otherwise (checked every minute; a theme picked from
the command palette holds until the next boundary). Unselected headlines are
dimmed, so the selected row (a lifted band plus the `>` marker) reads first. The
filter bar and a one-row docked footer share a tint slightly off the page color. The footer carries the current
filter's counts on the left (`2 shown · +0 ~0 −0`) and context-sensitive keys on
the right; when both do not fit, the keys shrink step by step down to `? help`,
then the status is cut with `…`. Failures appear there with a `✗` prefix and replace the counts until
the next successful refresh. Scrollbars follow the theme, and stories without a
usable timestamp simply omit the age.

Empty filters and feed failures get headed notices in the reading pane rather
than a bare line; the setup form is a bordered card on a dimmed scrim; the
reading pane caps at 100 columns with a hairline under the headline; votes
confirm in the status line.

### Queue sort and daily comparison

Queue plays each window's served Recommended cards, then Explore grouped as
Discover, Unsure (`uncertain`), Interest, and other badge kinds, then the first
five additional Popular stories. Duplicates and unavailable stories are skipped.
Queue keeps served order within each group and never shuffles. Voting advances
through the full queue; undo returns a story to its queue slot. Each window's
first shown feed freezes its queue for the session. Later refreshes append only
new Recommended cards at the end, retaining every original slot, origin, and
served rank. New Explore and Popular cards are not appended. Aged-out cards are
skipped; switching windows retains their snapshots, and reconnecting clears them.
At the end the reader shows **Caught up for this window**. The `s`, `h`, and `l` keys and sort
tabs/dropdown still reach Recommended, Popular, and Explore.

The startup sort alternates by UTC ordinal day: odd days open Queue, even days
open Recommended. The choice is made at launch, for the live comparison.

All four reading events (impression, dwell, article open, comments open) carry
`surface="queue"` and `queue_position` in Queue. `queue_position` is the slot in
the full frozen and extended queue, including voted and skipped stories.
`sort_mode` and `position` retain the origin view and that view's served rank.
For Explore cards in Queue, this is the index in the server's Explore order;
in the Explore view it is the index in the client-shuffled list.
Other sorts send `surface="views"` and
omit `queue_position`. Deploy the server's additive event-field support with
this client; older servers reject these events (reading remains best effort).
