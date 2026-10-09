**Result:** No finding blocks the two-line extraction-guard deployment. All findings are client-side TUI behavior or read-path timing, and none involves the extraction code. Coverage is partial (see end). Nothing was changed.

**Findings, highest priority first**

1. **Medium, proven in code:** Quitting can drop an in-flight vote. `app.py:2130` runs `workers.cancel_all()` before `api.close()` (`2127-2133`), which also cancels the `vote` worker (`app.py:1806-1834`). Trigger: press 1/2/3, then `q` before the POST returns. Effect: the story is hidden locally, the server never records the vote, and the story returns next session. Frequency unmeasured. Next step: on unmount, cancel only summary/refresh groups and await the vote worker with a short bound.

2. **Medium, plausible:** An unconfirmed vote can be wrong server-side. `api.py:239` sets a 10 s timeout, and `submit` never retries (`app.py:1819-1833`). If the server committed but the reply timed out, the story is restored locally and the server has the vote. Effect: it reappears until the next reload, and only the status text warns. Next step: after an unconfirmed vote, refetch the feed or reconcile that story.

3. **Low–medium, measured plus code:** A vote's reranked deck appears only on the 60 s poll. `app.py:1837-1839` sets `ready=False`, and `app.py:817` polls every 60 s (`1473-1509`). Measured: server rank median 15.1 s, p95 25.3 s, n=75 (`LIVE.json`). The ready endpoint took 6.2 ms locally. Effect: reorder can lag by up to about a minute plus rank time, with only "Vote saved." shown. Next step: poll `ready` every few seconds only while `ready=False`.

4. **Low–medium, plausible:** Counts changes clear every cached window. Any changed count bumps `counts_version` (`server.py:3820-3821`, every 600 s by `hot_refresh_interval_seconds`). The TUI then empties `self.feeds` (`app.py:1504-1507`). Effect: the next `d`/`D` window switch may show loading instead of the cached deck. Not timed. Next step: mark caches stale rather than clearing them, and time a window switch after a count change.

5. **Low, plausible:** Join and client timeouts match. The server join waits 150 s (`server.py:596`), and the client's generation timeout is 150 s (`api.py:240`). A joiner on a leader that runs past 150 s can get "didn't finish" (`server.py:2993`) at the same moment the client times out. Next step: give the server join a margin above the client timeout.

**Checked and not flagged:** Hot-refresh waits on ranking, but `rank_gate.py:22,47` caps that at 120 s, so there is no stall. Bindings (`app.py:678-702`) show no collisions. `r` forces a summary regeneration per press (`app.py:1741-1751`); that costs LLM quota, not latency, and is unmeasured.

**Scope and unknowns**
- Not read: the `Handler` lock/`_deck_for_user` region (~1930–2200), warm scheduler, TLDR stale-fallback and cache internals beyond spot checks, the narrow/wide CSS and `layout_panes`, and `reset_summaries`.
- No live UI, terminal, network, or timing was observed. Tests were not run.
- `LIVE.json` shows 1582 article-fetch failures, which was not analyzed.
- Known costs (first cold rerank, refit latency, ArcheShift36h archive delay) are excluded.
