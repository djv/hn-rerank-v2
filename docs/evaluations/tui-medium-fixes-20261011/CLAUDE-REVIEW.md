# Independent review: TUI medium fixes (quit drain + 5xx transient)

## Verdict

**Approve with one small fix.** The drain design holds up. It polls `is_finished`, so it never awaits `Worker.wait` and never cancels a vote. It snapshots the in-flight vote workers synchronously, and it blocks vote/undo in both `check_action` and the action bodies. It refuses to exit unless every snapshotted worker ended in `SUCCESS` and the failure counter did not move. I found no way for normal quit to lose or reorder a queued vote/undo. Votes still go through the FIFO `vote_lock`, and workers start in the order they were created. The 5xx → `TransientError` change goes through the existing "Summary unavailable / press r" branch, and 4xx still hides the story. I found one introduced Low defect and a few minor gaps, listed below.

## Findings

### 1. Low (introduced): `finally` turns the quit guard back on after a successful `exit()`
`clients/tui/src/hn_rerank/app.py:2628-2630`. `self.exit()` only posts `ExitApp`. Right after that, the `finally` sets `_quit_requested = False`. Until `ExitApp` is processed and `on_unmount` runs (`app.py:2632-2635`), `check_action` (`app.py:940`) and `action_vote`/`action_undo` (`app.py:2216`, `2227`) let new votes through again.
- **Repro:** queue one held vote, press `q`, and send a vote key (e.g. through the pilot) right after the vote is released, before the app has finished shutting down.
- **Consequence:** the new vote hides the story and starts a `submit` worker. `on_unmount`'s `workers.cancel_all()` then cancels it, possibly after the POST was already sent. That is the exact "vote lost on quit" class this patch targets, though the window is small. A second `q` in the same window also starts a pointless second drain worker.
- **Fix:** clear the flag only on the return paths that keep the reader open (timeout, unconfirmed). Leave it set after `self.exit()`.

### 2. Low (introduced UX): the quit error replaces the specific vote error
`app.py:2620-2626`. When a vote fails, `submit` shows a specific message: which story, whether it is back, and the exception text (`app.py:2280-2285`). The quit worker then overwrites it with a generic "Votes were not confirmed…". With several queued votes, the user can no longer tell which one failed. The test (`test_medium_reliability.py:194`) only checks the generic text. Suggestion: keep the earlier text, e.g. append "press q to quit" to it.

The reverse also happens. `submit`'s own "Saving vote…"/"Vote saved." overwrite "Saving pending votes before quitting…" (`app.py:2607`). If a vote finishes after a timeout, "Vote saved." also replaces the "still saving… press q" message. This is cosmetic: the reader stays open and the state stays correct.

### 3. Info (pre-existing, outside the targeted path): a profile change cancels queued votes
- `setup()` (`app.py:1055`) cancels the whole `vote` group and leaves the optimistic hides in place. If the first queued vote gets `InvalidProfile` while the app is draining, the later votes are cancelled. The drain correctly refuses to exit, because the counter moved and the workers are `CANCELLED`. But its status goes to the screen underneath the Setup modal, where nobody sees it. Setup's own Quit button (`app.py:601`) and `connected(None)` (`app.py:1067-1069`) call `exit()` directly with no drain. That is harmless only because the votes were already cancelled.
- `submit` does not count an `APIError` when `api is not self.api` (`app.py:2273`), and the worker still ends in `SUCCESS`. The only way to swap the API is through `setup()`, which cancels the group first, so I could not reach this during a drain.

Both behaviours predate this patch. Track them separately.

### 4. Info (introduced behaviour change, acceptable): background cache 5xx no longer falls back
`speculate` re-raises `TransientError` (`app.py:1649-1650`). Before, a 500/502/504 from `api/tldr-cache` was a plain `APIError`: speculate returned `None`, and `await_summary` tried `tldr-detail` directly (`app.py:1442-1446`). Now that shared task's error reaches the selected story as the transient "Summary unavailable" message, and the direct attempt is skipped. 503 already behaved this way, and the story is no longer hidden, so this fits the stated intent. Just note that one automatic fallback is gone for 500/502/504.

### 5. Info (pre-existing): `check_action` blocks every action during setup or Input focus
`check_action` (`app.py:938-939`) returns False for all actions while `setting_up` is true or an `Input` has focus, including App-level ctrl+q. The patch did not change this.

## Test quality

The new tests use the real `Reader` and the real `API`/`httpx.AsyncClient` over `MockTransport`. They hold the first request on an `asyncio.Event`, press real `q`/`ctrl+q`, and check observable outcomes: exit through the `exit()` override, `api.close()`, the server-side `saved` order, attempt counts, server-side cancellation, and `rated`. That meets the brief.

Gaps:
- Checks like `"still saving" in app.status_text` and `"not confirmed"` match UI copy, which is borderline "implementation strings". They are acceptable but brittle.
- `test_quit_timeout…` and the failure/cancel tests call `app.action_quit()`/`app.action_undo()` directly. So the `check_action` gate is not covered by a key press during the drain.
- No test covers finding 1, or `InvalidProfile` during a drain.
- `monkeypatch.setattr(..., raising=False)` would quietly fall back to the 5 s default if the constant were renamed. The test would then fail on the status assertion, so it is not a false pass.
- Changing `FailSummaryServer` from 500 to 404 (`test_client.py:330`) correctly keeps the permanent-hide coverage on a 4xx.

## Verification and limits

I reviewed the source only, against the files in the bundle and `SOURCE.diff`. Textual is not installed here, and the brief rules out heavy installs, so I **did not run** any tests. I did not check the implementer's 53-passed result. The statements about Textual 8.2.8 internals come from known behaviour: `exit()` posts a message, and `run_worker` registers the worker synchronously. I did not re-read the pinned source. I made no production requests and modified no files under review.
