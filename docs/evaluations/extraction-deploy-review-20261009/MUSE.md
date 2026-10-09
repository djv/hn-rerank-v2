**Verdict: approve — safe to deploy as scoped.**

Guard `server.py:423-424` (`if not html.strip(): return None`) is correct and minimal: short-circuits before all four extractors, avoids trafilatura empty-HTML noise. Fetch semantics preserved: `server.py:540` returns `status=200, error="empty_extraction"` with default `permanent=False` (`server.py:294`), so retryable as claimed. No blocker found.

**Adequacy of 4 cases:** `tests/test_fetch.py:218-243` parametrizes `body in ["", " \n\t "] × fallback in [False, True]` — covers direct-200 blank/whitespace plus 403→`guarded_urllib_fetch` fallback blank/whitespace. Assertions on `body None/status 200`/`empty_extraction`/`not permanent`/no ERROR logs verify both paths and the logging motive. Sufficient for a 2-line guard; NUL-byte (`non_html`, `server.py:501,517`) is a separate path correctly out of scope.

**Deployment scope:** appropriately tight — ff-only to `bf6e307` (avoids accidentally taking `origin/main 0242b84`), rollback tag at `be965fd`, config-hash verify, restart `hn_rewrite.service` with bounded journal scan, smoke via dashboard/cache-hit/missing-story 404 with no provider spend, simulated (extractor-raise) blank probe honestly labelled, profile-151 cookie without token exposure, scoped revert on failure. Watch: keep untracked research artifacts out of the merge/commit.
