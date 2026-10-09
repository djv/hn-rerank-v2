**Codex verdict: acceptable; no concrete bug or blocker found in the reviewed code.**

- [server.py:423](/home/d/code/hn-rerank/server.py:423) returns before every extractor for empty or whitespace-only input. Nonblank input follows the existing extraction path. [server.py:540](/home/d/code/hn-rerank/server.py:540) preserves `status=200`, `error="empty_extraction"` and retryable semantics. This suppresses blank-input extractor logging; it neither fixes upstream 403s nor explains earlier blank responses.
- [tests/test_fetch.py:219](/home/d/code/hn-rerank/tests/test_fetch.py:219) adequately covers four combinations: empty/whitespace × direct 200/initial 403 followed by fallback 200. Assertions cover body, status, error, permanence and ERROR logging. They do not explicitly prove every extractor was skipped; the planned extractor-raises probe supplies that check and must remain labelled simulated.

The deployment scope is appropriate **given the supplied commit/diff evidence**: fast-forward specifically to `bf6e307`, preserve research artifacts and trial state, reuse profile151, and avoid uncached provider generation. Refresh HEAD/tree/index prerequisites immediately before merging; verify deployed HEAD and unchanged config hash afterward. Normal restart warmup/reranking is an acknowledged runtime effect.

Cache reads and a missing-story 404 do not validate uncached generation; report that omission explicitly. Scoped rollback must preserve unrelated state. Do not claim a naturally fetched blank was fixed without logs proving it.

Saved validation was supplied, not rerun. Deployment/VPS state was not independently checked. This is the Codex review only; no additional reviewers were invoked.