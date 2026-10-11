# Flattened-footer search correction — 2026-10-11

The user chose Fix medium after the independent
[Claude cloud final-backend review](https://claude.ai/code/session_01FSGnkwkmLWpEvhfxstJQeg).
Only the reproduced footer-search issue is addressed. The two low findings
(feedback-query overhead and Codex-specific cache/prefetch coverage) remain
unchanged, as does the explicitly parked stale-summary fallback.

## Reproduction and correction

Before correction, the new tests failed on the original expression:
32,000 digits took 9.192 seconds and 32,000 spaces took 9.324 seconds on
the VPS. Both inputs ended with `Read full topic` but lacked a count footer.
The same expression truncated authored `v12 posts - 3 participants Read full
topic` to `v`. Twelve existing lookalike cases passed in that reproduction.

The count must now begin at the start of text or after whitespace. The
search starts at the count itself, with no leading whitespace repetition;
only a recognized complete terminal footer triggers trimming of its preceding
separator. Long digit and whitespace runs are scanned without retrying their
tails. Legitimate singular/plural and footer-only inputs still strip; authored
tokens, unmatched strings, leading post whitespace and idempotency are preserved.

The new performance regression covers digit and whitespace runs with one
generous one-second ceiling per case. The authored-token case extends the
existing lookalike test; footer-only and mixed-whitespace assertions extend
the existing removal test. No new dependency or broad test scaffolding.

A lightweight post-fix laptop timing check measured 8,000 / 32,000 digits at
0.245 / 0.973 milliseconds and spaces at 0.185 / 0.735 milliseconds. These
are synthetic checks, not a claim about measured production request latency;
the before/after timings came from different hosts.

## Verification and boundaries

Isolated VPS worktree: `/tmp/hn-footer-regex-check-20261011`, based on
954bd2492d281679456913132730fa1f24d4563f. Both changed-file hashes match the
laptop. Codex Forum tests: 37 passed in 2.71 seconds. Repository Ruff/ty
and both touched-file format checks passed. Full backend gate: 1,304 passed /
1 skipped in 45.74 seconds, with the existing sklearn MLP convergence warning.
The run used four workers within a two-CPU / 3 GiB scope and single library
threads; VPS load after the run was 1.04 with 5.0 GiB available memory.

The initial archive-copy fixture was started before its transfer completed
and failed collection because `server.py` was absent; it was not a code
failure or counted as verification. The transfer was stopped and verified
exited. All successful gates used the fresh Git worktree described above.

No production checkout update, service restart, database/cache mutation,
provider request or additional reviewer. Runtime smoke remains deferred with
the separately authorized rollout. Original Muse WIP is preserved.

Source hashes:

```text
7b9de462b2a101acf4f312ffe3857d6ed79115f7f88333247fe6f3344c91931b  pipeline/codex_forum.py
c8edc4a5bc5480ecfd7fa16c59c0422827ee18c68c440577fc31aca4057dd1e1  tests/test_codex_forum.py
```
