# Live-feed / parked warm-test review validation

2026-10-12. User requested `rv`; one independent Claude cloud review:
https://claude.ai/code/session_01S1oPNbGfpFVnajyZoHbBtj
Mailbox branch r/20261012-163339-hn-live-warm. See REVIEW.md for the
unaltered report. No reviewed source, STATUS.md or FINDINGS.md was edited.
No additional reviewer, deployment, restart, live vote or generation.

## Assessment

1. Impression mutation concern (review medium): cleared for this check.
   server.py:3923 feed GET calls _feed_for_user (1994), which builds the
   feed through pipeline/render.py:271. No impression insert there.
   The only server insert_interaction_events call is server.py:2916,
   in the interaction-event POST handler. Read-only VPS SQL found zero
   profile-151 events with received_at between 16:31:50 and 16:33:00 UTC,
   covering both check batches. HTTP GET of HTML does not execute its JS.
   No synthetic interaction event was recorded. This does not assert all
   GETs are free of cache/ranking effects.
2. STATUS.md:344 wording (review low-medium): accepted as a low precision
   issue. "Removes the discrepancy" describes the recorded seed, not a
   guarantee across every seed; the adjacent text calls the draft unvalidated.
   Prefer "reduces the recorded seed's probability gap below the assertion
   bound" if a later authorized docs edit revises it. Left unchanged for rv.
3. Process/readiness caveats (review low): no material completion error.
   pgrep was on the user's laptop, not the VPS, and only claims no matching
   process; it cannot exclude another name or host. Readiness HTTP 200 alone
   is insufficient, but all five feed bodies independently have ready=true
   and version=target_version. Physical rendering/delivery remain unverified.
4. Version/startup warning (review informational): resolved. Version decodes
   to 2026-10-12 16:09:57.103 UTC, not a future date. Precise journal:
   16:09:54.066758 stopping; .181247 old process exited status143;
   .181352 old unit failed; .188157 new unit started. Thus the warning
   precedes the new start. No claim that the shutdown status itself was fixed.
5. Warm<=cold iterations (review medium): theoretical flake risk, not
   reproduced. Independent synthetic sweep on the VPS of seeds 0..999
   at test-only tol=1e-6, exactly matching the test's generated data and
   model setup, found no warm>cold iteration count. Maximum held-out
   probability difference 0.0000899132358499477, below atol .005. Elapsed
   59.43 s, one BLAS thread, niced and timeout-bounded; no DB used.
   Finite evidence does not prove the iteration inequality for all seeds.
   Performance is better evaluated over a fixed sample, separately from
   mathematical equivalence, if this flake recurs. No counterexample or
   confirmed new test defect is claimed.
6. Production tolerance coverage (review low-medium): valid limitation.
   Draft tests tighter convergence; it does not guarantee production
   tol=1e-4 held-out score/ranking equivalence. The recorded 0.0076234
   default-tolerance probability gap remains observed; ranking-quality
   effect is unmeasured. No runtime bug or production change justified.
7. Explicit example: retained as a useful regression seed in the isolated
   draft. Full pytest/lint/type gates on that draft remain unrun; the sweep
   is not a substitute for the project's completion gate.

## State

Production unchanged. Main reviewed documents unchanged. Warm draft stays
uncommitted/unmerged in /home/d/code/hn-rerank-warm-test-20261012,
branch fix/warm-test-20261012. Both owned review/wait processes and the
bounded seed sweep exited. Post-sweep VPS load 0.66/0.37/0.62, available
RAM 4.8 GiB. Next authorized work could validate and finish the test draft;
this review does not authorize fixes or another review.
