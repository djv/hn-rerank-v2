# Independent review — HN live-feed check + parked warm-start test draft

Scope: offline review of supplied material only. Nothing was run. Paths are relative to `files/`.
Status split: **292e3ef (docs) is committed work**; runtime is b1897f4. The **warm-test draft is
unvalidated, uncommitted** on `fix/warm-test-20261012` — nothing below treats it as verified.

## Findings

**1. Medium — "no synthetic event occurred" is asserted, not shown.**
`tmp/hn-live-review-20261012/DOCS.patch:25` (also `:22-23` counts).
The check issued 5 authenticated `GET /api/feed` calls as profile 151 (21 cards each). Evidence
(`LIVE-EVIDENCE.md:4`) itself says authenticated GETs "can exercise ordinary server caching/serving";
nothing shows whether serving the feed records `impression` rows or trial exposure. If it does,
the check wrote up to ~105 impressions into the user's interaction/trial data, and the
117-in-24h / 5,141 counts may include them (query time vs GET time is not recorded).
Fix/verify: grep the `/api/feed` handler for impression/exposure inserts; run
`SELECT count(*), min(ts), max(ts) FROM events WHERE user_id=151 AND ts BETWEEN '…16:31:50' AND '…16:33'`
(read-only). Reword to "no POSTs or reader interaction" until confirmed.

**2. Low-Med — STATUS overstates the draft: "removes the discrepancy".**
`DOCS.patch:67-68`. One seed (1155516390) shrank from 7.6e-3 to 1.43e-5 at tol 1e-6. That is
strong evidence for that seed only; no multi-seed sweep and no test run. FINDINGS (`:37-46`) is
correctly hedged; STATUS should say "reduces the gap ~530x on the recorded seed; untested".

**3. Low — process check and readiness are weak signals, wording slightly too strong.**
`DOCS.patch:23-25`, `:18-19`. FINDINGS drops the evidence caveat that pgrep does not rule out a
differently named process or another host (the `hn` reader likely runs on the user's machine, not
where pgrep ran). "Reading-signal delivery is therefore still unverified" is the right conclusion,
but not because of pgrep. `ranking-ready?min_version=0` is trivially satisfiable and its body was
not captured, so "readiness 200" says only that the endpoint answers.

**4. Info — timestamp/version.** `version=1791821397103` decodes (epoch ms) to 16:09:57.103 UTC,
3 s after ExecMainStart 16:09:54 — consistent with a build at startup; the served feed was ~22 min
old at check time. The 16:09:54 `Failed with result exit-code` (`LIVE-EVIDENCE.md:4`) shares the
start second; at 1-s resolution "precedes" (`DOCS.patch:28`) is not established, and why the old
unit exited non-zero is unexplained. Not a defect claim — record it as unexplained, and check
`journalctl -u <svc> --since 16:09:50 --until 16:10:05 -o short-precise` if it matters.

**What the live report does verify (API/DB data only):** all 5 supported windows HTTP 200,
ready=true, version==target_version, 12 Recommended / 9 Explore, slot-12 Discover card matches the
stored interleave row arm `far` for this version, Explore 3/3/3, runtime hash b1897f4, unit active,
NRestarts=0, no warnings after 16:10, zero non-impression events. **Not verified:** physical
browser/TUI rendering, ranking quality/efficacy, reading-signal delivery, side-effect-free GETs.
Docs correctly disclaim the first three. The 3d/all HTTP 400 was correctly not treated as an issue.

## Warm-start draft (`WARM-TEST.patch`, `code/hn-rerank-warm-test-20261012/tests/test_joined_classifier.py:325-366`)

**5. Medium — warm ≤ cold iterations is not a universal invariant (line 359).** L-BFGS from a
closer start usually, but not provably, needs fewer iterations; curvature-pair history, line search
and a newly-kept word column starting at 0 (`pipeline/joined_classifier.py:114-125`) can make warm
take more. With `@given` drawing 10 fresh seeds per run, a counterexample is a future flake.
Tighter tol makes the ratio more stable (20/12, 29/22 on the seed) but proves nothing.
Verify: offline sweep ≥1,000 seeds at tol 1e-6 recording `warm_iter - cold_iter` and max gap.
Fix if needed: assert on a mean/median across seeds, or allow small slack, or
`@settings(derandomize=True)` so CI is reproducible.

**6. Low-Med — test now guards math, not production behaviour.** `WARM-TEST.patch:35`.
Setting `estimator.tol = 1e-6` (fit reads it, so it takes effect) makes the assertion meaningful
(1.4e-5 vs atol 5e-3, ~350x margin) and still checks warm-start correctness: same optimum, fewer
iterations, coefficient remapping across vocab change. The new comment (`:45-46`) is accurate.
But it no longer covers production tol=1e-4, where warm vs cold differ by up to 7.6e-3 in held-out
probability. That is not a bug (both are within solver tolerance of the convex optimum) but it
contradicts the runtime docstrings' "optimum does not change"/"same ranking"
(`pipeline/joined_classifier.py:95-97`, `pipeline/linear_blend.py:124-127`): scores depend on fit
history (in-memory warm vs restored `warm_store` vs cold). Not masking a crash, but masking an
unmeasured ranking effect. Verify: at production tol, compare cold vs warm `P(up)-P(down)` rank
order over many seeds (Kendall tau / top-12 overlap); if non-trivial, consider a modestly tighter
production tol — a production change requiring separate user approval, not part of this draft.

**7. Info — `@example(seed=1155516390)` (`WARM-TEST.patch:18`)** is good: it runs in addition to
the 10 random draws and pins the reported regression. Atol 5e-3 retained as stated.

## Required before calling the draft done
1. `pytest tests/test_joined_classifier.py -k warm_start` on the draft branch, then the full suite + lint/type gates.
2. Seed sweep per #5; impression side-effect check per #1.
3. Amend STATUS wording per #2 in the next docs commit.
