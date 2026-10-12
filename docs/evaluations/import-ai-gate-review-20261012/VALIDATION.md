# Gate review verification and proposed correction

Independent Claude cloud review of fc36912:
https://claude.ai/code/session_01LJvudpDzjxGT4XNPr6ZmCJ
One reviewer, source only; no DB/private context uploaded. REVIEW.md is
unaltered. Reviewed code was held unchanged until the user separately approved the
correction plan; the authorized corrections are now applied.

## Verified findings

- Medium M1 confirmed in a disposable DB with the real Flask feed API.
  Publisher date14h old, matched HN row3h old, issue deliberately absent
  from12h Recommended but present in1d/1w/1m: current gate returns PASS,
  checking only1d/1w/1m. Script lines127-129 choose windows by publisher
  timestamp; server eligibility uses each stored row's time. This misses
  an actually eligible12h window.
- Low L1 confirmed with a public-shaped RSS fixture: issue501 dated
  NOW+30s and issue500 datedNOW-60s. latest_issue selects500. If500 is
  already seen/voted, this can hide the newest issue behind NOT_APPLICABLE.
- Security checks were clean: loopback origin, proxies/redirects disabled
  for cookie-bearing calls, read-only DB, no credential output. These are
  static/integration findings; the observed live run was NOT_APPLICABLE.
- Other low items: stale feeds fail safely but immediately; caller must
  use the live DB, /api/user identity can cheaply verify the expected profile;
  exact-string URL identity is intentional; feed GETs may queue ordinary
  server warms, so read-only means no feedback/interaction writes, not no
  server work; argparse can exit2; no checks must never pass.
- CLI exit0/1/3 and duplicate-ID/mixed-age coverage need strengthening.
  No new production test, profile, vote, model fit or browser operation.

## Concrete proposed patch (requires separate fix authorization)

1. Keep seven-day freshness by publisher date, but derive checked windows
   from matched DB row times, exactly as the server does. Any matched row
   eligible in a window makes that window required. Fail if no recent
   window is eligible. Add HN-first mixed-time regressions in both
   directions; union with publisher windows would still false-fail when
   stored timestamps are older, so it is not sufficient.
2. Select the newest numbered issue without discarding future entries;
   fail explicitly on a future publication timestamp rather than falling
   back to an older seen issue. No speculative timestamp clamping.
3. Verify loopback /api/user returns the requested profile before feeds;
   document that --db must be the actual service DB. No live history reset.
4. Add real-API/disposable-DB CLI exit and duplicate-ID/history tests,
   age-window coverage and publisher off-host/future-date cases. Document
   argparse exit2 and safe stale-feed failure/ordinary server warming.
5. Repeat focused/full backend, Ruff/format/ty and real read-only
   applicability check; commit verified fixes. No source boost, deployment,
   service restart, timer or additional independent review.

## Authorized correction validation

The approved patch now uses matched stored row timestamps for every required
window, including mixed-age duplicates. Publisher timestamps determine freshness;
a future newest entry fails explicitly. Applicable checks verify /api/user identity.
Five regressions failed before the fix. Focused gate+CLI: 78 passed. Final isolated
VPS full suite: 1,403 passed, 1 skipped (35.38s); existing MLP ConvergenceWarning.
Repository Ruff and ty passed; three touched-source format checks passed. Script,
gate tests and CLI-registration hashes match the checked VPS source.
Live service remains b1897f4; no restart, timer, history reset or additional review.
Live corrected-source check with /home/dev/hn-rewrite/main/hn_rewrite.db:
exit3 NOT_APPLICABLE, issue476 already voted by profile151. This validates
publisher transport and live applicability, not an unseen ranking PASS. An initial
attempt named a nonexistent shared/hn_rewrite.db and failed read-only with exit1;
no DB was created. Service working directory and actual DB were then verified.
