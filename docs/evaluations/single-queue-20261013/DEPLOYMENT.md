# Single-queue rollout — 2026-10-13

User authorization: approved Muse review followed by merge/deploy/single
restart if no High issue. Separately approved sorting the two test import
blocks and continuing the same rollout. Muse verdict PASS, no High issue;
remaining findings and agent verification are in `MUSE-ASSESSMENT.md`.

## Revisions and verification

- Reviewed branch tip: `5b3b996`; import correction/review records `640c461`.
- Main merge and deployed runtime: `bede2ceda68d43559690a6daacdac015fcacaa61`.
- Previous production: `b1897f4c327c2de298ae443213b5344d1e77e2d4`.
- VPS backend: 1,445 passed / 1 skipped. Standalone TUI: 272 passed /
  1 skipped at four workers. Eight-worker quit-deadline failure under
  concurrent load is recorded in the assessment; no test weakened.
- Root/standalone Ruff and ty passed; 14 touched Python files format-clean.
- Standalone wheel built; installed wheel `--version` reported 0.1.0.
- GitHub main backend CI run `38467165477` and Terminal client run
  `38467165458` passed, including standalone build and wheel startup.

## Backup and preservation

Private online backup, before updating production:
`/home/dev/backups/hn-rewrite/single-queue-20261013T2057Z/hn_rewrite.db`.
Source DB size before backup: 1,223,266,304 bytes. Free filesystem space:
27 GiB. Backup quick_check: ok; SHA256:
`4fb37049f9ac19cf42020764d5f679d965fd7d112aa902ee87927928d5906669`.
The backup remains private, mode 0600; no retention deletion, upload or
credential copying. Validation is not a restore test.

Production fast-forward preserved SHA256s of `config.toml`, `uv.lock` and
unrelated `VPS-COLD-ARCHIVE-20261011.json`. The archive remains untracked.
No feedback writes, artificial telemetry or new profile was issued.
Startup applied the additive `surface` and `queue_position` columns;
read-only schema inspection confirmed both.

## Single restart and live checks

Restart: 2026-10-13 20:59:38 UTC (16:59:38 ET), MainPID 121960.
Active/running on follow-up, same PID/start timestamp. Systemd's exit-143
failure line belongs to the outgoing process during the requested stop;
there are no warning/error journal entries from the new process. Normal
tokenizer-only and Flask development-server notices remain unchanged.

Existing profile 151 verified through `/api/user` without exposing its
cookie. Dashboard, feed and readiness GETs returned 200. Startup warm
completed at 21:00:17 UTC, rank time 30.049 s. All five windows later
returned `ready=true`, deck version `1791925180942`, 12 Recommended cards.

Summary smoke: story 50073879 first had cache GET 200 but discussion
hydration invalidated that key on POST; POST generated and returned 200.
Story 50073633 had a stale-cache GET 204 and POST generated/returned 200.
Repeating POST for 50073879 returned 200 with
`result=post_enrich_cache_hit` in the journal. No forced refresh flag.

Import AI integration gate on the live DB from the deployed checkout:
exit 3, NOT_APPLICABLE because issue 476 is already voted. This is not PASS.
The bounded final warning journal scan was empty.

The laptop's `hn` alias uses the TUI venv whose editable path points at
this main checkout's `clients/tui/src`; a fresh reader launch loads Queue.
No personal reader was restarted. Real feel/rendering of rounds and live
delivery of new Queue events remain unverified until normal user reading.
Startup alternates Queue/Recommended by UTC ordinal day.

Code rollback would retain the additive schema and accumulated feedback;
do not restore this snapshot over subsequent user data. No rollback or
second restart was performed. Documentation-only follow-up commits do not
change the deployed runtime.
