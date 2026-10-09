# Authorized section/r/stats rollout — 2026-10-09

The user chose Save fixes, then explicitly authorized deployment. Saved/pushed
commit ad943867d1846710742691bfa06a9325c9ee927f contains the seven fixes,
regressions and handoff. At 22:42:32 UTC the VPS live checkout
/home/dev/hn-rewrite/main fast-forwarded from
7706816c51453aea966c0344374bbc5837b19a90 to ad94386 and
hn_rewrite.service restarted. No local modifications existed in the live
tracked tree/index; existing files and research work were preserved.

- Rollback reference: deploy-pre-section-refresh-20261009 at 7706816.
  Roll back code by switching to that tag and restarting after checking WIP;
  do not restore the DB snapshot over later user feedback for a code rollback.
  The nullable additive column is compatible with the old code.
- Online SQLite backup before restart/migration:
  hn_rewrite.db.pre_section_refresh_20261009, 1,185,443,840 bytes, in live tree;
  PRAGMA quick_check returned ok. This is backup validation, not a restore test.
- New service MainPID 4170967, active; start timestamp 22:42:32 UTC read back.
- Config SHA256 unchanged before/after:
  99a2ad40f7f9973fd8e25946dd5d95e2eeac887e3ec3decc19323b091c67f57e.
  No model, interleaving-arm, profile, feedback or research configuration edit.
- All entries in SEVEN-FIXES-SOURCE-SHA256.txt passed against deployed files.
  Same executable/test/config/template source as the full backend 1264 and TUI
  202 passing isolated gates; no runtime source change during deployment.
- Additive tldr_cache.source_comments column present after startup.
- DEPLOY-SMOKE.json: existing profile151 reused internally; dashboard/feed/
  readiness/cache 200; HN stats 200 with both fields live (Firebase); unsupported
  source 200 with both fields stored and explicit unsupported_source reason;
  malformed ID400/missing story404; unauthenticated stats401.
  Legacy cache response had no snapshot, correctly unknown.
- Bounded journal from restart through ~22:43:47 UTC: no matches for
  ERROR|Traceback|quota_denied|empty HTML tree. Ordinary RSS httpx403→urllib
  fallback messages appeared; not a claim all upstream sources are healthy.
- Local TUI PID1941821 started 18:41:40 EDT (22:41:40 UTC), after final app.py
  write 18:34:49 EDT. Its venv resolves hn_rerank.app to the updated local
  clients/tui/src checkout. Existing user process was not stopped or manipulated.

Live smoke uses only GETs and never generates a summary, votes, creates a
profile or changes trial settings. The service's normal startup warming/feed
refresh can still run under existing configuration. Uncached/forced live summary
generation and physical s/h/l/r/modal controls remain unverified; production
Pilot regressions cover these behaviors with mocks. No inference quota spent
by the smoke script. Documentation-only follow-up may advance checkout HEAD
without changing the ad94386 runtime loaded at restart.
