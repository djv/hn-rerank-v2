# Data and privacy

This describes the implementation in this checkout, not a promised privacy policy.
It is a personal local-first application. An internet-facing instance needs an
operator who controls access, provider configuration, logs and backups.

## Stored data and purpose

- SQLite `users` stores a profile ID, bearer token and creation time. The
  `hn_token` cookie or `/u/<token>` import link grants access to that profile.
  Tokens are credentials, not anonymization. Do not share profile links publicly.
- `feedback` stores story, profile, action, seen state and last update time for
  personalization. Changing a vote overwrites its label/time; undo deletes the
  current row. There is no immutable vote-history ledger.
- `interaction_events` records impressions, article/comments opens and dwell,
  with session/event identifiers, client time, receive time, surface, filters,
  positions, badges and deck version. It supports measurement and trial attribution.
  Story IDs deliberately have no foreign key here; historical events can outlive
  a story. App-level validation skips unknown story IDs at insertion.
- `deck_scores`, `interleave_decks`, `rank_perf` and `llm_usage_daily` hold served
  rankings/probabilities, trial arm assignment, timing and provider usage records.
  `llm_usage_daily` is keyed by day/provider, not profile; it is a deployment-wide
  usage aggregate and cannot be erased by user ID.
- Stories contain URLs, titles, source labels, stored article/comment text and
  engagement metadata. Embeddings, side vectors and summary caches derive from them.
- Warm-start model fits are written under `warm_start_dir`, or
  `$XDG_CACHE_HOME/hn-rewrite/warm_start` (default `~/.cache/...`). They derive
  from vote data and need the same handling as other personal model artifacts.
- The TUI stores an origin-bound profile token in its platformdirs config
  directory (`hn-rerank/profile.json`), outside the package cache.

## Retention today

`Database.prune_stories(max_age_days=60)` uses `fetched_at`, exempts archive
sources `bq_seed`/`ch_seed`, and protects all stories referenced by feedback.
It is dormant: only tests call it. There is no automatic story expiry today;
refreshing a story also updates `fetched_at`. Enabling pruning needs a separate
decision about accumulated data, not a documentation or deployment cleanup.

Telemetry, feedback, deck records and model caches have no retention job yet.
Request size limits and the server's in-memory cache bounds do not expire rows.
The [backup script](../scripts/backup_hn_db.sh) defaults to 30 newest snapshots
(`HN_KEEP_N`), not a guaranteed 30 days. It uses SQLite `.backup`, uploads a
checksum, and can copy a populated `.env` as `dotenv`. Treat the backup remote
as secret-bearing storage. Timers, remote access and copies are operator-managed.

## Deletion and export limits

Undo (`Database.delete_feedback`, exposed by the feedback API) removes only
one current vote. It does not erase interactions, served decks, cached model
fits, old snapshots or another client's saved credentials. Clearing a browser
cookie creates a new profile and leaves the earlier profile's records intact.
There is no account erasure/export endpoint or ready-made full deletion tool.

An operator handling an erasure request must identify the profile, stop writes,
and plan scoped removal of its `feedback`, `interaction_events`, `deck_scores`,
`interleave_decks`, `rank_perf` and `users` rows. Deployment-wide provider usage
aggregates have no profile key and require a separate retention decision. Invalidate
per-profile warm-start fits and in-memory decks, and remove local cookies/profile
files. Review shared derived story/summary artifacts separately; other profiles
may still need them. Verify remaining references and the next regenerated model.
Snapshots, restored copies and external-provider records require separate handling;
a row deletion cannot recall them. No deletion is performed by these instructions.
For research export, prefer aggregate results over vote-linked text or raw profiles.

`remove_source_stories.py` protects voted stories; regression coverage lives in
[tests/test_remove_source_stories.py](../tests/test_remove_source_stories.py).
Do not use source cleanup as a substitute for profile erasure.

## External transfers and access limits

Ranking and embeddings are local. Ingestion contacts ClickHouse, HN item services,
RSS publishers, Reddit archives and article hosts. Summary requests send selected
article/comment text and prompts to the configured LLM provider. The UI can
display related upvote titles to anyone holding the profile token. Some archived
offline LLM studies also used vote-derived context; those workloads are not fully
local. Provider logs and retention are outside this repository's control.

The public checkout omits database files, populated environment files and client
cookies. Evaluation artifacts derived from the owner's votes (labels, story IDs, scores)
are not included.

Same-origin mutation checks and request throttles reduce some web abuse; profile
bearer tokens are not a multi-user account system. See [deployment](DEPLOYMENT.md).
