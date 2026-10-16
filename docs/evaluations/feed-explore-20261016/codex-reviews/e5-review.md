Static review only; no files changed, script execution or database access.

1. **High — output can overwrite the live DB.** [exposure-gaps.py:315](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:315) opens arbitrary `OUT` with `"w"`, bypassing SQLite’s read-only protection. A mistaken path or symlink can truncate the DB. **Fix:** exclusively create a new JSON file with mode `0600`; reject existing targets and DB/sidecar paths.

2. **High — historical votes contaminate the universe.** [exposure-gaps.py:152](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:152) adds *all* voted stories, regardless of vote date; [line 228](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:228) also counts those old votes as found elsewhere. This changes clustering, pool shares and flags. **Fix:** add vote extras only when `updated_at >= SINCE`; fetch pre-SINCE reference ups separately.

3. **High — self-neighbour leakage.** [exposure-gaps.py:189](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:189) selects reference ups from the universe, and [line 200](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:200) includes their self-dot products, approximately 1. Cluster familiarity is inflated by its own voted members. Production explicitly excludes self in training neighbours ([ranking.py:1507](/tmp/hnx2-20261016/e5/pipeline/ranking.py:1507)). **Fix:** mask matching story IDs; return null where no other reference remains.

4. **High — candidate eligibility is too broad.** [exposure-gaps.py:139](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:139) admits every recent summarizable source, including disabled feeds and retired AINews rows. Production applies configured-source and legacy-row exclusions ([pipeline/__init__.py:354](/tmp/hnx2-20261016/e5/pipeline/__init__.py:354)). These cheap exclusions are missing beyond the documented cap/dedup approximation. **Fix:** replicate those filters and effective-story cleanup; then union required shown/voted extras.

5. **Medium — missing embeddings erase observed exposure/outcomes.** [exposure-gaps.py:174](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:174) drops shown/voted stories too; subsequent cluster **and source** totals cover only survivors ([line 206](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:206)). Coverage differences can bias ratios and rates. **Fix:** report missing coverage by source/outcome in JSON, retain unclusterable extras explicitly, and compute source statistics without requiring embeddings.

6. **Medium — reads are not one consistent snapshot.** Impressions, votes, stories and vectors are fetched separately ([exposure-gaps.py:105](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:105)); [database.py:273](/tmp/hnx2-20261016/e5/database.py:273) supplies five pooled connections. Live voting/enrichment can produce mismatched chronology or hash misses. **Fix:** perform extraction through one read-only connection and one explicit read transaction.

7. **Medium — undefined exposure becomes zero.** [exposure-gaps.py:230](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:230) reports zero shares/ratios when there are no impressions, including no Recommended impressions. **Fix:** use null for zero-denominator shares and ratios.

8. **Low — stdout exceeds the privacy contract.** [exposure-gaps.py:269](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:269) prints the profile ID. **Fix:** omit it; keep stdout to the requested aggregate table. No story-title/story-ID stdout path was found.

Wilson is correct ([line 77](/tmp/hnx2-20261016/e5/revote/exposure-gaps.py:77)); outcome timing matches [short-window-scores.py:60](/tmp/hnx2-20261016/e5/revote/short-window-scores.py:60). No day-forward folds, bootstrap, bins or AUC exist here. Absent outcome classes are handled; invalid/nonfinite vectors deliberately abort via [_validated_embeddings:139](/tmp/hnx2-20261016/e5/scripts/eval_ranker_variants.py:139).

No obvious quadratic-memory blowup: similarity uses 500-row chunks, about 18 MB per block at 4,500 references. Cap BLAS/OpenMP threads at two; runtime remains unmeasured.

**Verdict: RUN AFTER FIXES — findings 1–8.**

