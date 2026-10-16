# calibration.py — `uv run python revote/calibration.py USER SINCE_ISO OUT.json`
The DB constant matches the replay templates; every connection is read-only, with no fold DB.
Only the new OUT.json is written (exclusive creation, mode 0600); stdout contains aggregates.
Each UTC day with eligible votes trains strictly before midnight; the ranking clock is midnight.
Uses `_score_and_rank(training_feedback=...)` directly and eval's frozen embedder/validators.
Probability provenance: ranking.py:1698–1725 and 1843–1858; no reconstruction of margins.
Cold days without probabilities are counted and skipped; scoring errors/embedding gaps abort.
Stored 384-d normalized vectors are untouched; production handles metadata and side vectors.
Vote-day rows drive calibration once per story; REVOTE timestamps are excluded everywhere.
First-shown means earliest recorded impression in any view, including history before SINCE.
Prevotes (>1 s before that impression) are excluded; votes lacking impressions remain eligible.
First-shown rows may carry eventual votes, but these never train calibrators before vote day.
Reports 10 equal-count reliability bins, classwise/macro ECE, summed multiclass Brier and log loss.
Up/down metrics exclude neutrals and use P(up)/(P(up)+P(down)); binary Brier is squared error.
After 30 earlier labeled predictions, T minimizes log loss on a fixed [0.05,20] log-grid.
Platt fits logit(P(up)) vs rest (C=1e6, seed 0), preserving the down:neutral ratio for entropy.
Paired raw baselines/deltas use identical later-day cohorts; full reliability tables are in JSON.
Entropy is production Shannon bits; quintiles split ties deterministically; top-10 ties use ID.
Limitations: current mutable text/votes/config, no frozen DB snapshot, exposure selection bias,
small/tied bins, partial days, delayed-vote censoring; top-10 is a shown-pool proxy, not full Explore.
