# Calibrated Unsure: code review 2026-10-16

Target: the first draft of `pipeline/unsure_calibration.py` and its wiring
in `pipeline/ranking.py`, `pipeline/config.py` and `database.py`.
Reviewer: a panel of six read-only Muse Spark runs (pi,
`opencode-go/muse-spark-1.3-contributor`, xhigh, `--tools read,grep,find,ls`),
each given the code only, no data. Angles: statistics of the fit (r1, r2),
incremental deck reads and concurrency (r3, r4), robustness and validation
(r5), tests (r6). All six exited 0.

## Merged findings and dispositions

| Finding | Runs | Disposition |
|---|---|---|
| Unregularized fit (C=1e6): separable data gives huge or negative slopes | 5 | Fixed: C=1, at least 10 ups and 10 non-ups, convergence check, slope must be in (0, 5] |
| Incremental read keyed on created_at skipped late or replaced decks and re-parsed the last one | 4 | Fixed: re-read 300 s before the newest deck seen; absorbing keeps the earliest deck per story and is idempotent |
| Global lock held across DB reads | 4 | Fixed: reads run outside the lock |
| Unbounded per-user memory and cold-restart parse cost | 3 | Fixed: 7-day window (`unsure_calibration_days`); measured 4.3 ms per deck parse, about 3 s cold for the busiest profile |
| NaN probabilities, or no down/neutral mass, invented probability | 2 | Fixed: `calibrated` returns None and Unsure keeps raw entropy |
| `since = NaN` passed validation | 2 | Fixed: since and days must be finite |
| Changed or undone votes: feedback keeps only the final vote | 1 | Documented limitation; voted stories are rarely served again |
| Selection bias: Unsure changes which stories get votes, and the fit uses all views | 2 | Documented; matches the offline basis (E7, all voted first-shown stories). Monitor the slope |
| Recalibrate all three classes instead of up-vs-rest | 2 | Not adopted: temperature scaling did not improve ECE offline (E7 and the joined-model rerun) |
| `voted_at <= deck time` tie excludes same-second votes | 1 | Kept: conservative |
| Test gaps: end-to-end wiring, `get_feedback_times`, cache isolation, rejected fits, out-of-order decks, NaN fallback, fitted (not hand-made) calibrator | 1 | Added in `tests/test_unsure_calibration.py` |

Checks after the fixes: 1,443 passed / 18 skipped, ruff clean, ty at its
baseline of 34 diagnostics.
