# revote/label-noise.py — vote label noise vs a kNN noise ceiling

Read-only probe asking "how noisy are the user's labels, and how close is
the model to a plausible ceiling?" CLI: `uv run python revote/label-noise.py
USER OUT.json`. Stdout gets aggregate tables only; OUT.json adds the
near-duplicate conflicting pairs' story IDs (private, never titles).

Computes: (1) near-duplicate pair conflicts — all voted-story pairs with
cosine >= 0.90 / >= 0.95, conflict share overall, split up-down / up-neutral
/ down-neutral, plus same-normalized-URL pairs and conflict share by vote
time gap (<1d, 1-7d, 7-30d, >30d); (2) leave-one-out kNN agreement — up-share
of the k=1,5,15 nearest other up/down votes per vote, with AUCs, vs the
production ranker's day-forward up-vs-down AUC on the last 30 days of votes
(same FoldData/`_production_scores` pattern as short-window-scores.py);
(3) temporal drift — consecutive 30-day periods, k=5 AUC with neighbours
from inside vs outside the period.

Assumptions: labels down=0/neutral=1/up=2; REVOTE window excluded; stored
L2 embeddings only (missing vectors counted as skips); "conflict" is any
label mismatch; day-forward trains on all earlier votes including neutrals.

Limitations: kNN ceiling is crude (pure embedding similarity, no text/source
priors); production AUC pools day scores across days; tiny periods report
n/a AUCs; NaN appears in JSON where a bucket is empty; O(n^2) pair scan in
500-row chunks suits thousands of votes, not millions.
