# revote/exposure-gaps.py — feed exposure gaps

No model training. Asks "which topics does the feed under-show relative
to how often the user upvotes them?"

Universe: summarizable (`is_summarizable`) stories with stored embeddings
whose story time >= SINCE, plus every voted/shown story (any age), kept
only where a stored embedding exists (`_embedding_text_hashes` /
`_validated_embeddings`, L2-normalized, no scaling). MiniBatchKMeans(k,
seed 0, n_init=3) on those vectors.

Per cluster: pool share, any-view and Recommended-only shown shares and
exposure ratios (shown/pool), up/down/neutral among shown-with-outcome
(outcome rules as short-window-scores.py; prevotes dropped), Wilson 95%
interval, and mean max cosine to pre-SINCE ups. Never-shown votes count
separately as found-elsewhere. Flags: UNDER-SHOWN HIT (rate lo > overall,
exp < 0.7), OVER-SHOWN MISS (rate hi < overall, exp > 1.3). Same rates
per raw `stories.source`.

Assumptions: first any-view showing defines the outcome cutoff; flags use
the any-view ratio; "never shown" means no impression since SINCE.

Limitations: universe skips production's gravity/RSS caps, archive legs,
and dedup (a stated superset approximation); outcomes exist only for shown
stories, so flagged "hits" still need a causal read; clusters are
descriptive. DB read-only; titles only in OUT.json (8 nearest-centroid).
