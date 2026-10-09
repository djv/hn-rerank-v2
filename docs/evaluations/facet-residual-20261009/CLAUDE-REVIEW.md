**Verdict:** The implementation is valid enough to trust the negative result, and stopping this bounded facet design is justified. The result means no detectable facet gain from 140–210 residual rows. It does not show that facets, or LLM features in general, are useless.

**What the result shows**
- The real facets did no better than shuffled ones. Observed Δ arm3−arm1 is −.051, against a null mean of −.047.
- The deterministic proxy also lost .039. So the drop is the cost of adding about 30 dummy columns on 140 rows, not harm from the facets.
- Power was always weak. Noise alone cost about −.047, so passing +.02 needed about +.07 of real signal. The bootstrap CIs include 0.

**Correctness checks that passed**
- One-hot vocabularies are built from training rows only, with zeros for unseen test values (`evaluate.py:340-352`).
- Pooling is by pair count within each block.
- Permutations keep train and test separate, with multiset assertions.
- Model-cache leakage is ruled out: the cache key includes the training IDs and labels (`pipeline/ranking.py:1324-1348`), so no fit was reused across blocks.

**Concrete issues** (none of them reverse the verdict)
1. **Null strata** use the raw source (`evaluate.py:460`). Many `rss_reddit_<sub>` groups have one story, so those rows never get shuffled. The null partly contains the real facets and sits too close to the observed value.
2. **The residual design under-weights the ML score.** C=1 L2 shrinks the percentile coefficient as hard as the dummies (`evaluate.py:363`). The score is also reduced to one within-block percentile (no neutral probability, no confidence), and facet weights are learned from 140 rows while production learns from 3848. This setup could never beat the native score unless the facets were very strong. A version with the percentile as an unpenalized offset is a retune. It would need a new, independently frozen test, not this data again.
3. **Snapshot mismatch:** the annotator read `/tmp/opencode/cc-replay/snapshot.db` (`label_facets.py:41`). The proxy read the working copy (`evaluate.py:61`), which `oof_baseline.py:91` opened writable. Check that title, url, source, self_text and article_body are identical.
4. **Stale rubric:** `rubric.json:6` still lists `text_content` in the body order. The frozen rubric therefore misdescribes the repaired input; note it in the amendment.
5. **Secondary metric weights:** up-vs-rest pooling weights by three-class pair counts (`evaluate.py:419`), which is wrong. The dev99 up-vs-rest gain (+.04) is a lead to check, not evidence.
6. **Deviations:** the retry and the batch-size change were benign. Inputs and prompt stayed fixed.

**Next approach: test the embedding input before any LLM rewriting.** The facets exposed that `text_content` contains comments, and production embeds up to 6000 comment characters, mean-pooled. A topic signal dominated by comments may be what blurs format and stance. This is the precondition for LLM canonicalization (option B), and it can be checked with no LLM calls:
- Re-embed the 3848 voted stories with the production encoder, using title + self_text/article_body only (no comments).
- Rerun the existing `oof_baseline.py` four-block harness, with both arms under the same config.
- Freeze the gate before running: ordinal AUC gain pooled over blocks 2–5 of at least +.01, at least 3 of 4 blocks positive, and a paired bootstrap.
- If it fails, stop spending Muse quota on representation rewrites.
- If it passes, a bounded LLM-canonicalization trial becomes justified, but it must be validated on votes after 2026-10-09. The 99 dev stories are spent as development data.

Local inference is heavy work, so it should run on the VPS via `batch` with threads capped.
