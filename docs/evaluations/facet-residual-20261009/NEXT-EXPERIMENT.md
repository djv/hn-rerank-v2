# Next approach: representation audit before LLM canonicalization

Status: designed with Claude; not executed. No production changes authorized by
this document. All existing labeled datasets are development data.

2026-10-09 source audit: see REPRESENTATION-AUDIT.md, especially its primary
review. Before implementing the harness, inspect existing section artifacts and
check whether stored text_content equals the full composer on the frozen rows.
The two-arm question changes primary embedding inputs only: retain identical
story rows, lexical features, metadata and side vectors between arms. Removing
comments from all ranking channels is a separate experiment. Record source and
transformed input hashes independently; existing replay hashes represent source
input identity. No new annotation, encoding or fit has run in this audit.

Hypothesis: mean-pooled comment text can obscure the item's subject, artifact,
and firsthand content. Test the encoder input before paying an LLM to rewrite it.

## User-selected endpoint and mismatch policy (2026-10-09, before new fits)

The user selected **find more upvotes** in the native questionnaire. For this
future test, primary = within-block UP-versus-rest AUC, pooled by UP/non-UP
pair counts, using each arm's production ranking score. Secondary = known
upvote fraction at top 12; guardrail = known downvote fraction at top 12 must
not increase and known upvote fraction at top 12 must not decrease in the pooled
descriptive result (frozen before any challenger result). These are conditional on the
historically voted cohort, not unbiased live-feed precision estimates.
Ordinal AUC remains descriptive and cannot override the chosen primary.
Exploratory go gates: primary delta >= +.01; positive in >=3/4 test blocks;
paired story-bootstrap 95% lower bound >0; guardrail passes. Bootstrap remains
conditional on fitted models, and repeated development use is disclosed.
This replaces the ordinal endpoint in steps 3–4 below for the unrun test only;
the completed facet trial retains all original endpoints and verdicts.

Preserve all frozen old-cohort rows and blocks. On rows where stored text equals
the full composer, challenger input is the same composer without comments.
Where they differ, retain the original primary vector unchanged and report
those rows as unmodified; no implicit reconstruction or comment fallback.
Empty clean body intentionally gives title-only input. The read-only audit
found 51 mismatches and 516 empty clean bodies across old+dev combined; derive
old-cohort counts before encoding. Keep all original story fields, lexical
features and side vectors fixed. Validate baseline reproduction and disable
cross-arm warm-start/cache reuse before evaluating the challenger.

1. On a task-owned VPS snapshot, inspect the production embedding assembly and
   freeze the exact baseline and comment-free input definitions. Preserve title,
   self_text/article_body and encoder/truncation settings; remove only comments.
   Report how many items have no article/selftext. Do not add a hidden fallback.
2. Muse implements the two-arm harness and checks. Re-encode the same 3,848 voted
   stories with the production encoder; isolated embedding caches, one CPU,
   capped threads, resource checks. No LLM annotation calls and no live writes.
3. Reuse the chronological blocks and production ranking configuration. Fit each
   arm on earlier blocks only, evaluate blocks 2–5; no hyperparameter search.
   Primary: within-block ordinal AUC pooled by ordinal-pair count. Secondary:
   UP-vs-rest AUC pooled by UP/non-UP pairs, reported without overriding primary.
4. Freeze before results: pooled ordinal delta >= +.01, positive in >=3/4 blocks,
   and paired story-bootstrap lower bound >0 (conditional on fits; report limits).
   Failing any gate stops representation rewriting on this evidence.
5. Passing is an exploratory lead. Only then design a capped Muse canonicalizer
   that summarizes item facts without the user's votes/profile, compare against
   the clean-input arm, and freeze a prospective test on votes collected after
   the design freeze. Existing 99-story outcomes cannot validate that trial.

Alternative later: protect the native ML score as an offset and learn sparse LLM
residuals. This requires a new frozen protocol and untouched outcomes; the failed
facet trial cannot be retuned and presented as confirmation.

Endpoint alignment should be decided before the next LLM trial: the original
benchmark emphasized UP-vs-rest, while this facet trial prioritized the full
Down<Neutral<Up ordering. The secondary dev99 gain does not justify switching
this completed experiment's primary metric.
