You are an independent, read-only reviewer of the ML ranking model in this repository (the current directory): a personal Hacker News/RSS/Reddit reranker ("hn-rerank"). Topic: **how to improve the ML model's ranking quality**. You can only read files; do not attempt to run code. There is no database in this copy.

Context to read first (skim, then go deep where your angle needs):
- STATUS.md "Goals" (the four aims) and "Next step"; AGENTS.md "Working rules".
- ARCHITECTURE.md sections 3.2-3.5 and 3.13 (features, RBF SVM, linear blend, dual gate, joined-logistic challenger, live interleaving, selection passes).
- Code: pipeline/ranking.py, pipeline/linear_blend.py, pipeline/joined_classifier.py, pipeline/side_embeddings.py, pipeline/interleave.py, pipeline/config.py, pipeline/__init__.py, config.toml [hn_rewrite.model], scripts/ (evaluation scripts), tests/.
- Prior research, so you do not re-propose what was already tried and rejected: FINDINGS.md (ranker sections, e.g. "Stacked ranker", "Incremental ranker hill-climb", "Untuned embedding probe", "Ranking-quality study", "TF-IDF tuning sweep", "EmbeddingGemma 2"), docs/evaluations/RANKING-REASSESSMENT-20261009.md, docs/evaluations/model-ablation-20261007/ (README, FEATURE-REMOVALS, SINGLE-MODEL, INTERLEAVING, BROADER), docs/evaluations/diversity-20261012, discovery-20261012, missed-upvotes-20261009.

Goals and constraints (from the owner):
- Judge every proposal against all four aims: save time (most cards worth reading), discovery (things the front pages would not surface), stay current, calmer habit. AUC is near its ceiling; do not optimize AUC alone. The worst failure is a same-y feed (filter bubble): any proposal that cuts duds must not narrow topics or reduce novel up-votes. Say how each proposal affects topic spread.
- Single main user (profile 151, a few thousand votes: up/down/neutral) plus other small profiles; a new profile's first feed also matters. Small data: be skeptical of high-capacity models and of unusually high metrics (leakage).
- A 28-day live team-draft interleaving trial (production vs joined-logistic #2/#4) runs until 2026-11-05; no server restart or production ranking change before then. Ranking research is paused by the owner, so proposals are for after the trial: rank them by expected value per effort and say how to validate each offline first (which existing script/eval, which metric, which split).
- Raw L2-normalized embeddings must never be standard-scaled. Never propose deleting or rewriting the vote database.

Method: verify each claim against the actual code (cite file:line). Distinguish (a) concrete defects that hurt ranking quality today, (b) improvements not yet tried, (c) things already tried (cite the FINDINGS/docs section and say why your version differs or drop it). Mark uncertain items as such.

Output (max ~900 words), plain markdown:
1. Top findings/proposals, ranked, each with: title; category (defect / untried improvement / evaluation gap); evidence (file:line, doc section); expected effect on the four aims incl. sameness; effort (S/M/L); offline validation plan; confidence (high/medium/low).
2. Things you checked that look fine or were already tried (one line each).
3. Open questions.

Your angle for this review (focus here, but report anything important you find):
