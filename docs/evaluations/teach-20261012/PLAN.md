# Teach: active learning in place of Explore — plan (2026-10-12)

Status: proposed, not approved. Origin: the user's 2026-10-11 idea "redo
Explore to be just that [stories whose vote teaches the model] and call it
Teach"; the decision waited for the offline experiments, now done (recency
and neutral-as-down null, FINDINGS.md 2026-10-11).

## Why

- Explore barely gets up-votes: 5.6% of shown cards since 09-26 vs 26.8%
  for Recommended; its far (Novel-like) third got 1/78 up, 92% of its votes
  down. Down votes on stories the model already ranks low teach it little.
- Labels are the limit: blind re-votes put the user's own ceiling at about
  production's AUC (0.785 vs 0.778), and no ranker tweak tested on
  10-11 helped. More informative votes are the remaining lever.
- Unsure today is the entropy of the three-class probabilities, so it
  mostly picks neutral/down confusion, not the up-vs-rest boundary that
  decides what Recommended shows.

## Design

1. **Teach replaces Explore** in both clients (section label, help text;
   the API view key stays `explore` so old clients keep working). Interest
   picks go; Novel shrinks to a minority share.
2. **Random draws with a known probability per draw.** Each warm draws
   Teach's cards from a mixed proposal q over the window's unvoted, non-duplicate
   candidates: about 75% on the selection rule Phase 0 picks, 25% uniform.
   This is the active-testing setup already proven in the calibration tool
   (Kossen et al., ICML 2021). Each draw's q is stored, so Teach votes are both
   training labels and an importance-weighted (LURE) unbiased sample of the
   pool, which gives a live estimate of production's AUC with no extra votes.
3. **Storage, additive only.** New table `teach_draws(user_id, story_id,
   dashboard_version, window, q, rule, drawn_at)`; impressions already log
   `sort_mode` and `badges`, so a Teach vote is identifiable. Votes stay
   ordinary `feedback` rows (same training path, same undo).
4. **Near-duplicates of voted stories are skipped** (existing
   `is_feedback_match`), matching the user's habit of voting neutral on
   repeats.
5. **Footer hint** with the stricter voting rule: 1 glad I read it,
   2 read it / fine, 3 wouldn't click.

## Phases

- **Phase 0 — offline simulation (gate).** Codex writes
  `scripts/simulate_teach.py` and runs it on the VPS: on profile 151's
  temporal folds, each rule picks a budget of 50/100/200 held-out stories,
  their real votes are added to training, and the model is scored on the
  rest of the fold (up-vs-rest AUC, P@8). Selection rules:
  - random
  - entropy (today's Unsure)
  - near the window's Recommended cutoff (|score pct − cutoff| smallest)
  - P(up) margin (closest to the up/not-up boundary)
  - novel
  - 75/25 mix of the best rule with uniform

  Paired bootstrap against random. Go only if a rule beats random with a
  CI clear of zero; otherwise stop and keep Explore (or just drop its
  Novel picks). Limit: the pool holds only stories the user was shown and
  voted on, so the simulation favours rules that pick what the old views
  picked. About an hour of VPS CPU, no paid compute.
- **Phase 1 — build (after the user approves the Phase 0 result).**
  - Server: replace the three Explore passes in `assemble_window_deck`
    with the Teach draw; add the table and LURE weights (ported from the
    calibration tool's `proposal`/`lure_weights`).
  - Clients: TUI and web label and help text.
  - Tests: draw probabilities sum to 1, every candidate has q > 0,
    duplicates and voted stories are excluded, weights are unbiased on a
    synthetic pool (Hypothesis), and the web/TUI section label renders.
  - Codex implements (quota rule until 10-15); the agent reviews; the
    project's verification protocol applies.
- **Phase 2 — deploy and measure.** Deploying needs the user's OK to restart
  the service. Teach sits outside the interleaved Recommended views, and its
  votes train both arms, so the trial stays fair. Measures:
  - weekly LURE-weighted production AUC from Teach votes;
  - after 4 weeks, an offline ablation on later feed votes: the model with
    vs without the Teach-vote rows;
  - Teach and Recommended up rates, descriptive only.

## Out of scope

Changes to Recommended, the trial, or the ranker; per-story labels in this
repo; any change to existing votes.
