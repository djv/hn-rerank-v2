# Discover: Explore that finds new topics the user likes — plan (2026-10-12)

Status: approved to start by the user 2026-10-12 ("go ahead with the
explore plan"). Phase 0 done: R1 (best production score among far
stories, tau 0.60) passes the gate (FINDINGS.md "Discover Phase 0").
Phase 1 built with R1 at 0.60 (a module constant, `DISCOVER_MAX_SIM_UP`)
and deployed 2026-10-12 12:31 UTC (7c362d1). Phase 2 measurement runs to
about 2026-11-02.

## Why

- The user's worst failure is a same-y feed (STATUS.md "Goals"). Only 6% of
  up-votes since 09-28 were novel (max cosine similarity below 0.60 to every
  earlier up-vote), and 1 of 204 opened a topic with no up-vote in 90 days
  (FINDINGS.md "Sameness and novel up-votes").
- Explore is where the novel up-votes come from (5 of its 20), but its picks
  rarely land. Since badges were logged (09-30): Interest 7/76 up (9%),
  Unsure about 5%, Novel 1/58 (2%, half of them down-voted).
- Novel picks the stories *farthest from every vote*. That finds the
  unrelated, not the good-but-new. Far cards can work: Recommended's far
  cards got 11% up, against 22% for near ones.
- Explore sits outside the interleaving trial, so it can change now.

## Design

1. **Discover replaces Novel** (✨ keeps its slot count, EXPLORE_PER_BADGE).
   Candidates: the window's stories not already picked, not a duplicate of
   a voted story, whose max similarity to the user's up-votes is below a
   threshold τ (start 0.60, configurable). Among them, pick by the rule that
   wins Phase 0. Candidates:
   - R1 best-of-far: highest production score.
   - R2 quality-of-far: a user-independent quality prior (HN points
     velocity and comments; for other sources, the source's past up-rate).
   - R3 blend: rank average of R1 and R2.
   - R4 topic bandit: Thompson sampling over topic clusters with few
     up-votes (Beta posterior from the user's votes per topic), then the
     best-scored far story of the sampled topic.
2. **Interest stays** (best Explore badge). **Unsure stays** for now; its
   future is the parked Teach question.
3. **Attribution.** New badge `discover` in impressions, so its up-rate and
   novel up-votes are measured per card. Clients show "✨ Discover — new to
   you, likely good". The API view key stays `explore`.

## Phases

- **Phase 0 — offline (read-only, VPS, no paid compute).** On profile 151,
  for each day since 2026-09-24, take the stories shown that day in any
  view, keep the far ones (max similarity < τ to up-votes cast before the
  day), and score them with information from before the day (day-old
  production models, as in the short-window check). Report per rule:
  - up-rate of its top 3 far stories per window and day;
  - novel up-votes caught;
  - paired bootstrap over days against Novel (R0) and a random far pick.
  Gate: a rule beats random-far with a CI clear of zero and reaches about
  8% up (Interest's level). Otherwise report and stop.
  Limits:
  - The pool holds only stories already shown.
  - HN points and comments are read at snapshot time (leak).
  - The small n allows a directional read only.
- **Phase 1 — build (after the user approves Phase 0).**
  - Server: the Discover pass in `assemble_window_deck`, with τ and the
    rule in config.
  - Clients: TUI and web label and help text.
  - Tests: candidates respect τ and exclusions; badge and attribution
    logged; Recommended and Popular unchanged.
  - The project verification protocol applies; independent review before
    deploy.
- **Phase 2 — deploy and measure.** Restart only with the user's OK. Weekly,
  with `~/revote/sameness-metrics.py` plus a Discover split:
  - Discover up-rate;
  - novel up-votes per week, now about 6, aiming for +3;
  - Explore down-rate and total up-votes as guardrails.
  Read the result after 3 weeks, about 2026-11-02, next to the trial
  report.

## Out of scope

Recommended (diversity re-ranking such as MMR waits for the trial to end on
2026-11-05), the ranker, Teach, per-story labels in this repo.
