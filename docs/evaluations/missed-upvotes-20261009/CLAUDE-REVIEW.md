**Main conclusion:** none of these reports shows a Muse advantage. The full score-component audit is mostly a detour. A cheaper check comes first and decides whether the audit is needed: does the native ranker tell HN UPs from HN DOWNs as well as it does for other sources? If it does, the HN tail is a reasonable source prior, not a blind spot, and the HN line should stop.

**Findings, in priority order**

1. **The top-12 replay is synthetic.** missed-upvotes §Definitions ranks only *voted* rows (about 770 per block), not the live pool of eligible stories. Every voted story had already been shown by a ranker (cc-replay §1 exposure note), so the 48-slot ceiling, the "19 replaceable picks" and the miss rates describe this artifact, not dashboard behaviour. The arithmetic checks out; what it means for the live dashboard is unknown.
2. **Rank-based quintiles mix up base rate and error.** The pooled worst quintile uses absolute within-block rank. Blocks with more UPs (b3 has 236, b5 has 100) push more UPs into deep ranks mechanically (§1). §2 compares HN to the block rate but never to *HN DOWNs*. If HN carries many DOWNs, ranking HN low is correct, and HN UPs sink with them. The counts are verified; "blind spot" is a causal hypothesis that hasn't been tested.
3. **No Muse superiority claim survives.** CC replay: 2 fixes against 4 damages, and the damage came at high confidence. The targeted probe's 11/12 measures agreement with Opus, not accuracy against the user's votes. Facet arm3−arm1 is −.051 and all five gates fail. The CC damage pattern (Muse picks the story that matches the profile topic, and the user downvoted it) argues directly against building a semantic profile from past votes.
4. **Calibration.** Both the ML "confident error" margins and Muse's 68–86 confidences are uncalibrated. Uncertainty routing needs a calibrated uncertainty region where Muse beats ML, and neither exists.
5. **No fresh data.** There are 0 post-window votes (cc-replay §4). Every reused outcome is development-only.

**Alternatives**
- Pairwise corrections: these were already damaging.
- Semantic profile: see finding 3.
- Distillation or pseudo-labeling: only works if the labels beat ML, which is not shown.
- Routing: see finding 4.
- All four need annotations, and the budget is spent. Only the within-source check needs none.

**Next step: within-source discrimination gate (no annotations, aggregates only)**
- **Inputs:** the existing `compare_baseline` predictions (3,079 rows, blocks 2–5), with broad source, story age and block for each row.
- **Metrics:** per block, AUC (how well score order separates two groups) for UP vs DOWN and UP vs rest, comparing HN with all non-HN sources. Also the share of HN among UPs vs among DOWNs (base rate).
- **Control:** repeat within age tertiles to separate age from source.
- **Freeze before computing:**
  - *Blind spot:* HN UP-vs-DOWN AUC is at least .05 below non-HN in at least 3 of 4 blocks, and the gap persists within age tertiles.
  - *Source prior:* the gap is under .03 or shows in fewer than 3 blocks. Then stop the HN line and the component audit.
  - In between: report the numbers, take no action.
- **Only if the gate finds a blind spot:** run the component audit restricted to HN to find which score component puts HN UPs below HN DOWNs. Then freeze one within-HN Muse pairwise design that keeps native scores and counts recovered UPs and damaged DOWN controls separately. Evaluate it only on votes after the window, once the current interleaving closes on 2026-11-05.

This is a frozen measurement from existing predictions, not a new hypothesis search. It directly tests the claim the proposed audit assumes without checking.
