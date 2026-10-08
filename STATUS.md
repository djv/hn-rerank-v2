# HN Rerank status

Updated 2026-10-07. Previous handoff (including the Gemma 2 future-vote
recheck): [TUI sharing / Reddit](docs/status-archive/before-why-story-20261007.md).

## Objective
Evaluate one preference classifier replacing both dense logistic regression
and the word model. User explicitly selected removing both, requested other
embedding models/settings, and asked for Claude's opinion. Existing Why-this-
story preview and production model weights remain unchanged during research.

Execution update: user superseded VPS-only work, then explicitly requested
remaining experiments on the laptop GPU. BGE full256/title128 and Granite
Small English R2 encodings finished on Intel UHD.
Process-specific counters verify GPU use. Scikit-learn classifier fits are
CPU-only and ran locally with one CPU/background priority on AC; there
is no GPU training backend for these estimators on the Intel UHD.

Research complete: classifier/settings/features and fourteen matched encoder
comparisons finished. Twelve chronological blocks compare production
(AUC .8182,94 liked/2 disliked per144 top picks), one joined classifier
(.8295,97/6), no words (.8319,97/6), and no metadata (.8203,98/5).
Removing words preserves counts but lowers NDCG; removing metadata raises
NDCG while lowering recent AUC. Labels were reused during selection;
quality equivalence, live benefit and serving savings remain unverified.
User requested tradeoffs only: no production replacement selected.
Latest user preference: candidates #2 (one classifier, all features) and
#4 (the same classifier without metadata) look best. Keep both as the
shortlist, not a deployment selection. #2 has stronger overall/recent AUC;
#4 has higher NDCG@12 and98 liked/5 disliked versus97/6, but recent AUC
falls .8366→.8064. These point estimates do not establish superiority.

Two read-only Claude Opus5.5 reviews completed. The statistics CLI label
collision was fixed and routing/feature/probability/dedup tests added.
All old-driver and twelve-block selected-baseline reruns reproduced every
ID, label and score exactly, with stable source/input hashes. Four-test
Holm adjustment is complete (adjusted p .252–.785). The full suite passed
1,128 tests /18 skipped; Ruff and all21 touched Python formatting checks
passed. Type checking retains only the existing untracked inspection-script
diagnostic. No experiment job remains running.
Saved implementation/evidence commit28afc5c is pushed; GitHub backend and
Terminal client CI workflows both completed successfully.
[Feature report](docs/evaluations/model-ablation-20261007/FEATURE-REMOVALS.md).
[Full report](docs/evaluations/model-ablation-20261007/BROAD-ONE-CLASSIFIER.md).

Laptop TUI incident: public Funnel TLS connections failed while the VPS
application stayed healthy. Existing Funnel configuration was refreshed
and VPS tailscaled restarted. At22:21 ET, all three advertised public edges
returned200 from both VPS and laptop. The unchanged saved TUI profile
fetched46 stories and passed five independent authenticated requests.
No private route or production SSH tunnel is required. Logs show some
internal ingress drops; exact root cause remains unconfirmed. Profile
creations during anonymous diagnostic checks are not organic signups.

## Result
- Kagi News World RSS added to the local `config.toml` feed list. VPS
  validation: config parses, URL listed once, 12 valid live RSS items.
  Deployed only this feed-list addition to VPS production; service restart
  succeeded and dashboard smoke check returned HTTP 200.
- Shared Why-this-story labels renamed for Web/TUI, with algorithms and
  weights unchanged. VPS focused tests: 10 passed; full backend:
  1,111 passed / 1 skipped. Ruff, touched-file formatting and ty pass.
  Isolated preview restarted; live feed shows both new names. Refresh the
  existing preview page. Production remains unchanged.
- Broader word-model check: eight historical blocks lose 2/96 top-12
  upvotes without word; the four recent blocks gain 1/48. Across twelve
  judged blocks, full blend vs no-word is 94/144 vs 93/144 upvotes,
  AUC .8182 vs .8162, and 2/144 vs 4/144 downvotes. Recent shown/unvoted
  pool slightly favors no-word too. Effect is small and changes by period;
  no consistent improvement or exact preservation is established.
  [Broader evidence](docs/evaluations/model-ablation-20261007/BROADER.md).
- Recent-vote ablation: full blend 23/48 top-12 upvotes, AUC .820;
  remove word 24/48, .827; remove base 22/48, .808; remove content
  20/48, .815. Removing word is the clearest simplification candidate,
  but uncertainty spans no improvement; equivalence is not established.
  [Paired evidence and artifacts](docs/evaluations/model-ablation-20261007/README.md).
  No production weights, code or DB were changed by the experiment.
- Implemented web expander and TUI `w` panel; Escape restores the summary.
- Show active model-blend signals that helped/hurt relative to a
  middle-of-pool rating, with actual candidate percentiles and weights.
  Distinguish these from related upvotes and badge/discovery signals;
  Popular explains its points/age ordering rather than preference factors.
- Up to three distinct close upvotes retain raw/centered and badged-card
  safeguards. No LLM calls, new dependencies, or schema changes.
- Full backend: 1,093 passed / 18 skipped; both real-browser tests passed.
  Full TUI on VPS: 172 passed / 1 skipped. Ruff and formatting pass; `ty` retains
  only the existing untracked TLDR inspection-script diagnostic.
- Production patch preflight passes against the clean VPS checkout.
  The feature is currently local; production service is still running
  the prior code. No production database was changed by this work.

## Next step
- Retain user-shortlisted candidates #2 and #4 for any subsequent comparison;
  no new evaluation or production deployment was requested by this preference.
- No model removal implemented. Historical/exposure-pool checks are done;
  a predeclared prospective comparison would be needed to validate
  equivalence before treating the simpler blend as proven.
- User can test the Web preview now; wait for feedback before deployment.
- User requested testing without deployment. Separate VPS checkout:
  /home/dev/hn-why-preview-20261007; temporary preview service
  hn-why-preview-20261007.service on 8767, forwarded to laptop localhost.
  Preview uses a fresh isolated DB snapshot and cached summaries only.
  Verified ready: http://127.0.0.1:8767/u/local-why-preview (47 stories with
  model factors, 21 with related upvotes in 1w). User chose Web preview.
  Keep preview/tunnel running for testing; full TUI suite passed on VPS.
- Deploy only the six intended runtime files and restart/smoke-test the
  service after deployment authorization. Keep unrelated WIP untouched.
- Pending separate work: Reddit replacement before November 13, 1m
  Popular age-mix user check, real-terminal tint/footer check, and the
  October 21 Gemma 2 future-vote recheck; details are in the prior handoff
  and /home/d/TASKS.md.
