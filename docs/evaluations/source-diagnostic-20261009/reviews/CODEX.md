**Verdict: worth running as a bounded descriptive diagnostic.** The supplied SHA256 matches. Clarify the following definitions before relying on results; no additional experiments are needed.

1. **MUST-FIX specification ambiguity — PLAN lines 32–44:** explicitly keep AUC, support eligibility, weights and support-mass denominators within each block. Pooling scores across blocks could introduce ordering artifacts. UP/DOWN and UP/rest need separate retained-bin sets; compare sources using identical weights within each endpoint. Pooled all-row weights are coherent, but retained support is label-dependent. The <50% rule is a reasonable descriptive convention, not assurance of representativeness.

2. **Implementation checks — lines 21–22, 42–45, 67:** freeze quantile construction, boundary inclusion, duplicate-cut handling, AUC ties receiving half-credit, and bottom-20% rounding/tie-breaking. Define DOWN placement as bottom-quintile DOWN count divided by all DOWN rows of that source/block. Preserve score ties rather than splitting them to force equal-sized bands.

3. **Interpretation limits — lines 26–29, 45–55:** fixed-clock age bins are coherent; age-at-vote sensitivity also reflects voting delay and selection. UP/rest depends on neutral/DOWN composition. These voted development rows cannot represent the untouched eligible pool, and heterogeneous non-HN composition remains a confound.

Within-source AUC is invariant to an additive source offset, so it measures discrimination separately from score location. Shared-band occupancy and score summaries can reveal relative placement; they cannot uniquely establish an inappropriate offset or mechanism.

If adequate support shows consistent discrimination gaps or placement differences, carry that hypothesis into one separately frozen, fresh eligible-pool/exposure validation; otherwise stop. Neither outcome establishes transferable Muse advantage.