# HN ranking research status

Updated 2026-10-09. Preparation history: PREPARATION-STATUS.md.

Objective: find repeatable Muse advantages that improve ML upvote discovery,
using cheap Muse and existing annotations wherever possible.

Verified result: comment-free primary embedding comparison completed and failed
all three improvement gates. AUC .779843 → .778184; top-48 upvotes 29 → 29,
downvotes 1 → 1. Two of four blocks improved; delta interval includes zero.
Baseline reproduced exactly; parity20 and input-integrity checks passed.
Both task jobs exited. No production changes or new ranking annotations.
Exact experiment source: harness_frozen.py. Complete evidence: REPORT.md.

Decision: stop representation rewriting/canonicalization on this evidence.
User-selected missed-upvote analysis completed using existing predictions and
cached Muse annotations; current handoff: ../missed-upvotes-20261009/STATUS.md.

Limitation: reused exploratory outcomes; no verified ranking gain.
Next: inspect native score components on matched HN UP/DOWN controls before
choosing a correction design and fresh-vote validation.
