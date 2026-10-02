# HN Rerank status

Saved 2026-10-02 16:45 UTC. Previous project handoff:
[before this review](docs/status-archive/before-ranking-review-handoff-20261002.md).

## Objective
Review the recent Gemma, evaluation and attribution changes, and save the
findings plus Claude Code's assessment. Runtime fixes remain unrequested.

## Verified result
- Three defects reproduced with in-memory test databases, still present at
  `50abce9`: production evaluation omits Gemma side vectors (P1); the side
  encoder can miss replacement candidates when a source cap fills (P2);
  yield reports count `ch_seed`/`bq_seed` as non-HN (P2).
- Claude Code, invoked directly on the command line with read-only tools,
  confirmed #1. Earlier Gemma gains are unaffected by this defect: those
  runs explicitly joined stored and Gemma replay files, with the side flag
  off. Their broader offline/live limitations still apply.
- Review checks: 241 affected tests passed / 18 skipped; Ruff and touched
  Python formatting passed. The interrupted full run reached 956 passed /
  18 skipped / one cache timeout; that test passed alone.
- Durable reproduction recipes, outputs, validation boundaries and CC's
  proposed correction: FINDINGS.md, "Ranking review and direct Claude Code
  assessment — 2026-10-02". No application or production-data changes made.

## Blocker / limits
- Default `production` evaluation currently measures stored-only ranking
  even with the live side flag on; restore parity before using it to assess
  the deployed Gemma ranker.
- Full-suite completion was stopped under high laptop load. Type checking
  found only the existing untracked `scripts/inspect_tldr_failures.py:86`
  diagnostic. The affected-test results precede the concurrent startup-warm
  change `7cdb742`; they do not validate that change.
- Unrelated WIP preserved: `clients/tui/tests/test_client.py`,
  `docs/mockups/`, `scripts/inspect_tldr_failures.py`, `kernel.errors.txt`.

## Next step
- Resume fixes only on request. Start with a side-vector source/fold parity
  regression, then supply fold side vectors, detect unexpected fallback,
  and reject ambiguous replay-plus-side configurations.
- Follow with the capped-candidate encoder and canonical HN source reporting
  regressions. Keep all tests on temporary/in-memory databases.
- Existing live-yield follow-up, Popular decision, reader mockup, LLM reset
  and parked TLDR work remain in the linked previous handoff; preserve their
  authorization boundaries.
