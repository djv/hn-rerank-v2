# Decision-1 / Strands offline evaluation

Objective: resume cache-completeness, provenance and paired-control work offline;
keep Strands stopped, skip Colab, preserve existing data and avoid deployment.

Verified result: full-scope real/empty preparation and strict cache/report gates
implemented. All 104 legacy scores and the frozen snapshot remain unchanged.
4,132 jobs prepared; no validated paired scores exist. Actual partial-cache
reporting is refused. Backend 1,362 passed / 1 skipped; Ruff/format pass;
no new typing diagnostics. Evidence: [SAFETY-GATES.md](SAFETY-GATES.md).

Blocker: no authorized fast scoring route; removed model environment stays absent.
Twenty-six frozen candidates lack comment counts, so strict popularity controls
also refuse that missing data. No held-out quality/accuracy result is available.

Next step: retain the stopped state until another scoring route is authorized;
then complete both profiles, resolve the missing-comment control policy and use
the fixed .2 blend and paired comparison. Dating labels/Random100 remain untouched.
