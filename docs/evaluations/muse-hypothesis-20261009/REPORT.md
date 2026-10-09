# Muse familiarity hypothesis: completed retry, 2026-10-09

Export note: the evaluator copy was formatted for repository checks; its
results were not recomputed. Original run artifacts remain on the VPS.

The completed requests do not support a preference for stories with greater
title overlap with prior upvotes. A user-authorized retry recovered all missing
requests after the provider cooldown. No ML challenger or production change was made.

## Frozen method

Muse reviewed both corrections and regressions from the previous audit and
selected one hypothesis: familiarity with past upvoted titles may overshadow
useful novelty. Its proxy was the count of non-stopword title tokens shared
with the 50 most recent training upvotes. This lexical proxy is narrower than
semantic familiarity; a negative result does not rule out other forms of it.

The hypothesis, threshold, sampling procedure and code were saved before new
outcomes were inspected. A read-only, outcome-redacted query found 99 later
voted stories. The older training profile was frozen separately. Twenty-four
pairs were selected by source and title overlap, with seeded order and capped
story reuse; six reversed requests were planned. Candidate outcomes and the
familiarity classification were absent from the scoring prompts. Child requests
denied tool access and used Muse Spark 1.3 only, in real batches with caching.

No saved pre-vote ML scores covered these new stories. Current ML scores would
be contaminated by later training, so a fresh ML comparison remains unknown.

## Verified completed results

- 24/24 forward rankings and 6/6 reversed checks completed; every reversed
  choice matched its forward choice.
- Muse chose the higher-overlap story in 5/14 non-tied overlap pairs (36%),
  below the preregistered 67% threshold.
- All 24 completed pairs had matching UP-versus-rest status. There are zero
  informative pairs for that ranking accuracy measure; accuracy is undefined.
- The full candidate window contained 17 upvotes, 42 neutral votes and 40
  downvotes. The selected stories included two upvotes, paired with each other.
  This is a sampling limitation, not a
  reason to select future pairs using their outcomes.
- Repeated stories and topic matching further limit independence. No general
  model superiority or ML improvement can be inferred.

Completed selections and ranks were hashed before the fallback evaluator
opened candidate outcomes. The evaluator checked that their vote timestamps
still fell inside the frozen candidate window; none were missing or changed.
Machine-readable aggregate results are in `results.json`; `evaluate_partial.py`
is the read-only fallback evaluator. Private selection, profiles, response
caches, preregistration and scorer scripts remain on the VPS in
`/home/dev/hn-rewrite/llm-labels/muse-hypothesis-20261009/`.

## Provider recovery and next step

OpenCode returned: "OpenRouter fallback paused: credit exhausted or unverified.
Go retries after cooldown." The final forward batch and reversed batch produced
no parseable answers after bounded retries. Router inspection found an upstream
Go HTTP 500 and a five-minute cooldown; Go quota was not exhausted. The retry
completed nine missing answers in two requests, retaining the original 21.
No additional model or paid fallback was used to bypass the failure.

The retry occurred after partial outcomes had been reviewed. Sampling,
hypothesis and text-only prompts stayed fixed, and new isolated scoring sessions
had no outcome access. Original partial results and their freeze are preserved.

No challenger was built. For another experiment, preregister an outcome-blind sampling rule that
creates stronger predicted preference contrasts within topics, using a frozen
training-only scorer, and reserve untouched later outcomes. Record pre-vote ML
scores if a valid comparison becomes available. Muse access recovered; a useful
sample and a valid frozen ML baseline remain prerequisites for another experiment.
