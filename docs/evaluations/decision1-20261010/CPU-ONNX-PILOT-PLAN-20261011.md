# Proposed next CPU pilot — requires model-change approval

## Decision

Test the published q8 ONNX conversion of the older
`StrandsAgents/strands-decider-2B-hobson-v19` release on the existing VPS. This
changes the model under evaluation from the pinned qwen3.5-v1-2610 release, and
uses quantized merged weights. It is not an optimization proven equivalent to
the existing model. The user has not yet authorized this change of experiment.

Published conversion and measured limitations:
https://huggingface.co/onnx-community/strands-decider-2B-hobson-v19-ONNX
The author reports that q8 uses8bit block-quantized decoder weights and4bit
embeddings (1.80GB), and ONNX Runtime's CPU provider can run it. Published speed
results are WebGPU/Mac measurements, not VPS benchmarks. They do not establish
the below5second target or HN ranking quality. q4f16 is excluded because the
author reports more answer changes and probability drift.

## Bounded work if approved

1. Pin the exact converter revision and verify the required files and hashes.
   Download only the q8 graph/external data, tokenizer, configuration and public
   conversion/reference cases directly on the VPS. Reserve at least1.5GB free
   disk space; stop if the roughly1.8GB download cannot fit. No old files deleted.
2. Use an isolated `uv` environment and compatible pinned ONNX Runtime CPU
   package, without altering the existing scorer environment or live HN service.
   Check CPU operator compatibility first; unsupported operators stop the pilot.
3. Reuse16 frozen held-out requests (eight real/empty pairs across short, middle
   and long suffixes). Use existing prompt fitting/rendering; verify option/answer
   positions, attention masks and noul temperature against published reference
   cases. No access to confirmation labels or mutable live database.
4. Measure loading, per-batch real/empty inference and independently evaluated
   batch-one checks on every pilot row. Compare same-runtime batch and standalone
   probabilities at1e-4; score drift from the current release is not a parity
   check because this is a different model. Rejection reasons remain explicit.
5. Use at most3CPU threads, Nice15, a5.5GiB hard memory ceiling,128MiB swap cap,
  20minute overall timeout and no restart. Monitor other services and available
   memory; stop only the owned job on resource pressure or failed prerequisites.
6. Publish zero evaluation scores. Save technical timing/parity evidence. Do not
   resume full scoring, change production, book paid hardware, or mix the new
   model with the56 original cached scores. A complete new model/backend identity
   and pairing plan would be required before later efficacy evaluation.

## Acceptance gate

The target remains below5seconds per model score including the explicitly stated
parity overhead, with representative lengths and both preference conditions.
Passing speed does not establish HN ranking quality. Any later full run needs its
own measured runtime estimate, agreed check policy and authorization.

An optional independent Claude cloud plan review can assess model substitution,
prompt/temperature correctness, numerical checks and resource limits before this
pilot. It requires separate approval; no additional review is currently running.
