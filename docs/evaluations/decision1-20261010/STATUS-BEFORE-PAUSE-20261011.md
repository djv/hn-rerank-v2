# HN paired scoring — CPU speed gate, 2026-10-11

## Objective
Find a VPS scoring route below5seconds per model score including parity-check
overhead, as selected by the user. Keep the cancelled full run stopped meanwhile.

## Verified result
Read-only audit:56 valid v3 scores,0 invalid,4076 missing out of4132. No held-out
quality result. All CPU probes publish zero scores; scoring-worker source, frozen
input, caches, installed packages and live service remain unchanged.

Length bucketing saves9.22% padded suffix tokens but only about4.04% total tokens
with batch4/check1. Actual-model profiling places92.05% of a short cached request
in Linear layers. Convolution and block-inverse tweaks do not provide a major win.
Dynamic int8 rejected: cached/plain drift0.3178 exceeds1e-4.

Exact-model batch16/check1 probe on sixteen short validated real requests passed
all reference checks and sampled plain parity, but took13.113s/score excluding
startup/prefix. That reduced parity fraction is a measurement, not an adopted
production policy. Packed FP16 with three threads and a5.5GiB hard cap passed
all16 reference comparisons plus sampled plain check, but took8.991s/score on the
same favorable real sample. Mixed empty batch4/check1 took17.023s/score. All owned
probe units stopped/PID0; full scorer inactive/PID0, no worker/evaluator remains.
Live HN PID531386/start03:26:55UTC unchanged. No production setting was adopted.

## Blocker
No verified route meets the user's below5s target. Runtime/parity coverage must be
accepted before a new full launch; no quality or winner claim from partial scores.

## Next step
Choose whether to test an older release's q8 ONNX CPU variant (changes the model)
or retain the exact model and investigate other hardware. A bounded, zero-publication
proposal is saved in CPU-ONNX-PILOT-PLAN-20261011.md; it has not been approved.
Do not automatically
resume full scoring. Paid GPU/API, Colab, confirmation access and deployment remain
unauthorized. Evidence: CPU-SPEED-20261011.md and SCORING-20261011.md.
