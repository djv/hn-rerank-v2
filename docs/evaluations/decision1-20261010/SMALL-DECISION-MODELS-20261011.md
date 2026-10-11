# Small decision models for the HN profile task — survey 2026-10-11

Task shape: about 2,100-token reader profile (same within a fold) + about 250-token
story, one yes/no question, VPS CPU (4 cores, no AVX-512). Facts are from the Hub
model cards/API on 2026-10-11; speed figures are the authors' (mostly GPU) unless
marked as measured here.

| Model | Size / base | Context | Notes | Source |
|---|---|---|---|---|
| Laya-multilingual (running) | 322M mmBERT-base encoder | 8K (1K default) | measured here: 4.7 s/score at 3 threads, ~9 s real / ~0.6 s empty | https://huggingface.co/convaiinnovations/laya-multilingual |
| Kev-0.8B | Qwen3.5-0.8B-Base + LoRA, pointer head | 64K state | decoder: shared profile can be prefix-cached once per fold | https://huggingface.co/jaredpalmer/kev-0.8b |
| decider-0.8b | Qwen3.5-0.8B-Base, 752M | not in card excerpt | Jev format, same recipe as decider-2b | https://huggingface.co/Mapika/decider-0.8b |
| LightDec | 160M ettin-encoder-150m | 512 (2,048 only for >24 options) | fastest, but would truncate the profile; ECE 0.025 on its own test | https://huggingface.co/Falconsai/LightDec |
| opendecider-nano | 395M ettin-encoder-400m | not verified | CPU supported; 24 req/s on 8 CPU cores (short inputs) | https://huggingface.co/manjunathshiva/opendecider-nano |
| von | 395M ModernBERT-large | 8K | order-invariant option scoring; heavier per token than mmBERT-base | https://huggingface.co/wfzyx/von |
| MIMIR-1 | 419M ModernBERT-large | not verified | can abstain; license "other" | https://huggingface.co/Mythologic/MIMIR-1 |
| GLiNER2.5-Decide | 486M DeBERTa-v3-large | not verified | label classifier, not built for long states | https://huggingface.co/fastino/GLiNER2.5-Decide |
| systemone-lite-0.5b | Qwen2.5-0.5B | not verified | JevBench 50.7%, ECE 0.245 (weak) | https://huggingface.co/dwidlee/systemone-lite-0.5b |
| jev-typed-decisions-causal-0.6b | Qwen3-0.6B LoRA | not verified | trained on typed-decisions only | https://huggingface.co/abidlabs/jev-typed-decisions-causal-0.6b |
| StartLux-Decision 0.8B | GGUF | not verified | CC-BY-NC license | https://huggingface.co/startlux-models/StartLux-Decision-v1-0.8B-Q8_0-GGUF |

Excluded as too heavy for the VPS: 2B+ models (Strands, decider-2b/4b, Kev-4B,
lev 4B, opendecider-small 4B, clef-flash 9B, JEV 9B+).

Reading: the cost is dominated by re-encoding the ~2.1K-token profile for every
story. An encoder cannot reuse it; a small decoder (Kev-0.8B, decider-0.8b) can
encode it once per fold and score only the ~250-token story suffix. A rough FLOP
estimate (unmeasured) puts a 0.8B suffix-only pass in the same few-seconds range
as Laya on this CPU, so the speed gain is uncertain. LightDec is fast only because
it would read about 500 tokens.

## Faster than Laya (added 2026-10-11, user: "faster than laya", laptop GPU)

Laya-multilingual is mmBERT-base (22 layers, hidden 768). Same-depth models with
hidden 384 cost roughly 3-4x less per token (estimate, unmeasured):

| Model | Size / base | Context | Notes | Source |
|---|---|---|---|---|
| Julia-1 | 144M mmBERT-small | runtime up to 8,192 (benchmarks at 1,024) | noul 80.7% on its own test | https://huggingface.co/SupersonicLabs/Julia-1 |
| Mira-v1 | 140M mmBERT-small | 8,192 positions | CPU latency measured by authors on i5-1335U | https://huggingface.co/sagea-ai/Mira-v1 |
| krite-0.15b-v1 | 141M mmBERT-small | 8,192 positions | late interaction: state encoded once and KV-cached; questions/options <= 32 tokens | https://huggingface.co/skyoo2003/krite-0.15b-v1 |
| oscar-1-68m | 75M ettin-68m | trained at 512; encoder reads ~8K | typed-decisions acc 0.7455 | https://huggingface.co/mgoeckel/oscar-1-68m |
| sys1-micro | 69M ettin-68m | default 2,048 (4,096 guard) | MIT | https://huggingface.co/sulabhkatiyar/sys1-micro |

Laptop: Intel UHD 620 integrated GPU (Comet Lake-U), no CUDA; CPU i7-10510U
4c/8t, 11.5 GB RAM, capped at 2 CPUs for batch jobs. Not expected to beat the VPS
for these encoders; a smaller model is the real speed lever.

## Published benchmarks (from model cards, 2026-10-11; self-reported, not cross-checked)

typed-decisions test = LocalLLaMA/typed-decisions, 400 cases / 2,000 decisions.
Several models train on its source families (oscar says so explicitly), so it
flatters in-distribution models. None measures personal-taste prediction from
a list of story titles; only the HN pilot can show that.

| Model | Params | typed-decisions acc | Noul / other | Calibration | Speed (authors) |
|---|---|---|---|---|---|
| Laya-multilingual (zero-shot) | 322M | 0.342 (random 0.318) | card warns noul may look weak (laya issue #156) | n/a | measured here: ~9 s per 2.1K-token state, VPS CPU |
| laya-typed-decisions (fine-tuned) | 421M | 0.7685 (in-distribution) | | ECE 0.2156 | |
| Julia-1 | 144M | 0.7315 (CUDA bf16) | noul 80.7%; CPU fp32 reproduction 483/600 | ECE 0.234 (Mira card) | short query p50 321 ms CPU (Mira card) |
| Mira-v1 | 140M | 0.745 | noul slice 0.790 | ECE 0.025 | short query p50 30 ms on i5-1335U CPU; one pass per option |
| oscar-1-68m | 75M | 0.7455 (in-distribution) | | ECE 0.117 | 3.9 ms p50 GPU |
| LightDec | 160M | 0.689 | 58-task acc 0.725 | ECE 0.025 | 512-token context |
| krite-0.15b | 141M | n/a (own 5-suite macro 0.787) | | ECE 0.060 | 8 ms warm CPU at 512-token state, state KV-cached |
| typed-decider-150m | 150M | n/a | JevBench noul 79.7% (chosen on JevBench) | | CC-BY-NC |
| sys1-micro | 69M | numbers only in charts | | | T4 at 2,048 context |
| Kev-0.8B | ~0.8B | n/a | out-of-domain locked test 0.697 | ECE 0.045 | long states validated to 8K+; prefix-cacheable |
| decider-0.8b | 752M | n/a | held-out 24 tasks 0.707 | ECE 0.096 | cacheable schema-first layout |

No card reports latency at a ~2K-token state on CPU.
