# Parameter effects: what each knob buys, and on which axis

Measured with `llama-batched-bench` (8192-token prompt) and `llama-server` over HTTP
(107,743-token prompt), one fresh process per point, on an RTX 3070 (8 GiB) + RTX 5060 Ti
(16 GiB) pair — the 5060 Ti drives the display — with a 20.2 GiB Q4_K_M MoE model
(35.5 B total / ~3 B active, 256 experts, 8 used, hybrid attention, 40 layers + 1 MTP head).

Use these as **shapes to expect**, not absolute targets. Differences under ~8% are noise on a
machine whose GPU also drives a display.

> ## ⚠️ The shapes are MODEL-CLASS-specific. Re-measure; do not assume.
>
> An earlier version of this file said "the shapes generalise; the numbers do not." **That was
> wrong.** Running the same campaign on a *dense* 27 B model (Qwen3.8-27B-Q4_K_M, 64 layers +
> MTP head) on the same box contradicted **five** of the findings below:
>
> | Finding from the MoE | On the dense 27 B |
> | --- | --- |
> | KV cache is ~4× *smaller* than the naive formula | **3× LARGER than the MoE's.** 16 full-attention layers at `n_embd_k_gqa=1024` = 64 MiB f16 per 1024 tokens → f16 at 130k needs **8128 MiB** and is impossible; f16 caps at ~72k |
> | `-ub` is the dominant prefill knob (1368→2651) | **Flat**: 1141–1222 across 256→2048. Only 128 costs anything (−22%) |
> | `q4_0` KV is *faster* (frees headroom for a bigger `-ub`) | **Not faster**: 1187 vs 1208. It buys 2032 MiB and nothing else |
> | Non-pipelined fallback: +43% prefill | **0%** (1208 vs 1219) — though still worth 422 MiB of compute buffer |
> | Context taxes prefill (−31% at 130k) | **No tax**: 1208 t/s at 130k vs 1202 at 8k |
> | `draft-mtp`: −7% to −29% | **+110%** (22.4 → 47.1 t/s at `n-max 4`) |
>
> The one thing that did transfer: **the *method*.** Predict → measure at the real `-c` → one
> fresh process per point → quality-gate → grep the logs. Every specific number needed
> re-measuring, and the *sign* of two effects flipped.
>
> **So: treat this file as a worked example, not a lookup table.** Establish the model's class
> first — dense vs sparse-MoE, and how many layers are genuinely full-attention — because that
> is what decides which knobs matter.

## Contents

- [Axis reminder](#axis-reminder)
- [`-ub` — the micro-batch](#-ub--the-micro-batch)
- [`-c` — the context/speed frontier](#-c--the-contextspeed-frontier)
- [`-ts` — the split boundary](#-ts--the-split-boundary)
- [`-ncmoe` — CPU expert offload](#-ncmoe--cpu-expert-offload)
- [`-ctk` / `-ctv` — KV cache type](#-ctk---ctv--kv-cache-type)
- [`-fa` — flash attention](#-fa--flash-attention)
- [Pipeline parallelism](#pipeline-parallelism)
- [Synthetic vs real throughput](#synthetic-vs-real-throughput)
- [Quantisation level](#quantisation-level)
- [Summary: gain per knob](#summary-gain-per-knob)

---

## Axis reminder

**PP (prefill)** is compute-bound — all prompt tokens process in parallel, so batching helps.
**TG (generation)** is memory-bandwidth-bound — one token at a time, so batching is
irrelevant. A knob that moves PP by 30% will often move TG by 0%.

---

## `-ub` — the micro-batch

**With memory to spare** (small context) the curve is simple:

| `-ub` | PP t/s | TG t/s | Compute buffers |
| --- | --- | --- | --- |
| 128 | 1368 | 106 | ~0.4 GiB |
| 256 | 2017 | 102 | ~0.8 GiB |
| **512** | **2651** | 105 | ~1.6 GiB |
| 1024 | 2615 | 106 | ~3.2 GiB |
| 2048 | OOM | — | ~6.5 GiB |

512 is the peak; 1024 doubles memory for nothing. **TG is flat across the whole range** —
generation has no batch, so never spend VRAM on `-ub` for a chat workload.

**At the memory ceiling** (130k context) the ranking inverts, because a high `-ub` makes the
pipelined reserve fail and forces the faster lean path:

| KV | `-ub` | PP t/s | TG t/s |
| --- | --- | --- | --- |
| `q4_0` | 256 | 1616 | 102 |
| `q4_0` | 512 | 1787 | 104 |
| **`q4_0`** | **1024** | **2650** | **105** |
| `q4_0` | 1536 / 2048 | fails | — |
| `q8_0` | 256 | 1466 | 95 |
| `q8_0` | 512 | 1850 | 98 |
| `q8_0` | 1024 | fails | — |

**Lesson:** the best `-ub` depends on how much memory the rest of the configuration leaves.
Sweep it *with the real `-c` allocated*, never at a convenient small context.

---

## `-c` — the context/speed frontier

Near the memory ceiling, context and prefill trade against each other steeply.
`q8_0`, `-ub 512`:

| `-c` | PP t/s | TG t/s |
| --- | --- | --- |
| 122,880 | 2225 | 106 |
| **126,976** | **2323** | **109** |
| 129,024 | 2101 | 106 |
| 130,048 | 1850 | 98 |

**Giving up 2.4% of the window bought 26% more prefill.** Right-sizing `-c` returns memory
*and* speeds prefill up — one of very few genuinely free wins.

> **Correction worth knowing.** An earlier reading of this data claimed a smooth ~31%
> "context tax" from the attention mask growing with allocated context. That was mostly an
> artefact of running out of memory at the top end, not a law. Once the configuration fits
> comfortably, 130k prefills at essentially the same speed as 8k (2650 vs 2651 t/s). Do not
> tell users that large contexts are inherently slow — tell them large contexts are hard to
> *fit*, and that fitting is what costs speed.

---

## `-ts` — the split boundary

At small context, more layers on the smaller card kept helping:

| `-ts` | GPU 0 share | PP t/s |
| --- | --- | --- |
| 6/18 | 25% | 2244 |
| 8/18 | 31% | 2541 |
| 10/18 | 36% | 2883 |

At 130k the memory limit arrives first and the window collapses to one value:

| `-ts` | Result |
| --- | --- |
| 11,30 | fails |
| **12,29** | **works** |
| 13,28 | OOM |
| 14,27 | OOM |

`11,30` fails despite putting *less* on the small card — it moves that layer onto the
display GPU, which has less free memory. **The optimum is a memory boundary, not a compute
one. Start from free VRAM, not capacity.**

Proportional-to-free-memory arithmetic suggested `15,26` here — three layers above the
measured optimum — because compute buffers do not split by the `-ts` ratio; every device
carries a near-full one. **Sweep downward from the arithmetic suggestion.**

---

## `-ncmoe` — CPU expert offload

`-c 130048`, `-ub 512`, `q8_0` KV:

| `-ncmoe` | Layers on CPU | PP t/s | TG t/s |
| --- | --- | --- | --- |
| 2 | 2 / 40 | 826 | 88 |
| 4 | 4 / 40 | 541 | 82 |
| 6 | 6 / 40 | 421 | 76 |
| 8 | 8 / 40 | 344 | 60 |

Against the all-GPU best of **2650**, `-ncmoe 2` costs **69% of prefill for 5% of the
model**. Prefill activates nearly all 256 experts per batch; generation activates 8 — so the
CPU becomes the bottleneck while both GPUs idle. Prefill degrades about twice as fast as
generation.

**`-ncmoe` is a last resort, never a tuning knob.**

---

## `-ctk` / `-ctv` — KV cache type

`-c 130048`, best `-ub` for each:

| Pair | KV size | PP t/s | TG t/s | Fast kernel? |
| --- | --- | --- | --- | --- |
| `f16`/`f16` | 2622 MiB | doesn't fit | — | ✅ |
| **`q8_0`/`q8_0`** | 1422 MiB | 1850 | 98 | ✅ |
| **`q4_0`/`q4_0`** | ~720 MiB | **2650** | **105** | ✅ |
| `q5_1`/`q5_1` | ~1100 MiB | 107 | 19 | ❌ |
| `q4_0`/`q8_0` | ~1050 MiB | 160 | 25 | ❌ |

The bottom two rows load cleanly, produce correct output, and are **14× slower**. This is
not "4-bit is slow" — it is "that *pair* has no compiled kernel". Two distinct faults:
asymmetry, and unsupported type. See trap 2 in `known-traps.md`.

Note the top of the table: **the smaller cache was faster**, because it freed the headroom
for `-ub 1024`. A KV type is not just a memory choice; it changes what `-ub` you can afford.

**Quality:** `q4_0`/`q4_0` **passed** a 108k-token needle-retrieval test (see
`needle-test.ps1`), at 1758 t/s real prefill versus `q8_0`'s 1414. That clears it for
retrieval-style work; it does not prove it lossless for reasoning or code edits.

KV size on this hybrid model was ~20 MiB per 1000 tokens at f16, ~11 MiB at `q8_0` — roughly
**4× smaller** than the standard formula predicts, because only about a quarter of the layers
are full-attention.

---

## `-fa` — flash attention

| Setting | Compute buffers at 130k |
| --- | --- |
| `-fa on` | 1.6 GiB |
| `-fa off` | **10.4 GiB** |

Nine gigabytes for no benefit. Always `-fa on`, explicitly rather than via `auto`.

---

## Pipeline parallelism

With >1 GPU, llama.cpp keeps 4 copies of intermediate activations so devices can overlap.
On a **mismatched** pair this is a double loss: the pipeline stalls on the slower card, and
the extra copies break the fit.

`-c 130048`:

| Path | PP t/s | TG t/s |
| --- | --- | --- |
| pipelined | 1850 | 98 |
| **non-pipelined (lean)** | **2650** | **105** |

Reproducible across three runs: 2757 / 2599 / 2594 (±3%).

> ### ⚠️ `GGML_SCHED_MAX_COPIES` is a compile-time `#define`, not an environment variable
> ```c
> #ifndef GGML_SCHED_MAX_COPIES
> #define GGML_SCHED_MAX_COPIES 4
> #endif
> ```
> Setting `$env:GGML_SCHED_MAX_COPIES = "1"` **does nothing**. An earlier version of this
> reference credited a 13% gain to it; that was run-to-run variance on a machine with an
> ±8% noise floor.

Two real ways to get the lean path:

1. **Rebuild** — `cmake -B build -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1`. Deterministic.
2. **Raise `-ub` until the pipelined reserve fails**, and let llama.cpp fall back:
   ```
   sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
   ```
   This works and is reproducible under stable VRAM conditions, but you are depending on an
   allocation *failing* — free 1.4 GiB and the slower pipelined path may succeed instead.

---

## Synthetic vs real throughput

Same configurations, 8192-token synthetic prompt versus a 107,743-token real HTTP request:

| Config | Synthetic PP | Real PP | Synthetic TG | Real TG |
| --- | --- | --- | --- | --- |
| `q4_0`, `ub 1024` | 2650 | **1758** | 105 | **46.6** |
| `q8_0`, `ub 512` | 1850 | **1414** | 98 | **47.5** |

**Generation roughly halves at depth** (98–105 → ~47 t/s). Prefill drops ~25–35%. Any
"tokens per second" figure quoted without a prompt length is a shallow-context number.
Always finish a campaign with a real deep request.

---

## Quantisation level

| Model | Size | TG t/s |
| --- | --- | --- |
| Q3_K_XL | 15.69 GiB | ~94 |
| **Q4_K_M** | **20.21 GiB** | **117** |

**The larger file generated 24% faster.** Q3_K matrix kernels on CUDA are slower than Q4_K.
Choose a quant for quality and fit; measure speed rather than assuming it tracks file size.

---

## Summary: gain per knob

For an agentic coding workload, best value first:

| Optimisation | Gain | VRAM cost |
| --- | --- | --- |
| Prompt caching (`-np 1`, `-cram`) | **~34× on repeat prefill** | 0 (host RAM) |
| Not overflowing VRAM | up to 5× | 0 |
| Non-pipelined path on mismatched GPUs | **+43% PP** | **frees memory** |
| `q4_0` KV enabling a larger `-ub` | +43% PP | **frees 700 MiB** |
| Right-sizing `-c` (−2.4%) | +26% PP | **frees memory** |
| `-ub` one step up (when it fits) | +21% PP | +819 MiB |
| Optimal `-ts` vs naive | +10–20% PP | 0 |
| Speculative decoding (MTP) | ~+17% TG | ~530 MiB |

The largest wins are not exotic: **do not overflow, do not over-allocate context, and reuse
work already done.**

