# 🔴 KEY FINDINGS — read this first

Everything important learned while tuning a 20.2 GiB MoE model onto an RTX 3070 + RTX 5060 Ti
(24 GiB total, mismatched, display on the 16 GiB card). Every claim here is measured on this
machine, with a log file behind it in [`benchmarks/`](benchmarks/).

**The headline: prefill went from ~300 t/s to ~2650 t/s. That is ~8.8×, and none of it came
from buying hardware.**

---

## 🥇 The single biggest win has nothing to do with flags

**Prompt caching is worth more than every other optimisation combined.**

```
57,559-token request, cold           →  32.5 s
Same prefix + a small edit           →   0.96 s   (790 tokens actually evaluated)
```

**A 34× reduction in real wait, for zero VRAM.** It needs `-np 1` (so a long prompt fits in
one slot) and a generous `-cram`. Get this right before touching anything else.

Watch for it in the server log: `f_sim_best = 0.987` — above 0.95 means it is working.

---

## 🚨 On Windows, a model that doesn't fit does NOT fail. It gets slow.

Since NVIDIA driver 536.40, a CUDA allocation past VRAM **silently spills into system RAM**
instead of erroring. You get a 3–10× slowdown and no message.

**This was the whole original mystery:**

```
LM Studio:      ~300 t/s prefill,  CPU 47% busy,  both GPUs ~0-6% utilised
Raw llama.cpp: ~2900 t/s prefill
```

Nothing was broken. One of them had overflowed and the driver hid it.

> ### FIX THIS BEFORE TUNING ANYTHING
> **NVIDIA Control Panel → Manage 3D Settings → CUDA - Sysmem Fallback Policy →
> "Prefer No Sysmem Fallback"**

Until that is set, **every measurement you take is untrustworthy**, because "slow" and
"overflowing" look identical. We caught two configurations doing exactly this mid-campaign
(GPU1 pinned at 16005/16311 MiB, prefill collapsed to 1036 t/s).

**Corollary: always record peak VRAM next to every throughput number.**

---

## ⚠️ The KV cache trap that costs 14× and shows no error

A stock CUDA build compiles flash-attention kernels for **only four symmetric K/V pairs**:

```
f16/f16      bf16/bf16      q8_0/q8_0      q4_0/q4_0
```

Anything else falls off the fast path and **attention leaves the GPU entirely**:

| Setting | Prefill | Fault |
| --- | --- | --- |
| `-ctk q8_0 -ctv q8_0` | 1850 t/s | ✅ |
| `-ctk q4_0 -ctv q4_0` | 2650 t/s | ✅ |
| `-ctk q4_0 -ctv q8_0` | **160 t/s** | **asymmetric** — `K->type != V->type` |
| `-ctk q5_1 -ctv q5_1` | **107 t/s** | **symmetric, but `q5_1` is not a supported type** |

**Symmetry is necessary but not sufficient.** The pair must be one of those four.
`fattn.cu` enforces it: `if (K->type != V->type) return BEST_FATTN_KERNEL_NONE;`

These configurations **load cleanly and produce correct output**. The only visible symptom is
gigabytes appearing in *host* RAM (5277 MiB instead of 793).

> **NEVER mix KV types. NEVER use `q5_1`, `q5_0`, `q4_1`, or `iq4_nl`** — *on a stock build.*
> Want asymmetric (precise keys, compact values)? Rebuild with
> `-DGGML_CUDA_FA_ALL_QUANTS=ON`.

### ✅ We built it. The trap is a build option, not a property of the formats

**[Chapter 14](14-build-experiment.md)** compiled `GGML_CUDA_FA_ALL_QUANTS=ON` and measured it
against a control build from identical source, compiler and CUDA:

| `-ctk q8_0 -ctv q4_0` | Prefill | Generation |
| --- | --- | --- |
| CONTROL (stock flags) | 38.9 t/s | 6.2 t/s |
| TREATMENT (`FA_ALL_QUANTS=ON`) | **1187 t/s** | **22.0 t/s** |

**×30.5 prefill, and afterwards indistinguishable from the symmetric pairs** (1178 t/s). The
penalty is not reduced, it is gone — so on a build with this flag, asymmetric KV is *free* and
`q8_0` keys + `q4_0` values becomes the natural 130k layout.

`q5_1` and `q4_1` stop being traps on that build too: every K/V pair reports the same 147 MiB
host compute, because all **49** kernels are compiled instead of **4**. Still unbanned rather
than recommended — there is no quality evidence for any of them.

> 🚫 **Do not read ×30.5 as "the rebuild is faster".** It is a penalty removed from a config
> this page told you never to use, so it gains you nothing until you switch to that config —
> and that is a *memory* decision. Best-against-best, both at `-ub 1024`: control
> `q8_0`/`q8_0` 1257.91 t/s / 22876 MiB vs treatment `q8_0`/`q4_0` 1255.55 t/s / 21860 MiB.
> **Same speed, ~1 GiB less VRAM.** The entire return is headroom
> ([§14.9](14-build-experiment.md)). The cheap throughput win on this model is unrelated to any
> flag: **`-ub 1024` instead of `-ub 512`, worth 3.1% on the binaries you already have.**

---

## 💡 A SMALLER KV cache can be FASTER

Counter-intuitive and worth internalising: `q4_0` KV is ~700 MiB smaller than `q8_0`, and that
freed headroom is exactly what lets `-ub` go from 512 to 1024 — which in turn triggers the
fast non-pipelined path. **The two effects compound:**

| KV | `-ub` | Prefill |
| --- | --- | --- |
| `q8_0` | 512 | 1850 t/s |
| `q4_0` | 512 | 1787 t/s |
| **`q4_0`** | **1024** | **2650 t/s** |

**A KV type is not just a memory choice — it changes what `-ub` you can afford.**

And it survived a quality gate: `q4_0` **passed** a 107,743-token needle-retrieval test
(details below), at 1758 t/s real prefill versus `q8_0`'s 1414.

---

## 🔬 Benchmarks cannot see a quality regression — so gate it

`q4_0` buys 24% more prefill. No throughput benchmark on earth would notice if it had started
losing the middle of your documents. So we hid an unguessable fact at the **start** of a
3000-record monotonous log and asked for it after ~108,000 tokens:

```
RECORD 0001: The commissioning passphrase for reactor bay 7 is CRIMSON-PELICAN-4417.
RECORD 0002..3000: routine telemetry; no anomalies ...
Question: what is the commissioning passphrase for reactor bay 7?
```

| Config | Needle | Real prefill | Real generation |
| --- | --- | --- | --- |
| `q8_0`, `ub 512` | **PASS** | 1414 t/s | 47.5 t/s |
| `q4_0`, `ub 1024` | **PASS** | 1758 t/s | 46.6 t/s |

Both retrieved it exactly. **This clears `q4_0` for retrieval-style work — it does not prove
it lossless** for multi-step reasoning or code edits. Reproduce with
[`needle-test.ps1`](../.claude/skills/tuning-llamacpp-configs/scripts/needle-test.ps1).

**Methodology trap:** the first two attempts returned *empty answers for both KV types*,
which looked like total quality collapse. Cause: the raw `/completion` endpoint makes an
instruction-tuned model emit EOS immediately. **A failure that hits every configuration
equally is a bug in your harness, not a finding.**

---

## ❌ CORRECTION: `GGML_SCHED_MAX_COPIES` is NOT an environment variable

An earlier version of this wiki told you to set `$env:GGML_SCHED_MAX_COPIES = "1"`.
**That does nothing.** It is a compile-time `#define`:

```c
#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif
```

The proof was in our own logs: a run **with the variable set** still allocated 1407 MiB of
pipelined buffers before falling back. The "+13%" originally credited to it was variance on a
machine with an ±8% noise floor.

> **Lesson: confirm a knob exists at runtime before crediting a measurement to it. A gain
> near your noise floor is not evidence of anything.**

---

## 🔀 Pipeline parallelism is a LOSS on mismatched GPUs

With >1 GPU, llama.cpp keeps 4 copies of intermediate activations so devices can overlap. On
a mismatched pair that is a double loss: the pipeline stalls on the slower card, **and** the
extra copies are what break the fit.

| Path | Prefill | Generation |
| --- | --- | --- |
| pipelined (default) | 1850 t/s | 98 t/s |
| **non-pipelined** | **2650 t/s** | **105 t/s** |

Reproducible: 2757 / 2599 / 2594 across three runs (±3%).

We found it by accident — a run whose pipelined buffers didn't fit printed
`retrying without pipeline parallelism` and then *beat* every properly-pipelined run.

> **When your error-recovery path outruns your happy path, adopt the error-recovery path.**
> Deliberately: rebuild with `-DGGML_SCHED_MAX_COPIES=1`. Accidentally (what we do today):
> raise `-ub` until the pipelined reserve fails.

---

## 💀 CPU expert offload destroys prefill on an MoE

The "obvious" way to fit a too-large MoE model is `-ncmoe`. Don't.

| `-ncmoe` | Layers on CPU | Prefill | vs all-GPU |
| --- | --- | --- | --- |
| 0 | 0 / 40 | **2650 t/s** | — |
| 2 | 2 / 40 | 826 t/s | **−69%** |
| 8 | 8 / 40 | 344 t/s | −87% |

**Two layers out of forty — 5% of the model — costs 69% of your prefill.** Generating a token
activates 8 of 256 experts; prefilling a 512-token batch activates nearly all of them, so the
CPU becomes the bottleneck while both GPUs idle.

> **`-ncmoe` is a last resort, never a tuning knob.** Shrinking `-ub` is far better.

---

## 🐌 Speculative decoding makes THIS model slower

Despite the vendor claiming 1.17× for MTP on 35B-A3B:

| Config | Generation | vs baseline |
| --- | --- | --- |
| **baseline** | **97–100 t/s** | — |
| `draft-mtp` n-max 1 | 90.1 t/s | **−7%** |
| `draft-mtp` n-max 2 | 70.3 t/s | **−29%** |
| `ngram-simple` n 2/4/8 | 96.6–97.6 t/s | 0% (noise) |

And the drafting was working *well* — **67–72% acceptance, 2.34 tokens per step.**

Each drafted token costs roughly a full forward pass, and this model's pass is already
extremely cheap (~3 B active of 35.5 B). There is nothing to amortise. Cost scales directly
with `n-max`: −7% at 1, −29% at 2.

> **HIGH ACCEPTANCE DOES NOT MEAN A WIN.** Always compare end-to-end tokens/second against
> baseline. Never conclude from the acceptance rate.

**Recommendation: leave speculative decoding off on this machine.** `ngram-simple` is free but
found nothing to copy when writing new code — retest it on *editing* tasks, where output
copies input.

---

## 📉 Generation roughly HALVES at real depth

Every "tokens per second" number you read online is a shallow-context number.

| | Prefill | Generation |
| --- | --- | --- |
| 8192-token prompt (synthetic) | 2650 t/s | 105 t/s |
| **107,743-token prompt (real HTTP)** | **1758 t/s** | **46.6 t/s** |

**Always finish a tuning campaign with a real, deep request over HTTP.**

---

## 📏 Near the memory ceiling, a tiny context cut buys a lot of speed

`q8_0`, `-ub 512`:

| `-c` | Prefill |
| --- | --- |
| **126,976** | **2323 t/s** |
| 129,024 | 2101 t/s |
| 130,048 | 1850 t/s |

**Giving up 2.4% of the window bought 26% more prefill** — and kept the higher-precision KV
cache. Right-sizing `-c` returns memory *and* speeds prefill up.

> Note a correction: we first read this as a smooth "context tax" from the attention mask.
> It is mostly an artefact of *running out of memory* at the top end. Once a config fits
> comfortably, 130k prefills as fast as 8k (2650 vs 2651 t/s). **Large contexts are not
> inherently slow — they are hard to fit, and fitting is what costs speed.**

---

## 🎯 Other findings worth stating plainly

**A smaller quant is NOT automatically faster.**
Q3_K_XL (15.69 GiB) generated at ~94 t/s; Q4_K_M (20.21 GiB) at **117 t/s**. Q3_K CUDA
kernels are slower than Q4_K. The bigger file was 24% faster.

**`-ub` does nothing for generation.** 128 → 1024 moved prefill 1368 → 2651 t/s and
generation 106 → 106 t/s. Generation has no batch. Never spend VRAM on `-ub` for chat.

**`-fa off` costs 9 GiB.** Compute buffers go from 1.6 GiB to 10.4 GiB. Always `-fa on`.

**`-np` silently divides your context.** `-c 130048 -np 4` gives each request 32,512 tokens.
A 50k prompt is then unusable *and* uncacheable. For one user, `-np 1`.

**The optimal `-ts` is a memory boundary, not a compute one.** At 130k only `-ts 12,29`
loads; `11,30`, `13,28` and `14,27` all fail. `11,30` fails despite putting *less* on the
small card — it moves that layer onto the display GPU. **Budget from free VRAM, not
capacity**, and sweep *downward* from the proportional estimate (arithmetic said 15, truth
was 12).

**Your desktop is a tuning parameter.** 2.0 GiB of the 24 was held by Slack, Signal,
WhatsApp, Camtasia, Snagit and the NVIDIA overlay — almost all on the display GPU. Closing
them is the only optimisation that costs nothing.

**Benchmark hygiene that changed conclusions:**
- `llama-bench` has **no `-c` flag**, so it never allocates your real KV cache. Use
  `llama-batched-bench` for anything memory-related.
- **One fresh process per data point.** The same config measured third in a sweep read 1703
  t/s versus 2883 t/s measured alone — VRAM fragmentation.
- **Grep every log**, not just the results table, for `retrying`, `out of memory`, `failed
  to`, `ignoring`. That is how the best configuration in this wiki was discovered.
- Differences under **~8% are noise** when a GPU also drives a display.

---

## ✅ The configurations that came out of all this

```powershell
# QUALITY-FIRST -- q8_0 KV, full 130k.  1850 t/s synth / 1414 real, TG 98/47.5
-c 130048 -np 1 -ngl 999 -sm layer -ts 12,29 -fa on -b 2048 -ub 512 `
  -ctk q8_0 -ctv q8_0 -fit off -cram 24576

# SPEED-FIRST -- q4_0 KV, full 130k.   2650 t/s synth / 1758 real, TG 105/46.6
-c 130048 -np 1 -ngl 999 -sm layer -ts 12,29 -fa on -b 2048 -ub 1024 `
  -ctk q4_0 -ctv q4_0 -fit off -cram 24576

# BEST ALL-ROUND -- q8_0 KV at 127k.   2323 t/s, TG 109
-c 126976 -np 1 -ngl 999 -sm layer -ts 12,29 -fa on -b 2048 -ub 512 `
  -ctk q8_0 -ctv q8_0 -fit off -cram 24576
```

Full commands and the reasoning: [Chapter 7](07-recommended-configs.md).

---

## 🔨 The one thing still worth doing

**Rebuild llama.cpp:**

```powershell
.\.claude\skills\building-llamacpp-cuda\scripts\build-llamacpp.ps1 `
  -CMakeExtra '-DGGML_SCHED_MAX_COPIES=1','-DGGML_CUDA_FA_ALL_QUANTS=ON'
```

> ⚠️ **Do not run the plain `cmake -B build -DGGML_CUDA=ON ...` from llama.cpp's docs on this
> machine — it fails twice before building anything.** You need the **Ninja** generator inside
> a `vcvars64.bat` session, plus `-DCMAKE_CUDA_ARCHITECTURES="86;120"` (the 5060 Ti is
> Blackwell **12.0**, not Ada 8.9 — a prior build here used `86;89` and silently JIT-compiled
> for the faster card). See chapter 7 §7.5 and the `building-llamacpp-cuda` skill.

- `GGML_SCHED_MAX_COPIES=1` makes the fast non-pipelined path **deterministic** instead of a
  side effect of running out of memory — and should let `q8_0` KV run at `-ub 1024` at full
  130k, i.e. quality-first speed *and* speed-first speed together.
- `GGML_CUDA_FA_ALL_QUANTS=ON` unlocks `-ctk q8_0 -ctv q4_0`: precise keys, compact values.

**Untested. This is the top open item.**

---

← Back to the [index](README.md) · Full data: [Chapter 6](06-results.md)

