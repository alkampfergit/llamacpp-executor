# Known traps: the five silent failures

Each of these produces **no error message**, plausible output, and a large performance
loss. They are ordered by how much damage they do.

## Contents

1. [Sysmem fallback — the driver hides overflow](#1-sysmem-fallback)
2. [KV cache type pairs — attention leaves the GPU](#2-kv-cache-type-pairs)
3. [CPU expert offload — cheap for generation, brutal for prefill](#3-cpu-expert-offload)
4. [Pipeline parallelism — a loss on mismatched GPUs](#4-pipeline-parallelism)
5. [The `-np` context divisor](#5-the--np-context-divisor)
6. [Speculative decoding on a sparse MoE](#6-speculative-decoding-on-a-sparse-moe)
7. [Lesser traps](#7-lesser-traps)

---

## 1. Sysmem fallback

**Symptom:** throughput 3–10× below expectation. GPU utilisation low, CPU busy,
`memory.used` pinned at the card maximum, Task Manager shows non-zero **Shared GPU
Memory**.

**Cause:** since NVIDIA driver 536.40 (June 2023), a CUDA allocation exceeding VRAM on
Windows does not fail. The driver places the overflow in system RAM and lets the program
run across the PCIe bus.

**Fix:** NVIDIA Control Panel → Manage 3D Settings → **CUDA - Sysmem Fallback Policy** →
**Prefer No Sysmem Fallback**. Set globally, or per-program against the executable that
actually creates the CUDA context (not a `.bat`/`.ps1` wrapper). The policy is read at
context creation, so restart any running server.

**Why it matters most:** it makes every other measurement untrustworthy, because
"overflowing" and "badly configured" look identical. Fix this before tuning anything.

Not an issue on Linux — CUDA has always OOM'd properly there.

### How to actually verify residency — and the calibration that stops a false positive

`nvidia-smi` reports **dedicated** VRAM only and cannot see a spill. The WDDM counters can,
per process and per adapter:

```powershell
(Get-Counter '\GPU Process Memory(*)\Non Local Usage').CounterSamples |
  Where-Object { $_.InstanceName -like "pid_${serverPid}_*" }
```

`scripts/check-vram-residency.ps1` automates the whole procedure (launch, sample at peak
during a real HTTP prefill, grep the log, verdict).

> ### ⚠️ Non-zero non-local memory is NOT a spill
> llama.cpp always keeps several hundred MiB of host-visible buffers — CPU-side weights and
> pinned staging — and WDDM correctly counts those as non-local. Measured on this box, healthy
> runs read **480–940 MiB**, and the **fastest configuration had the highest figure (938 MiB)**
> while the slowest had 556 MiB. A detector that flags "non-local > 0" calls every run a spill.
>
> Compare against llama.cpp's own intent instead: `llama-fit-params ... -fitp on` prints a
> `Host` row (`Host 272 0 517` → 789 MiB). A **gross** overflow shows as non-local far above
> that budget with dedicated usage pinned at the adapter maximum.
>
> **But no absolute threshold is sufficient, however well calibrated.** The spill that actually
> cost 26% of prefill on this box was **+64 MiB on a 482 MiB baseline — still 200 MiB *below*
> the predicted budget.** Detect a spill as a **delta between two runs differing in one
> variable**, and watch the **adapter-wide** counter
> (`\GPU Adapter Memory(*)\Shared Usage`, all processes) as well as the per-process one: most
> of that demotion was other processes' surfaces, invisible in llama-server's own numbers.
> See `wiki/19-vram-residency.md`.

### The real trap on a multi-GPU box: it is the DISPLAY GPU's headroom

On Qwopus, `-c 130000` costs 26% of prefill against `-c 120000`. The cause is **not** the KV
cache, the pipeline fallback, or context length — all three were measured and refuted. It is
**free VRAM on the GPU that drives the display**, with a threshold near **500 MiB**. Below it,
WDDM demotes ~133 MiB off that adapter (64 MiB of llama-server's own, the rest other processes')
and prefill drops a quarter, silently.

Proved by intervention, not correlation: at a *fixed* `-c 120000`, pinning **128 MiB** of ballast
on the display GPU reproduces the whole effect, while **384 MiB** on the other GPU costs nothing
(`scripts/ballast.cu`). So:

- **Closing GPU-using desktop apps is worth up to 26% of prefill here** — a measured tuning
  action, not housekeeping.
- Budget `-c` to leave the display GPU ≥ 500 MiB free, and re-check when the desktop changes:
  idle usage on the display card drifted 646 → 1065 MiB in one session, enough on its own to
  flip a borderline config between the fast and slow bands.
- A `-ts` change that relieves the display GPU may simply move the failure: `-ts 13,28` at
  `-c 130000` hard-OOMs the other card.

See `wiki/19-vram-residency.md`.

---

## 2. KV cache type pairs

**Symptom:** you quantise the KV cache to save VRAM and throughput collapses by ~14×.
Output is still correct. Host RAM usage jumps by gigabytes.

**Cause:** a stock CUDA build compiles flash-attention kernels for only four **symmetric**
K/V pairs. From `ggml/src/ggml-cuda/CMakeLists.txt`:

```
fattn-vec-instance-f16-f16.cu
fattn-vec-instance-bf16-bf16.cu
fattn-vec-instance-q8_0-q8_0.cu
fattn-vec-instance-q4_0-q4_0.cu
```

and `ggml/src/ggml-cuda/fattn.cu` enforces it:

```c
if (K->type != V->type) {
    return BEST_FATTN_KERNEL_NONE;   // when GGML_CUDA_FA_ALL_QUANTS is undefined
}
```

Supported types without that macro: **F32, F16, Q4_0, Q8_0, BF16**.

So there are two independent ways to fall off the fast path:

| Config | Fault |
| --- | --- |
| `-ctk q8_0 -ctv q4_0` | **Asymmetric** — `K->type != V->type` |
| `-ctk q5_1 -ctv q5_1` | **Symmetric, but `q5_1` is not a supported type** |

**Symmetry is necessary but not sufficient.** The pair must be one of the four compiled
instantiations. When none qualifies, attention runs on another backend — which is why the
host memory figure explodes.

**Rule:** `f16/f16`, `bf16/bf16`, `q8_0/q8_0`, or `q4_0/q4_0`. Never mix; never use a type
outside that list.

**Diagnosis:** the `Host` column reveals it before you waste a benchmark run.

```powershell
./llama-fit-params.exe -m <model> -c 130048 -fa on -ctk q8_0 -ctv q4_0 -fitp on
```

A `Host` figure in the thousands of MiB rather than a few hundred is the signature.

**Escape hatch:** to use asymmetric KV (attractive, since keys tolerate quantisation worse
than values) you must compile it yourself — `GGML_CUDA_FA_ALL_QUANTS` is **OFF** by
default:

```
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
```

Cost: long compile, much larger CUDA binary.

---

## 3. CPU expert offload

**Symptom:** `-ncmoe 2` on a 40-layer MoE — 5% of the model — halves prefill.

**Cause:** the prefill/generation asymmetry. Generating one token activates 8 of 256
experts (~3% of a layer's expert weights), so the CPU's share is cheap. Prefilling a
512-token batch activates *nearly every expert*, so the CPU must stream the entire expert
matrix through its memory bus while both GPUs idle waiting.

**Consequence:** prefill degrades roughly twice as fast as generation as you offload more.

**Rule:** treat any non-zero `-ncmoe` as evidence that something else should shrink first.
Shrinking `-ub` from 512 to 256 is far better than offloading two layers.

If unavoidable, note llama.cpp's own hint:

```
tensor overrides to CPU are used with mmap enabled
  -- consider using --load-mode none for better performance
```

---

## 4. Pipeline parallelism

**Symptom:** a configuration that *fails* to fit its pipelined buffers, falls back
automatically, and runs **faster** than a successfully pipelined one.

```
graph_reserve: failed to allocate compute buffers
sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
```

**Cause:** with multiple devices llama.cpp can keep several copies (default 4) of intermediate
activations. Those copies are often exactly what breaks a tight fit. The retry uses one copy.

The historical 1850-versus-2650 comparison changed KV type and ubatch as well as execution
mode. Controlled Qwopus tests later found runtime fallback and a one-copy build only **1.6%**
apart at identical `c64000 / ub512 / ts13,28` settings. Changing only the tensor split from
`12,29` to `13,28` gave **+12.8% prefill**. See `wiki/18-fallback-causality.md`.

### The sub-trap: the knob is not a runtime setting

`GGML_SCHED_MAX_COPIES` is a **compile-time `#define`**, defaulting to 4:

```c
#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif
```

**Setting `$env:GGML_SCHED_MAX_COPIES = "1"` does nothing.** This has been believed and
written down as if it worked; the evidence that it does not is that a run with the variable
set still allocated 1407 MiB of pipelined buffers before falling back. Any speed change
attributed to it was variance.

**If the reservation memory is the problem:**

1. **Rebuild for one copy (deterministic):**
   ```
   cmake -B build -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1
   ```
2. Reduce `-ub`, context, or rebalance `-ts` to restore headroom. A naturally occurring retry
   followed by a result row is valid recovered data, but do not raise `-ub` merely to trigger it.

**Rule:** one copy is smaller; its speed must be measured. Optimise the tensor split separately
and record whether each run used fallback.

### ⛔ And do NOT try to reach it with `-ngl`. Measured: −54% prefill.

The constant is not the real switch. `sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1`,
and `parallel` comes from `cparams.pipeline_parallel`, decided at `src/llama-context.cpp:428` by
five conditions — one of which looks CLI-reachable for free:

```cpp
model.n_gpu_layers() > model.hparams.n_layer_all
```

So `-ngl <n_layer_all>` does switch pipelining off, verified by log: no
`pipeline parallelism enabled`, no fallback warning, and **451 MiB less peak VRAM**. It is still
wrong, because a model has **`n_layer_all` + 1 assignable slots** — the blocks plus the output
layer — so `-ngl n_layer_all` strands one layer on the **CPU**:

```
load_tensors: layer 0 assigned to device CPU     <- -ngl 41  (n_layer_all = 41)
load_tensors: layer 0 assigned to device CUDA0   <- -ngl 999
```

Measured on Qwopus3.6-35B-A3B, 3 interleaved reps each, identical config otherwise:

| `-ngl` | prefill | generation | CPU layers |
| ---: | ---: | ---: | ---: |
| 999 | **2371** | **98.5** | 0 |
| 41 | 1085 (**−54%**) | 85.7 | **1** |

One CPU layer costs 54% of prefill — the same family as the `-ncmoe` trap in §3, where two cost
69%.

**And `-ngl 42` cannot rescue it:** `42 > 41` is precisely the gate condition, so pipelining
switches back on. **The gate is the "fully offloaded" test**, by design — llama.cpp only
pipelines when nothing sits on the CPU. Full offload and no-pipelining are the same condition;
`-ngl` is not a lever here. Use the rebuild. See `wiki/17-pipeline-parallelism-ngl.md`.

---

## 5. The `-np` context divisor

**Symptom:** you launch with `-c 130048`, then a 50,000-token prompt is truncated or
rejected, and every request reprocesses its whole prompt.

**Cause:** with statically divided caches, `-np N` splits the context budget:

```
-c 130048 -np 4   ->   4 slots x 32512 tokens each
-c 130048 -np 1   ->   1 slot  x 130048 tokens
```

A prompt that does not fit a slot cannot be cached either, so you lose prompt caching —
which is worth more than every other optimisation combined.

**Rule:** choose `-np` by asking how many *simultaneous long* conversations are needed.
For one user, `-np 1`. Check the startup log for `n_slots` and `kv_unified` rather than
assuming; recent builds default to unified accounting, which is more forgiving.

---

## 6. Speculative decoding on a sparse MoE

**Symptom:** you enable `--spec-type draft-mtp`, the server reports **excellent** draft
acceptance, and end-to-end generation gets *slower*.

Measured on a 35B-A3B MoE (~3 B active), generating 300 tokens of fresh code:

| Config | TG t/s | vs baseline | Draft acceptance |
| --- | --- | --- | --- |
| baseline | **97–100** | — | — |
| `draft-mtp --spec-draft-n-max 1` | 90.1 | **−7%** | — |
| `draft-mtp --spec-draft-n-max 2` | 70.3 | **−29%** | **0.67–0.72, mean len 2.34** |
| `draft-mtp --spec-draft-n-max 3` | server OOM | — | — |
| `ngram-simple` n 2/4/8 | 96.6–97.6 | 0% (noise) | — |

**Cause.** The server log shows how drafts are produced:

```
common_speculative_init_result: creating MTP draft context against the target model
```

Each drafted token costs roughly a **full forward pass**. Speculative decoding is a trade of
*compute* for *latency*, and it needs the target model's per-token pass to be expensive
enough to amortise the drafting. A sparse MoE activating ~3 B of 35.5 B parameters has almost
nothing to amortise, so the drafting overhead is not hidden — it *is* the bill. The cost
scales directly with `n-max`: −7% at 1, −29% at 2.

This is why vendor figures for the same family show **1.73× on a dense 27 B** model but only
**1.17× on the 35B-A3B MoE**. On slower hardware that trend continues past 1.0× into a loss.

> **THE RULE: high draft acceptance does not mean a win.** 70% acceptance and 2.34 tokens per
> step still lost 29%. Always compare end-to-end tokens/second against a no-drafting baseline
> on the same prompt. Never conclude from the acceptance rate, and never from vendor claims.

**Expectations by model type:**

| Model | Prior expectation |
| --- | --- |
| Dense, large (27 B+) | Speculation likely helps — expensive pass to amortise |
| Sparse MoE, low active params | Likely neutral-to-harmful — measure before enabling |
| Any | `ngram-*` drafters cost 0 VRAM and ~0 speed, so they are a safe bet |

**A second trap inside this one:** n-gram drafters measured 0% here, but the test asked the
model to *write new code*. They work by replaying repeated token sequences from context, so
there was nothing to copy. That is a limitation of the test, not a verdict. Test n-gram
drafting on **edit-style** work (refactor this, add type hints to this file) where the output
largely reproduces the input.

---

## 7. Lesser traps

| Trap | Reality |
| --- | --- |
| `-fa off` or omitted | Compute buffers grow ~6×. Never a real option. |
| `-mg` with `-sm layer` | No effect. Delete it. |
| `-ub 1024` / `2048` | Double the memory for no prefill gain past 512. |
| `-c 262144` "for headroom" | Consumes KV memory and can force slower fit compromises; the old universal 31% empty-context tax was refuted. |
| "Smaller quant is faster" (or slower) | **Measure it.** The old "Q3_K kernels are slower than Q4_K" line compared two different fine-tunes. Same-base-model quants generate within **1.0%**; what quant moves is **prefill (76% spread) and VRAM (3.8 GiB)**, and the smallest file won both. `references/best-commandlines.md` §3. |
| Tuning with `llama-bench` | No `-c` flag, so it never allocates the real KV cache. |
| Multi-value sweep in one process | VRAM fragmentation; later runs read slower. |
| `--cache-prompt`, `-kvo` | Accepted but no-ops in current builds; the behaviour is now default. |
| `failed to fit params ... already set by user` | Not an error. The fitter stood down because you set `-ngl`. |
| `cublasCreate ... resource allocation failed` | Out-of-memory under a different name. |
| `unused tensor blk.N ... ignoring` | Expected on MTP/draft models; those weights cost nothing until `--spec-type` activates them. |
