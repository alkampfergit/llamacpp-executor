# Chapter 17 — You cannot turn off pipeline parallelism with `-ngl`

The fastest measured stock configuration for the MoE on this box includes a **recovered
allocation failure**. llama.cpp tries to reserve pipelined compute buffers, cannot fit them, logs

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 444.13 MiB on device 0: cudaMalloc failed: out of memory
graph_reserve: failed to allocate compute buffers
sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
```

…and then completes at **2371 t/s prefill**. This chapter originally compared that with a
roughly 1850 t/s run and credited the retry. [Chapter 18](18-fallback-causality.md) now controls
the variables: the `13,28` split is the large effect (+12.8%), while runtime fallback and a
one-copy build are only 1.6% apart at identical `c64000 / ub512 / ts13,28` settings.

This chapter tests a way to do it without rebuilding. **The way does not work, the wiki was
right, and the reason is worth knowing** — because anyone who reads the relevant source will
reach the same wrong conclusion I did.

---

## 17.1 The source says the constant is not the lever

Re-verified on the current tree. `GGML_SCHED_MAX_COPIES` is still compile-time only:

```c
// ggml/src/ggml-backend.cpp:760
#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif
```

No `getenv` anywhere in `ggml/`, `src/` or `common/`; no CLI flag in `common/arg.cpp`. So chapter
6's warning holds: **`$env:GGML_SCHED_MAX_COPIES = "1"` does nothing.**

But the constant is not what decides. One line later:

```c
// ggml/src/ggml-backend.cpp:1816
sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;
```

A `parallel` boolean gates it, and that boolean comes from a **context parameter** —
`cparams.pipeline_parallel` (`src/llama-cparams.h:55`), decided at `src/llama-context.cpp:428`
by five conditions that must *all* hold:

```cpp
bool pipeline_parallel =
    model.n_devices() > 1 &&
    model.n_gpu_layers() > model.hparams.n_layer_all &&   // <-- reachable from -ngl
    model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
    cparams.offload_kqv &&
    !model.has_tensor_overrides();
```

Four of those are CLI-reachable but ruinous: one GPU (`-dev`) throws away half the VRAM,
`-sm row` changes the split model entirely, `--no-kv-offload` is catastrophic, and `-ot` is the
CPU-offload trap of §6.7. The second looked free. `n_layer_all = 41` for
`Qwopus3.6-35B-A3B-Coder-MTP` (40 layers plus the MTP head at `blk.40`), so:

- `-ngl 999` → `999 > 41` → pipelining **on**, then it fails to fit and falls back
- `-ngl 41` → `41 > 41` is false → pipelining **never enabled**

That reasoning is correct as far as it goes, and it is why this chapter exists.

---

## 17.2 The measurement

Six runs, the user's own working server configuration verbatim, arms interleaved:
`-c 64000 -ctk q8_0 -ctv q8_0 -sm layer -ts 13,28 -fa on -b 2048 -ub 512 -fit off`, stock
binaries, `llama-batched-bench` via `bench-harness.ps1` (no sampling, so none of chapter 15's
acceptance noise applies).

| arm | prefill (3 reps) | mean | generation | mean | peak VRAM |
| --- | --- | ---: | --- | ---: | ---: |
| `-ngl 999` | 2418 · 2463 · 2231 | **2371** | 97.2 · 102.3 · 96.1 | **98.5** | 23306 |
| `-ngl 41` | 1070 · 1113 · 1073 | **1085** | 86.0 · 86.6 · 84.5 | 85.7 | **22855** |

`-ngl 41` is **54% slower on prefill** and 13% slower on generation. Within-arm spread is 4%
(`ngl41`) and 10% (`ngl999`), so the ranges are nowhere near overlapping.

**The mechanism check passed exactly as predicted**, which is what makes the result interesting
rather than merely disappointing. Every log was grepped for three signatures:

| arm | `pipeline parallelism enabled` | `retrying without pipeline parallelism` | `cudaMalloc failed` |
| --- | --- | --- | --- |
| `-ngl 999` (×3) | — | **yes** | **yes** |
| `-ngl 41` (×3) | — | — | — |

So `-ngl 41` really did prevent pipelining from ever being attempted, and it really did save the
buffers: **451 MiB less peak VRAM**, against the 444 MiB the failed allocation asks for. The
source reading was right. The conclusion drawn from it was not.

---

## 17.3 Why it is slower: there are 42 slots, not 41

`llama-fit-params -v` prints each layer's assignment. Counting them:

| `-ngl` | CPU | CUDA0 | CUDA1 | offloaded |
| ---: | ---: | ---: | ---: | ---: |
| 999 | **0** | 14 | 28 | 42 / 42 |
| **41** | **1** | 13 | 28 | **41 / 42** |
| 42 | **0** | 14 | 28 | 42 / 42 |

The model has **42 assignable slots** — 41 blocks plus the output layer — so `-ngl 41` strands
exactly one on the CPU:

```
load_tensors: layer   0 assigned to device CPU, is_swa = 0     <- -ngl 41
load_tensors: layer   0 assigned to device CUDA0, is_swa = 0   <- -ngl 999
```

One CPU layer is the entire 54%. Every token of an 8192-token prefill batch crosses to host
memory and back for layer 0. That is the same effect §6.7 measures for `-ncmoe`, where **two**
CPU layers cost 69% of prefill; one costing 54% is entirely in keeping.

### And this is not a tuning problem, it is the design

`-ngl 42` restores full offload — and `42 > 41` is true, so pipelining switches back on. **The
gate `n_gpu_layers > n_layer_all` *is* the "model is entirely on GPU" test.** llama.cpp only
pipelines when nothing is left on the CPU, which is obviously correct: pipelining across devices
is pointless if a host round-trip sits in the middle.

> **There is therefore no `-ngl` value that gives full offload and no pipelining.** The two
> conditions are the same condition. `-ngl` is not a lever on this at all, and the rebuild
> really is the only deterministic route.

---

## 17.4 A retraction

While chasing this I told the user *"the wiki is wrong on this point, in three places"* —
naming `known-traps.md` §4, chapter 6 §6.2 and chapter 7 §7.5 for saying a rebuild is the only
deterministic fix, and I handed over an `-ngl 41` command line **before measuring it**.

**Those three passages are correct. The `-ngl 41` advice was wrong and is withdrawn.** What I had
was a true reading of the source (`cparams.pipeline_parallel` is indeed the real switch, and
`-ngl` does indeed reach it) attached to an untested assumption that reaching it was free. It
costs a CPU layer, which is far worse than the accidental fallback it was meant to replace.

The pattern is the same one chapter 15 records, one rung further down: there, a plausible
mechanism explained an effect that turned out not to exist. Here, a *correct* mechanism was used
to justify a configuration nobody had run. **Reading the source tells you what a flag does; it
does not tell you what a flag costs.**

---

## 17.5 What to actually run

Keep `-ngl 999`. This remains the best-evidenced stock configuration for this model:

```
-c 64000 -np 1 -ctk q8_0 -ctv q8_0 -ngl 999 -sm layer -ts 13,28 -fa on -b 2048 -ub 512 -fit off
```

**2371 t/s prefill · 98.5 t/s generation**, three reps, stock binaries. Independently
corroborated by a real request through `llama-server` on the same config: **2314.79 t/s over
28,107 prompt tokens**, 75.06 t/s generation.

Two properties worth noting:

- **`-ts 13,28` is doing the performance work, not the retry.** 13:28 puts ~31.7% of the model
  on the 3070 against 12:29's 29.3%. A five-repetition same-build A/B in chapter 18 measures
  +12.8% prefill from that split alone. It also makes the 444 MiB reservation fail on stock.
- **The trigger is on device 0, and that is unusually good.** Configurations that exploit this
  fallback normally depend on the *display* GPU staying tight, which breaks the moment a browser
  opens. Here the failing allocation is on the 3070, which no desktop app ever touches — so it
  reproduces regardless of what is on screen.

The cost is living at the ceiling: 23306 MiB peak of ~23.2 GiB free. `-ngl 41` buys back 451 MiB
of that, and it is not worth 54% of prefill.

---

## 17.6 Scorecard

| | |
| --- | --- |
| **Refuted** | That `-ngl <n_layer_all>` disables pipeline parallelism usefully. It does disable it — at −54% prefill, because it strands one layer on the CPU |
| **Corroborated** | `GGML_SCHED_MAX_COPIES` is compile-time only (no `getenv`, no CLI flag, re-verified); `-ngl` cannot disable it without leaving a layer on CPU |
| **New** | `cparams.pipeline_parallel` (`llama-context.cpp:428`) is the real switch, and its `n_gpu_layers > n_layer_all` term is the same thing as "fully offloaded" — so the two cannot be separated. This model has **42** assignable slots, not 41 |
| **Best measured** | Qwopus 35B-A3B at `-c 64000`, `q8_0` KV, `-ts 13,28`, `-ngl 999`: **2371 / 98.5**, 3 reps |
| **Resolved in ch.18** | Matched rebuilds show max-copies 1 versus 4 differs by +1.8% at `ub512`; runtime fallback versus max-copies 1 differs by 1.6%. Neither explains the large gain. The `13,28` split does: +12.8% with scheduler mode fixed |

> **The closing point.** This is the third time in this wiki that a mechanism has been read
> correctly out of the source and then over-trusted. Chapter 14 caught a flag that was never
> read; chapter 15 caught an instrument noisier than its effect; this one caught a true premise
> with an initially unmeasured cost. The habit that keeps working is dull and cheap: **measure the thing
> you are about to recommend, before recommending it.** Six runs and twelve minutes were enough
> to overturn a command line I had already handed over.

---

Evidence: `wiki/benchmarks/pipeline-parallel-ngl/` — 6 rows, 6 logs, plus a `PROVENANCE.md`.

Correction and controlled causal A/B: [chapter 18](18-fallback-causality.md).

Previous: [Chapter 16 — Best command lines](16-best-commandlines.md) ·
Next: [Chapter 18 — The retry worked; the split made it fast](18-fallback-causality.md) ·
Back to [README](README.md).
