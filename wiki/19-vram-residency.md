# Chapter 19 — Proving what is actually in VRAM

> **The question.** Raising `-c` from 64000 to 130000 on Qwopus, with every other flag
> identical, dropped real prefill from ~2700 to ~2000 t/s through a GitHub Copilot endpoint.
> The natural explanation: *the KV cache no longer fits, part of it is in system RAM, and PCIe
> is the tax.*
>
> **The answer, established by controlled intervention rather than argument:** the KV cache is
> fully resident and never spills. But something *does* get pushed to system RAM — about
> 130 MiB, on the **display GPU**, and it is not the cache. The controlling variable is
> **free VRAM on GPU1, with a threshold near 500 MiB**. `-c` only matters because it is one of
> several things that eats into that headroom.

## Contents

- [19.1 Why nvidia-smi cannot answer this](#191-why-nvidia-smi-cannot-answer-this)
- [19.2 The counters that can](#192-the-counters-that-can)
- [19.3 The calibration: an absolute non-local figure proves nothing](#193-the-calibration-an-absolute-non-local-figure-proves-nothing)
- [19.4 The observations](#194-the-observations)
- [19.5 What the source says sizes the reservation](#195-what-the-source-says-sizes-the-reservation)
- [19.6 Three explanations, all refuted](#196-three-explanations-all-refuted)
- [19.7 The ballast experiment: intervening instead of correlating](#197-the-ballast-experiment-intervening-instead-of-correlating)
- [19.8 The mechanism](#198-the-mechanism)
- [19.9 What to actually do](#199-what-to-actually-do)

---

## 19.1 Why nvidia-smi cannot answer this

`nvidia-smi` reports **dedicated** VRAM only. Since driver 536.40 a CUDA allocation that
exceeds VRAM on Windows does not fail — the driver satisfies it out of system RAM and the
program keeps running over PCIe. When that happens `memory.used` sits at the card maximum
and nothing else changes. A spilling process and a merely slow one look identical.

Worse, on a **display** GPU the spill need not be a llama.cpp allocation failing at all. WDDM
manages residency for every process on the adapter and will demote whatever it likes to make
room for the desktop. Nothing in llama.cpp's log mentions it.

## 19.2 The counters that can

Windows' WDDM performance counters expose GPU memory two ways. Both are needed.

**Per process, per adapter** — what llama-server itself holds:

```
\GPU Process Memory(pid_<pid>_luid_<adapter>)\Dedicated Usage   <- genuinely on the card
\GPU Process Memory(pid_<pid>_luid_<adapter>)\Non Local Usage   <- in system RAM
```

**Per adapter, all processes** — what the whole card is pushing out, including the desktop
compositor's surfaces, which never appear in llama-server's own numbers but still cost it GPU
time:

```
\GPU Adapter Memory(luid_<adapter>)\Shared Usage
```

`.claude/skills/tuning-llamacpp-configs/scripts/check-vram-residency.ps1` samples both at peak
while a real prompt prefills over HTTP, prints `llama-fit-params`' predicted host budget beside
them, greps the log for allocation warnings, and separates the cold repetition from the warm
mean.

```powershell
& .\.claude\skills\tuning-llamacpp-configs\scripts\check-vram-residency.ps1 `
    -Model <model.gguf> -Ctx 130000 -Ts '12,29' -Reps 6 `
    -OutDir S:\OsDevelop\llamacpp\wiki\benchmarks\vram-residency
```

> Quote `-Ts '12,29'`. Unquoted, PowerShell parses `12,29` as an array and the script rejects it.

## 19.3 The calibration: an absolute non-local figure proves nothing

**The first version of that script was wrong twice, and both errors are instructive.**

**Error 1 — it flagged any non-local memory above 64 MiB as a spill,** and so reported
`SPILLING` for every configuration measured, including the fastest. llama.cpp always keeps
several hundred MiB of host-visible buffers — CPU-side weights and pinned staging — and WDDM
correctly counts those. Healthy runs here read **480–940 MiB**.

**Error 2 — the obvious fix was also wrong.** Comparing against `llama-fit-params`' `Host`
budget (`Host 272 0 517` → 789 MiB) looks rigorous, and it does catch a gross overflow. But
the spill that actually mattered here is **64 MiB on top of a 482 MiB baseline — still 200 MiB
*below* the predicted budget.** No absolute threshold, however well calibrated, can see it.

> **The rule that survives:** a spill is detected as a **delta between two runs that differ in
> one variable**, not as a number. Run a matched control. Watch the adapter-wide figure as well
> as the per-process one.

## 19.4 The observations

Stock root binaries, build `10509 / fe8156f78`. `llama-server` over HTTP, one fresh process per
configuration, a ~4,920-token prompt with a unique GUID prefix per repetition so nothing hits
the prompt cache. Six repetitions; **rep 1 is always cold** and is excluded from the mean.

Shared flags: `-ngl 999 -sm layer -ts 12,29 -fa on -b 2048 -ub 512 -np 1 -fit off --no-mmap`

| `-c` | KV | CUDA0 reserve | pipelined? | **prefill** | gen | GPU0 free | **GPU1 free** |
| ---: | --- | ---: | :-: | ---: | ---: | ---: | ---: |
| 64000 | `q8_0` | 444 MiB ✅ | yes | **2521** | 90.1 | 965 | 814 |
| 120000 | `q8_0` | 772.63 MiB ❌ | no | **2550** | 90.6 | 831 | 528 |
| 126976 | `q8_0` | 813.13 MiB ❌ | no | ~2112 | — | 789 | 440 |
| 130000 | `q8_0` | 831.13 MiB ❌ | no | **1969** | 89.6 | 771 | 450 |
| 130000 | `q4_0` | 831 MiB ✅ | yes | **2187** | 88.4 | 563 | 492 |
| 130000, `-ts 13,28` | `q8_0` | — | — | **hard OOM** | — | — | — |

Evidence: [`benchmarks/vram-residency/`](benchmarks/vram-residency/).

Read off immediately: **generation is flat** (88.4–90.6, a 2.4% spread). Only prefill moves.
And the last row matters — at `-c 130000` you cannot relieve GPU1 by moving a layer to GPU0,
because GPU0 then fails outright, even on the single-copy retry. At that context the two
constraints are in direct conflict.

## 19.5 What the source says sizes the reservation

`src/llama-context.cpp`, `llama_context::sched_reserve()`:

```cpp
const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
...
mctx = memory->init_full();
```

The worst-case prompt-processing graph is reserved at `-ub` width against the **entire
allocated KV window**. So the reserved compute buffer grows linearly with `-c` — this is where
`-c` costs VRAM a *second* time, beyond the cache itself, and it is the part people forget.

| `n_ctx_slot` | CUDA0 reservation |
| ---: | ---: |
| 64000 | 444 MiB (predicted) |
| 120064 | 772.63 MiB |
| 126976 | 813.13 MiB |
| 130048 | 831.13 MiB |

**6.0 KiB of CUDA0 compute buffer per 1024 tokens of `-c`** (0.00586 MiB/token). Extrapolating
the 120064 measurement back gives 772.63 − 0.00586 × 56064 = 444.1 MiB against
`llama-fit-params`' independent 444. Note `n_tokens = min(n_ctx, n_ubatch)`: the reservation is
sized by **`-ub`, not `-b`**. Lowering `-b` is not a lever.

Above ~`-c 118000` on this split that reservation no longer fits on the 3070, and the recovery
path in the same function runs — producing the alarming but benign log block:

```
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 831.13 MiB on device 0: cudaMalloc failed: out of memory
E graph_reserve: failed to allocate compute buffers
W sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
```

## 19.6 Three explanations, all refuted

| Explanation | Refuted by |
| --- | --- |
| **The KV cache spilling to system RAM** | The cache is 1.4 GiB at 130k and sits entirely in dedicated VRAM in every configuration. Per-process non-local memory never approached it. |
| **The pipeline-parallel fallback** | `-c 120000` triggers the identical `cudaMalloc failed` → `retrying without pipeline parallelism` sequence and is the **fastest** configuration measured. Consistent with [chapter 18](18-fallback-causality.md)'s controlled 1.6%. |
| **Long context is intrinsically slower to prefill** | `llama_kv_cache::get_n_kv()` returns `GGML_PAD(cells.used_max_p1(), n_pad)` capped at cache size — the *used* cells, not the allocation. For a 4,920-token prompt the attention graph is identical at any `-c`. Confirmed by flat generation and identical `graphs reused` counts (7/13/19/25/31/37) in every run. |

Also eliminated from source: nothing in the CUDA backend branches on free VRAM. The cuBLAS
workspace is a fixed 4 MiB (`common.cuh`, 32 MiB only on Hopper+), and `ggml_cuda_pool_vmm::alloc`
wraps `cuMemCreate` in `CU_CHECK`, which aborts rather than degrading to a slower path.

The second row is the one worth dwelling on: it was the working hypothesis after four
configurations, and the `-c 120000` control killed it. Two repetitions could not have seen
that — the rep-1-to-rep-2 spread inside a single configuration reached 24%, as large as the
effect being chased.

## 19.7 The ballast experiment: intervening instead of correlating

Correlation across configurations could not separate GPU0 headroom, GPU1 headroom and `-c`,
because all three move together. So hold `-c` and every flag **fixed** at the fast setting and
vary free VRAM directly, with a 20-line CUDA program that pins a chosen number of MiB on a
chosen device and idles.

**Ballast on GPU0**, `-c 120000` fixed:

| GPU0 ballast | GPU0 headroom | prefill |
| ---: | ---: | ---: |
| 0 | 831 | 2578 |
| 64 | 630 | 2562 |
| 192 | 502 | 2542 |
| 384 | 374 | **2575** |

**Flat.** Starving GPU0 to 374 MiB — far below the 771 MiB the slow configuration had — costs
1.4%, inside noise. **GPU0 headroom is not the variable**, and by extension neither is the
compute-buffer reservation that fails there.

**Ballast on GPU1**, the same `-c 120000` fixed:

| GPU1 ballast | GPU1 headroom | prefill | vs control |
| ---: | ---: | ---: | ---: |
| 0 | 528 | 2578 | — |
| 288 | 469 | 1939 | **−25%** |
| 128 | 415 | 1896 | **−26%** |

**128 MiB of ballast on the display GPU reproduces the entire effect**, at a context size that
was fast moments earlier. Every measurement in this chapter now sorts by one number:

| configuration | GPU1 free | prefill |
| --- | ---: | ---: |
| `c64000` | 814 | 2521 |
| `c120000` | 528 | 2578 |
| `c120000` + 384 MiB on **GPU0** | 528 | 2575 |
| `c130000` `q4_0` | 492 | 2187 |
| `c120000` + 288 MiB on **GPU1** | 469 | 1939 |
| `c130000` | 450 | 1969 |
| `c120000` + 128 MiB on **GPU1** | 415 | 1896 |

**The threshold is ~500 MiB of free VRAM on the display GPU.** Above it, ~2550 t/s. Below it,
~1900–1970. `-c` never appears in the causal chain except as one of the things consuming that
headroom.

## 19.8 The mechanism

Two arms differing only in 128 MiB of GPU1 ballast, with adapter-wide sampling added:

| arm | GPU1 free | llama-server non-local | **adapter-wide shared** | prefill |
| --- | ---: | ---: | ---: | ---: |
| control | 607 | 482 | **489** | 2559 |
| +128 MiB GPU1 ballast | 434 | 546 | **622** | 1917 |

Once free VRAM on the display adapter crosses the threshold, WDDM demotes about **133 MiB**
into system RAM: **64 MiB of it llama-server's own**, the remaining ~69 MiB belonging to other
processes on that adapter — the desktop compositor and friends. llama.cpp is never told. There
is no log line, no allocation failure, and `nvidia-smi` shows a card that merely looks full.

So there *is* a spill, and the original intuition was directionally right. It is simply not the
one that was guessed: not the KV cache, not sized by `-c`, and small enough in absolute terms
(546 MiB total non-local) to sit **200 MiB below llama.cpp's own legitimate host budget** —
which is exactly why every absolute-threshold detector misses it.

**Not established:** which tensors get demoted, and how the 26% divides between PCIe traffic and
WDDM residency stalls. A 64 MiB region re-read once per layer per ubatch is the right order of
magnitude for the observed ~600 ms, but that is arithmetic, not a measurement.

This also explains a standing inconsistency in this wiki. The display GPU's own idle usage
drifted **646 → 1065 MiB** across one session here, which moves GPU1 headroom by 400 MiB on its
own. A configuration sitting near the threshold therefore flips between the fast and slow bands
depending on what is open on the desktop — which is why chapter 6 recorded `-c 126976` at
2323 t/s and a re-measurement found ~2112 with the identical command line.

## 19.9 What to actually do

**Target ≥ 500 MiB free on the display GPU.** That, not the context number, is the thing to
tune. Three ways to buy it, cheapest first:

1. **Close GPU-using desktop apps.** Directly worth up to 26% of prefill here — no longer
   housekeeping advice, a measured tuning action.
2. **Use `-c 120000`.** 92% of the window, `q8_0` KV retained, **+30% prefill** over `-c 130000`
   (2550 vs 1969) with generation unchanged. It is the largest context that still leaves GPU1
   above the threshold while GPU0 can still load.
3. **Do not try to fix it with `-ts` at 130k.** `-ts 13,28` moves a layer off GPU1 as the theory
   wants, and then GPU0 fails outright — even the single-copy retry. At that context there is no
   split that satisfies both cards.

```
llama-server.exe -m <model.gguf> --alias qwopus-coder --host 0.0.0.0 --port 9010 `
  -c 120000 -np 1 -ctk q8_0 -ctv q8_0 -ngl 999 -sm layer -ts 12,29 `
  -fa on -b 2048 -ub 512 -fit off --no-mmap
```

If the full 130k window is genuinely required, `q4_0` KV buys back 11% (2187) by freeing ~470 MiB
of cache — and it has **passed** the 108k needle-retrieval gate on this model.

Four rules this chapter earned:

1. **`nvidia-smi` cannot see a spill.** Use the WDDM counters.
2. **No absolute non-local threshold works.** The spill that mattered was *below* llama.cpp's own
   host budget. Detect it as a **delta against a matched control**, and sample the **adapter-wide**
   figure, not just the per-process one.
3. **When variables move together, intervene.** Correlation across configurations could not
   separate GPU0 headroom, GPU1 headroom and `-c`. Pinning VRAM with a 20-line CUDA program
   settled it in two runs.
4. **Two repetitions cannot resolve a 20% effect here.** Cold-start spread reaches 24%. Six reps,
   cold rep excluded, or do not draw the conclusion.

Previous: [Chapter 18 — The retry worked; the split made it fast](18-fallback-causality.md) ·
Back to [README](README.md).
