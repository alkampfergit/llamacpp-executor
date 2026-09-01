# VRAM residency campaign — 2026-09-01

Answers: "raising `-c` from 64000 to 130000 halved nothing but cost ~25% of prefill — is the
KV cache spilling into system RAM?" Full analysis in [`wiki/19-vram-residency.md`](../../19-vram-residency.md).

## Binary

Stock root binaries — **never** rebuilt or replaced for this campaign.

```
version: 0.1.2-dev (build 10509, commit fe8156f78)
built with Clang 20.1.8 for Windows x86_64, CUDA 13.3
```

Source read for the causal claims was verified against that same commit
(`git -C llama.cpp show fe8156f78:src/llama-context.cpp`), not against the submodule's current
checkout, which sits on the unrelated build branch `95aab10a9`.

## Hardware state

RTX 3070 (8192 MiB, device 0) + RTX 5060 Ti (16311 MiB, device 1, drives the display).
Idle display-GPU usage drifted 725–810 MiB across the session; recorded per run in each JSON.

## Method

`.claude/skills/tuning-llamacpp-configs/scripts/check-vram-residency.ps1`

- One fresh `llama-server` process per configuration, launched from the repo root (its own
  `bin/` — these are the root baseline binaries).
- Real HTTP requests to `/v1/chat/completions`, ~4,920-token prompt, **unique GUID prefix per
  repetition** so no request can hit the prompt cache. `cache_prompt: false` as well.
- 6 repetitions per configuration. **Rep 1 is cold** (allocator warm-up, clock ramp) and is
  excluded from the reported mean — it ran 10–13% below the warm reps in every configuration.
- Peak sampling every ~600 ms of `\GPU Process Memory(pid_*)\{Dedicated,Non Local,Shared} Usage`
  plus `nvidia-smi` per device.
- Expected host budget taken from `llama-fit-params -fitp on` (`Host` row) for the same flags.

Shared flags: `-ngl 999 -sm layer -ts 12,29 -fa on -b 2048 -ub 512 -np 1 -fit off --no-mmap`

## Files

| File | Configuration |
| --- | --- |
| `residency-rep6-c64000-q8_0.*` | `-c 64000`, `q8_0/q8_0` — 6 reps |
| `residency-rep6-c120000-q8_0.*` | `-c 120000`, `q8_0/q8_0` — 6 reps |
| `residency-rep6-c130000-q8_0.*` | `-c 130000`, `q8_0/q8_0` — 6 reps |
| `residency-rep6-c130000-q4_0.*` | `-c 130000`, `q4_0/q4_0` — 6 reps |
| `residency-verify-c130000-q8_0.*` | 3-rep reproduction after the verdict logic was corrected |
| `residency-c*.{json,log}` (no `rep6`) | Exploratory 2-rep passes, including `-c 126976` |

`.log` is the server's stderr; `.json` carries the peak counters, verdict and per-rep timings.

## Result

| `-c` | KV | pipelined | prefill (reps 2–6) | generation | GPU1 free |
| ---: | --- | :-: | ---: | ---: | ---: |
| 64000 | `q8_0` | yes | 2521 | 90.1 | 814 |
| 120000 | `q8_0` | no (fallback) | **2550** | 90.6 | 528 |
| 130000 | `q8_0` | no (fallback) | 1969 | 89.6 | 450 |
| 130000 | `q4_0` | yes | 2187 | 88.4 | 492 |
| 130000, `-ts 13,28` | `q8_0` | — | hard OOM on GPU0 | — | — |

## The ballast experiments — the part that settles causality

Correlation across the table above cannot separate GPU0 headroom, GPU1 headroom and `-c`,
because all three move together. `scripts/ballast.cu` (compile with `nvcc -arch=sm_86` for the
3070, `sm_120` for the 5060 Ti) pins a chosen number of MiB on a chosen device and idles, so
free VRAM becomes an independent variable while every llama.cpp flag is held fixed at
`-c 120000`.

| ballast | device | GPU1 free | prefill |
| ---: | :-: | ---: | ---: |
| 0 | — | 528 | 2578 |
| 64 | **GPU0** | 528 | 2562 |
| 192 | **GPU0** | 528 | 2542 |
| 384 | **GPU0** | 528 | 2575 |
| 288 | **GPU1** | 469 | 1939 |
| 128 | **GPU1** | 415 | 1896 |

**GPU0 headroom is not the variable** (flat to 374 MiB free). **GPU1 — the display GPU — is**,
with a threshold near **500 MiB free**. 128 MiB pinned there reproduces the entire −26% at a
context that was fast moments before.

Adapter-wide sampling on the matched pair shows the mechanism:

| arm | GPU1 free | llama-server non-local | adapter-wide shared (all processes) | prefill |
| --- | ---: | ---: | ---: | ---: |
| control | 607 | 482 | 489 | 2559 |
| +128 MiB GPU1 ballast | 434 | 546 | 622 | 1917 |

Below the threshold WDDM demotes ~133 MiB off the display adapter — 64 MiB of llama-server's
own, ~69 MiB belonging to other processes. **The KV cache never spills**; this small demotion
does, and it is the cause.

## Caveats

- Measured over HTTP with a ~4,920-token prompt. **Not comparable digit-for-digit** with the
  `llama-batched-bench` / 8192-token synthetic rows in `../results.tsv`; these numbers are not
  appended there for that reason.
- The two-rep exploratory files are kept for the record but are **too noisy to conclude from**:
  within-configuration spread reached 24% before the cold rep was separated out. They produced
  a wrong hypothesis (the pipeline fallback) that the 6-rep `-c 120000` control refuted.
- **Not established:** which tensors WDDM demotes, and how the 26% divides between PCIe traffic
  and residency stalls. The order of magnitude works out for a 64 MiB region re-read per layer
  per ubatch, but that is arithmetic, not a measurement.
- The display GPU's idle usage drifted **646 → 1065 MiB** across this session. A configuration
  near the threshold will flip bands depending on what is open on the desktop — which is why
  `-c 126976` was recorded at 2323 t/s in chapter 6 and re-measured at ~2112 here.
