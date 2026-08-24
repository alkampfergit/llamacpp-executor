# `pipeline-parallel-ngl/` — can `-ngl` disable pipeline parallelism?

Evidence for [chapter 17](../../17-pipeline-parallelism-ngl.md). **Answer: it can, and you must
not — it costs 54% of prefill.**

- Model: `Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf` (20.21 GiB, `n_layer` 40, `n_layer_all` 41,
  **42 assignable slots** counting the output layer).
- Binaries: **the stock baseline** in `S:\OsDevelop\llamacpp\` (`build 10509, commit fe8156f78`).
  Not `build-fa` — the point was to test a runtime flag, so the build had to stay fixed.
- Config, identical in both arms except `-ngl`:
  `-c 64000 -ctk q8_0 -ctv q8_0 -sm layer -ts 13,28 -fa on -b 2048 -ub 512 -fit off`,
  `npp 8192 / ntg 128 / npl 1`. This is the user's own working `llama-server` config verbatim.
- Arms interleaved (999, 41, 999, 41, 999, 41) so drift could not land on one arm.
- `llama-batched-bench`, so there is no sampling and none of chapter 15's draft-acceptance
  variance applies here.

Logs are complete: **6 rows, 6 files** in `logs/`, one per row, named by label.

## Results

| arm | prefill (r1·r2·r3) | mean | generation | mean | peak VRAM |
| --- | --- | ---: | --- | ---: | ---: |
| `-ngl 999` | 2418.16 · 2463.12 · 2230.97 | **2370.8** | 97.19 · 102.26 · 96.11 | **98.5** | 23306 |
| `-ngl 41` | 1069.85 · 1113.05 · 1073.35 | **1085.4** | 85.96 · 86.57 · 84.45 | 85.7 | **22855** |

`-ngl 41`: **−54.2% prefill, −13.1% generation, −451 MiB peak.** Within-arm spread 4% and 10%,
so the arms do not overlap.

## The qualitative half — grep every log, do not infer

Each log was checked for three strings. This is the part that makes the result interpretable:

| arm | `pipeline parallelism enabled` | `retrying without pipeline parallelism` | `cudaMalloc failed` |
| --- | --- | --- | --- |
| `ngl999_r1/r2/r3` | — | **yes** | **yes** |
| `ngl41_r1/r2/r3` | — | — | — |

`-ngl 41` genuinely never attempted pipelining, and genuinely saved the ~444 MiB reservation.
The mechanism worked. It is simply not worth having.

## Why it is slower (from `llama-fit-params -v`)

| `-ngl` | CPU | CUDA0 | CUDA1 |
| ---: | ---: | ---: | ---: |
| 999 | **0** | 14 | 28 |
| 41 | **1** | 13 | 28 |
| 42 | **0** | 14 | 28 |

`-ngl 41` strands `layer 0` on the CPU. One CPU layer costs 54% of prefill — consistent with
chapter 6 §6.7, where two CPU layers cost 69%.

**And `-ngl 42` cannot help:** `42 > n_layer_all(41)` is exactly the gate condition at
`src/llama-context.cpp:428`, so full offload and no-pipelining are the *same* condition. There is
no `-ngl` value that gives both.

## ⚠️ Do not read a cross-arm VRAM saving as a win

`-ngl 41`'s 22855 MiB peak looks attractive next to 23306. It is real, and it is bought with a
host round-trip on every token. If you need that 451 MiB, take it from `-c` or `-ub`, never from
`-ngl`.
