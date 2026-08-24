# `a3b-asym-kv/` — asymmetric KV on the MoE

Evidence for the asymmetric-KV section of
[`references/best-commandlines.md`](../../../.claude/skills/tuning-llamacpp-configs/references/best-commandlines.md) §2.

- Model: `Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf` (20.21 GiB).
- Binaries: **`llama.cpp\build-fa\bin`** — `GGML_CUDA_FA_ALL_QUANTS=ON`,
  `GGML_SCHED_MAX_COPIES=1`, base `c80c4c8a8`, MSVC 14.44, CUDA 13.3.73.
  **Not the stock binaries.** On stock, an asymmetric K/V pair hits
  `if (K->type != V->type) return BEST_FATTN_KERNEL_NONE;` in
  `ggml/src/ggml-cuda/fattn.cu`, attention leaves the GPU, and throughput collapses to ~19–25
  t/s — which is what the `q5_1` and `q4_0K` rows in the main `results.tsv` actually are.
- Shared config: `-ngl 999 -sm layer -ts 12,29 -fa on -b 2048 -ub 512 -fit off`,
  `npp 8192 / ntg 128 / npl 1`.
- Sysmem-fallback precondition verified for **this** exe before any measurement:
  `-ts 100,0 -c 130048` → `cudaMalloc failed: out of memory`. The driver refuses rather than
  spilling silently.

Logs are complete — 22 rows, 22 files in `logs/`, one per row.

## The three suites

| `suite` | what it answers |
| --- | --- |
| `a3b-asym-kv` | four KV pairs at c126976, 2 interleaved reps each |
| `a3b-ctx-ladder` | how high `q8_0`/`q4_0` climbs |
| `a3b-ladder-control` | **where `q8_0`/`q8_0` actually stops**, plus replication of the ladder |

That third suite exists because the first two did not license the conclusion. "q8q4 reaches
c180224" means nothing without the context where the *symmetric* pair fails — and nobody had
ever run this model above c130048, so the symmetric ceiling was assumed, not measured. It turned
out to be **163840**, not 130048, which cut the claimed gain from ~50k tokens to +16384.

## Results

**Ceilings** (load / no-load, so noise-immune):

| KV | loads at | OOM at |
| --- | ---: | ---: |
| q8_0 / q8_0 | 163840 | 180224 |
| q8_0 / q4_0 | **180224** | 196608 |

**At c126976**, 2–3 reps each:

| K / V | prefill mean | generation mean | peak VRAM | vs q8q8 |
| --- | ---: | ---: | ---: | ---: |
| q8_0 / q8_0 | 2007 | 101.4 | 23312 | — |
| q8_0 / q4_0 | 2302 | 103.0 | 23128 | **−184 MiB** |
| q4_0 / q4_0 | 2302 | 104.1 | 22888 | −424 MiB |
| q4_0 / q8_0 | 2244 | 105.5 | 23192 | −120 MiB |

## ⚠️ Two reasons these numbers do not become a recommendation

1. **`build-fa` is slower than stock at an identical config.** Stock q8_0/q8_0 at c126976 was
   2323 t/s prefill / 109.1 t/s generation; `build-fa` q8_0/q8_0 is 2007 / 101.4. `build-fa`
   also carries `GGML_SCHED_MAX_COPIES=1`. Chapter 18 later tested matched rebuilds on this
   Qwopus model: at `ub512`, one copy versus four changed prefill by only +1.8%. The old
   stock-versus-build deficit is therefore not evidence against the scheduler flag.
2. **`V=q4_0` has never been quality-gated on this model.** Chapter 11 argued keys tolerate
   quantisation worse than values, so `q8_0`/`q4_0` is the *right* asymmetric choice — but that
   is reasoning, not measurement.

Also: **c180224 loads but is not usable.** Generation swung 61.22 / 75.90 / 105.19 across three
reps with peak VRAM flat against the wall at ~23.6 GiB. The single 2177 t/s / 105.19 t/s first
sample briefly looked like "throughput improves with context"; three reps refuted it.
