# Best command lines per model — the canonical table

**This file is the single source for "what do I actually run".** `wiki/16-best-commandlines.md`
discusses provenance and gaps but deliberately does **not** repeat these commands — see the
duplication scar in `CLAUDE.md`.

Hardware these are tuned for: **RTX 3070 (8 GiB, device 0) + RTX 5060 Ti (16 GiB, device 1,
drives the display)**. Nothing here transfers to another GPU pair.

`-ts` is a **proportion**, not a layer count — llama.cpp's own help reads *"fraction of the
model to offload to each GPU, comma-separated list of proportions, e.g. 3,1"*. So `-ts 12,29`
means 12:29 ≈ 29%/71%, and `-ts 1,2` means 33%/67%. It therefore *arithmetically* transfers to
any model, which is exactly the trap: the right proportion depends on that model's weight and
KV footprint, and proportions round to whole layers, so the same string lands on different
layer counts for models of different depth. **Re-derive it per model.**

## How to read the confidence column

| | |
| --- | --- |
| **A** | Repeated runs, logged, rows in a `results.tsv`. Trust the number |
| **B** | Single run, logged, row exists. Trust the config, treat the number as ±10% |
| **C** | Prose in a wiki chapter, no row, no log. Treat as a starting point and re-measure |

Every number below was produced by the **baseline binaries in `S:\OsDevelop\llamacpp\`**
unless the row says otherwise.

---

## 1. Qwen3.8-27B-Q4_K_M — dense, 15.66 GiB, MTP head at `blk.64`

```
S:\HuggingFace\lmstudio\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf
```

Shared flags: `-np 1 -ngl 999 -sm layer -ts 22,43 -fa on -b 2048 -ub 512 -fit off`
(`-ts 23,42` **fails** — the 3070 runs out.)

| # | Use it for | `-c` | KV | drafting | prefill | generation | peak VRAM | conf |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | :-: |
| **1** | **max speed** | 65536 | q8_0/q8_0 | `draft-mtp` | — | **53–56** greedy, 42–58 at temp 0.6 | not recorded | **A** |
| 2 | daily driver, full window | 130048 | q4_0/q4_0 | `draft-mtp n4` | — | 47 shallow, 26 @108k | 22893 | B |
| 3 | fastest prefill, no drafting | 130048 | q8_0/q8_0 | none | **830** @108k | 22 shallow, 15 @108k | 22453 | B |
| 4 | maximum context | 196608 | q4_0/q4_0 | none | — | 22 | 22126 | B |

Launcher: `serve-qwen38-27b.ps1 -Profile B` is row 2; `-Profile C` row 3; `-Profile D` row 4.
Row 1 is `-Profile A -Ctx 65536`.

Row 1, expanded:

```powershell
cd S:\OsDevelop\llamacpp
.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf" `
  --host 127.0.0.1 --port 9020 -c 65536 -np 1 `
  -ctk q8_0 -ctv q8_0 `
  -ngl 999 -sm layer -ts 22,43 -fa on -b 2048 -ub 512 -fit off `
  --spec-type draft-mtp --spec-draft-n-max 4 `
  --temp 0.6 --top-p 0.95 --top-k 20
```

**Notes that change the outcome:**

- **`n_max` 3–8 are indistinguishable.** The wiki's old n4 optimum is sampling noise
  (`wiki/15`). 4 is a safe plateau value. To find a real optimum use
  `mtp-test.ps1 -Temperature 0`.
- **Drafting gain is workload-dependent, badly.** 55 t/s is 400 tokens of C#-shaped code at
  greedy with thinking **off**. A real reasoning-on request measured **37.9 t/s, acceptance
  0.577** — still +68% over undrafted, but not +147%.
- **Send `"chat_template_kwargs": {"enable_thinking": false}`** when you don't need reasoning.
  On one request it was the difference between ~400 and **9172** generated tokens.
- Profile A (130k + q8_0 + MTP) peaks at **23560 MiB** and leaves the display GPU ~400 MiB.
  Row 2 exists because of that; prefer it.
- Quality above 108k on row 4 is **unvalidated**.

---

## 2. Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M — MoE, 3 B active, 20.21 GiB

```
S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf
```

Shared flags: `-np 1 -ngl 999 -sm layer -ts 12,29 -fa on -b 2048 -fit off`
(`-ts 12,29` is the only ratio that loads near the ceiling.)

| # | Use it for | `-c` | KV | `-ts` | `-ub` | prefill | generation | peak VRAM | conf |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | :-: |
| **0a** | **best custom-build, q8_0 KV at 64k** (`build-fa`) | 64000 | q8_0/q8_0 | **13,28** | 512 | **2696** (5 reps) | **104.5** | 23101 | **A** |
| **0b** | **best stock-binary, q8_0 KV at 64k** | 64000 | q8_0/q8_0 | **13,28** | 512 | **2371** (2418/2463/2231) | **98.5** | 23306 | **A** |
| **1** | **max prefill, full window** | 130048 | q4_0/q4_0 | 12,29 | 1024 | **2650** (2757/2599/2594) | 105.7 | 23456 | **A** |
| **2** | best all-round | 126976 | q8_0/q8_0 | 12,29 | 512 | 2323 | **109.1** | 23281 | B |
| 3 | quality-first, full window | 130048 | q8_0/q8_0 | 12,29 | 512 | 1850 synth, **1414** real | 47.5 @108k | 23371 | B |
| 4 | max generation, tiny window | 4224 | f16/f16 | 10,18 | 512 | — | **116.7** | — | B |

Row 0b has a real-request corroboration through
`llama-server`: **2314.79 t/s over 28,107 prompt tokens**, 75.06 t/s generation. Note the
non-obvious part is **`-ts 13,28`, not the context** — 13:28 puts ~31.7% of the model on the 3070
against 12:29's 29.3%. A controlled five-repetition A/B with scheduler mode fixed measures that
split itself at **+12.8% prefill**. On the stock binary it also makes the 444 MiB reservation
fail; chapter 18 shows the successful retry is recovery, not the cause of the large gain.

Row 0a uses `llama.cpp\build-fa\bin`, not the stock root binaries. Five interleaved same-source
repetitions at the row's exact arguments averaged 2696 / 104.5. The matched control recovered
through runtime fallback at 2653 / 104.6, so the deterministic one-copy build removes the
warning and memory fragility without claiming a material speed advantage of its own.

Launcher: `serve-qwopus-q4-130k.ps1` is row 1; `serve-qwopus-fast.ps1` row 2;
`serve-qwopus-130k.ps1` row 3.

**Notes that change the outcome:**

- ⛔ **Never pass `--spec-type` to this model.** MTP is a **net loss** here: −7% at `n-max 1`,
  −29% at `n-max 2`, *despite* 67–72% acceptance. A 3 B-active forward pass has nothing
  expensive to amortise. The `-MTP` in the filename is a trap.
- ⛔ **Keep `-ngl 999`. Do not "tidy" it to `-ngl 41`.** `n_layer_all` is 41 but there are **42**
  assignable slots (blocks + output layer), so `-ngl 41` strands one layer on the CPU: measured
  **−54% prefill** (1085 vs 2371). It does switch pipeline parallelism off and save 451 MiB — and
  that is not worth having. `-ngl 42` puts pipelining back on, because the gate *is* the
  full-offload condition. See `wiki/17-pipeline-parallelism-ngl.md`.
- **Row 1 expects an alarming log line.** `-ub 1024`'s larger reservation does not fit, so
  llama.cpp retries with one scheduler copy. An `out of memory` line followed by a result row is
  **OK recovered fallback**, not a failed run and not evidence that the retry caused the speed.
- **`-ts 13,28` (rows 0a/0b) also hits this same fallback at `-c 64000`.** A fresh check
  reproduced `cudaMalloc failed: out of memory` on CUDA0 during `graph_reserve`, silently
  recovered as "retrying without pipeline parallelism" — same pattern as row 1's `-ub 1024`
  above, just a different trigger. `-ts 13,29` (one layer moved off the 3070 onto the 5060 Ti)
  cleared it entirely, with no measured speed cost across a quick 3-run check. **Not yet
  re-verified with `bench-harness.ps1` repetitions** — rows 0a/0b's logged numbers may already
  reflect fallback-mode performance, which would make `13,29` a real (if small) gain rather than
  a wash. Worth a proper re-measurement before promoting it into the table.
- **The `-c 126976` in row 2 is not a typo.** Giving up 2.4% of the window buys 26% more
  prefill, because the curve is steep at the ceiling: 130048 → 1850, 129024 → 2101,
  **126976 → 2323**, 122880 → 2225.
- Needle test at 108k: **PASS** on rows 1 and 3.

### `--no-mmap` — cuts idle system RAM ~94%, no measured speed cost

Quick single-session check (curl-driven completions, 3×200 tokens each — **not** a full
`bench-harness.ps1` campaign, so treat as informal, re-measure with repetitions before fully
trusting the digits) at `-c 64000 -ctk q8_0 -ctv q8_0 -ts 13,29 -fa on -b 2048 -ub 512 -fit off
-ngl 999 -sm layer -np 1`:

| | load time | generation (avg of 3×200 tok) | system RAM (Working Set) after load |
| --- | ---: | ---: | ---: |
| with mmap (default) | 11.2 s | 84.8 t/s | **~20.8 GiB** |
| `--no-mmap` | **6.1 s** | 87.7 t/s | **~1.3 GiB** |

The generation difference (84.8 vs 87.7) is inside this repo's 8% noise floor — **not a
performance trade, a strict win** for a fully-offloaded model, and load was faster too, not
slower as expected.

Root cause: `llama_mmap::impl::unmap_fragment()` on Windows (`llama-mmap.cpp:580-583`) is a
**no-op** — `GGML_UNUSED(first); GGML_UNUSED(last);`, nothing else. The POSIX build really does
call `munmap()` on the unused range once weights are uploaded to GPU (`llama-mmap.cpp:490-507`),
but on Windows that release never happens, because `MapViewOfFile`/`UnmapViewOfFile` can only
(un)map a *whole* view — there is no Windows equivalent of unmapping a sub-range. So the entire
mmap'd GGUF file stays part of the process's resident working set for the life of the server,
not just during load, on this platform specifically. `--no-mmap` avoids the problem instead of
working around it: weight upload routes through 4×64 MiB pinned staging buffers instead
(`llama-model-loader.cpp:1443-1454`), which *are* freed correctly after load (ordinary
`ggml_backend_buffer_free`, no platform gap).

Applies whenever `-ngl 999` reaches full offload — mmap's own benefit (zero-copy CPU-resident
tensors) no longer applies at that point, so only the Windows-only cost remains.
**Recommend `--no-mmap` by default on this machine for any config that fully offloads.**

---

### Asymmetric KV (`q8_0`/`q4_0`) — measured, and deliberately not recommended

Requires `llama.cpp\build-fa\bin` (`GGML_CUDA_FA_ALL_QUANTS=ON`); on the baseline binaries an
asymmetric pair falls off the flash-attn path entirely and collapses to ~19–25 t/s.

It **does** buy room — 184 MiB, and one context step: q8_0/q8_0 OOMs at 180224 while
q8_0/q4_0 loads there. But it is **not** the config to run:

| at c126976 | prefill | generation |
| --- | ---: | ---: |
| stock binaries, q8_0/q8_0 | **2323** | **109.1** |
| build-fa, q8_0/q4_0 | 2302 | 103.0 |
| build-fa, q8_0/q8_0 | 2007 | 101.4 |

`build-fa` also carries `GGML_SCHED_MAX_COPIES=1`. Chapter 18's matched-rebuild A/B clears that
flag as a <2% effect at `ub512`; the old stock-versus-build deficit was build-confounded.
Use it **only** if you need context above 163840, and note `V=q4_0` has never been
quality-gated on this model. c180224 loads but is unstable: generation swung 61–105 t/s across
three runs with VRAM flat against the wall.

---

## 3. Qwen3.6-35B-A3B — MoE. Prefer **UD-Q3_K_XL**: fastest AND smallest

Three quants of the same base model, measured head-to-head at an identical config
(`-c 32768 -ctk q8_0 -ctv q8_0 -ub 512 -ts 12,29`, 3 reps each, stock binaries):

| # | file | size | prefill | generation | peak VRAM | conf |
| --- | --- | ---: | ---: | ---: | ---: | :-: |
| **1** | `unsloth/…-UD-Q3_K_XL` | 15.69 GiB | **2778** | **90.5** | **18273** | **A** |
| 2 | `unsloth/…-UD-Q4_K_S` | 19.46 GiB | 2429 | 91.5 | 22035 | **A** |
| 3 | `lmstudio-community/…-Q4_K_M` | 19.71 GiB | 1579 | 91.3 | 22414 | **A** |

**Row 1 wins on every axis that is not quality**: +14% prefill over Q4_K_S, **+76%** over
lmstudio's Q4_K_M, identical generation, and **~3.8 GiB less VRAM**. Pay for Q4 here only if
you want the quality, never for the speed.

⛔ **Row 3 is unstable — avoid it.** Across five runs it gave prefill 1304 / 1952 / 2079 / 1361
/ 1297 and generation as low as 79.5. It is the largest of the three and sits closest to the
VRAM wall (22464 MiB). Rows 1 and 2 never varied by more than ~4%.

**Generation does not depend on the quant here — 90.5 / 91.5 / 91.3, a 1.0% spread.** This
kills the old claim that "Q3_K CUDA kernels are slower than Q4_K", which came from comparing
Q3_K_XL against a *different fine-tune* (Jackrong's Qwopus). See `wiki/16` §16.3.

Row 1 at full context (`serve-qwen36-130k.ps1`, `-c 130048`, **f16** KV — pass no `-ctk`/`-ctv`
— `-ts 1,2 -ub 512`) remains **confidence C**: the 15.69 GiB file leaves ~2.5 GiB spare so f16
should fit, but nobody has measured that config, and its `-ts 1,2` is a guess. The numbers above
were taken at c32768 with `q8_0` KV, so they do not license the 130k row.

---

## 4. Models on disk with NO measurements — do not invent a command line

There is no recommendation for these. Twelve+ GGUF files are present and unbenchmarked; the
ones large enough to matter:

| size | model | why it is interesting |
| ---: | --- | --- |
| 21.2 GB | `lmstudio-community/Qwen3.6-35B-A3B-Q4_K_M` | The **clean control** for §3's confound: same base model as UD-Q3_K_XL, quant as the only variable |
| 21.2 GB | `Jackrong/Qwopus3.6-35B-A3B-v1-Q4_K_M` | Non-Coder Qwopus; isolates the fine-tune |
| 20.9 GB | `unsloth/Qwen3.6-35B-A3B-UD-Q4_K_S` | |
| 16.8 GB | `Jackrong/Qwopus3.6-27B-Coder-MTP-Q4_K_M` | A **dense** 27B with an MTP head |
| 16.8 GB | `lmstudio-community/gemma-4-26B-A4B-it-Q4_K_M` | Different architecture entirely |

### Deriving one, in order

1. `vram-budget.ps1 -Model <path>` — predict the fit before launching anything.
2. Start from the nearest row above **of the same shape** (dense vs MoE), not the nearest size.
   Chapter 6 vs chapter 10 disagreed on five of six effects between a sparse 35B-A3B and a
   dense 27B, and two flipped sign.
3. `-ts` must be re-derived: it is layers-per-device, so it does not survive a layer-count change.
4. Sweep with `bench-harness.ps1 -Model <path> -OutDir <new dir>`. A new model gets its **own**
   results directory — the main `results.tsv` has no build or hardware column.
5. Gate with `needle-test.ps1` before recommending any lossy KV type.
6. Only claim a drafting speedup after `mtp-test.ps1 -Temperature 0`. At the 0.6 default,
   throughput tracks resampled acceptance at r² = 0.872 and swings ±20%.
