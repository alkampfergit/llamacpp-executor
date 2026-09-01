# Chapter 7 — Recommended Configurations

Copy-paste commands for this machine. Each states **what it optimises**, **what it gives
up**, and whether the numbers are **measured** or a **starting point**.

Ready-to-run scripts:
[`../.claude/skills/tuning-llamacpp-configs/scripts/`](../.claude/skills/tuning-llamacpp-configs/scripts/).

---

## Choosing between them

The real decision is **KV cache precision vs prefill speed**, and the honest answer depends
on what you do:

```
Do you need the full 130k window?
├── NO (prompts under ~120k)
│     → Profile C.  q8_0 quality AND 2323 t/s. The best all-round config.
└── YES
      ├── Quality matters more than latency (code edits, careful reasoning)
      │     → Profile A.  q8_0 KV, 1850 t/s.
      └── Latency matters more (search, retrieval, bulk reading)
            → Profile B.  q4_0 KV, 2650 t/s -- 43% faster, and it PASSED a
                          108k needle-retrieval test (§6.4).
```

**Before any of them, do the two free things:**

1. Set **NVIDIA Control Panel → Manage 3D Settings → CUDA - Sysmem Fallback Policy →
   "Prefer No Sysmem Fallback"**. Without it, an over-large config silently runs at a fifth
   speed. Two configurations in chapter 6 were caught doing exactly this (§6.5).
2. Close GPU-using desktop apps. Every profile here sits within a few hundred MiB of the
   ceiling.

> ### ⚠️ Do not set `GGML_SCHED_MAX_COPIES` as an environment variable
> Earlier versions of this wiki told you to. **It has no effect** — it is a compile-time
> `#define`, not a runtime variable (§6.2). The commands below no longer set it. A successful
> automatic retry saves reservation memory, but controlled tests do not show a material `ub512`
> speed gain from it. The large measured improvement comes from the tensor split (chapter 18).

---

## Profile A — full 130k, `q8_0` KV (quality-first)

**Measured:** 1850 t/s prefill synthetic; **1414 t/s and 47.5 t/s generation on a real
107,743-token request**. Needle test: **PASS**.
**Gives up:** 30% of prefill versus Profile B.

```powershell
cd S:\OsDevelop\llamacpp

.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf" `
  --alias qwopus-coder `
  --host 0.0.0.0 --port 9010 `
  -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 12,29 -fa on `
  -b 2048 -ub 512 `
  -ctk q8_0 -ctv q8_0 `
  -fit off `
  -cram 24576 `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

| Flag | Why |
| --- | --- |
| `-ctk/-ctv q8_0` | Symmetric pair on the compiled FA path; the quality choice (§6.3) |
| `-ub 512` | 1850 t/s. `ub 1024` does **not** fit with q8_0 at 130k |
| `-ts 12,29` | The only ratio that loads. 11,30 and 13,28 both fail (§6.8) |
| `-np 1` | Otherwise each request gets `130048/N` tokens |
| `-fit off` | Don't let the fitter alter a measured config |
| `-cram 24576` | Prompt cache — worth more than everything above (§5.5) |

---

## Profile B — full 130k, `q4_0` KV (speed-first)

**Measured:** 2650 t/s prefill synthetic (2757 / 2599 / 2594 over three runs); **1758 t/s and
46.6 t/s on the same real 107,743-token request** — 15 seconds faster end-to-end than
Profile A. Needle test: **PASS**.
**Gives up:** KV precision. Validated for *retrieval*; not validated for long-context
reasoning or code-edit fidelity (§6.4).

```powershell
cd S:\OsDevelop\llamacpp

.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf" `
  --alias qwopus-coder `
  --host 0.0.0.0 --port 9010 `
  -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 12,29 -fa on `
  -b 2048 -ub 1024 `
  -ctk q4_0 -ctv q4_0 `
  -fit off `
  -cram 24576 `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

The `q4_0` cache is 700 MiB smaller than `q8_0`, and that is exactly what makes `-ub 1024`
reachable. This measured configuration also needs the one-copy retry to fit, but the retry is
not the demonstrated cause of its throughput.

> **Expect an alarming log line.** You will see `failed to allocate compute buffers` followed
> by `retrying without pipeline parallelism`. This profile may recover and continue; confirm
> `model loaded` and `listening on` follow it. Treat the line as a recorded execution-mode change,
> not as a speed feature (§6.2 and chapter 18).

---

## Profile C — 127k, `q8_0` KV (best all-round)

**Measured:** 2323 t/s prefill, 109 t/s generation — the **fastest `q8_0` configuration
found**, and faster than Profile B's `q4_0` at `ub 512`.
**Gives up:** 3072 tokens of context (2.4%).

```powershell
cd S:\OsDevelop\llamacpp

.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf" `
  --alias qwopus-coder `
  --host 0.0.0.0 --port 9010 `
  -c 126976 -np 1 `
  -ngl 999 -sm layer -ts 12,29 -fa on `
  -b 2048 -ub 512 `
  -ctk q8_0 -ctv q8_0 `
  -fit off `
  -cram 24576 `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

**If you can live with 127k instead of 130k, this is the profile to use.** You keep the
higher-precision KV cache *and* get 26% more prefill than Profile A. Giving up 2.4% of the
window for 26% of the speed is the best trade in this entire wiki.

---

## Profile D — Qwen3.6 Q3_K_XL, 130k with headroom

**For:** long context without living at the ceiling. At 15.69 GiB the model leaves ~2.5 GiB
spare, so `f16` KV and `-ub 512` both fit.
**Status:** the 130k load is **proven**; `llama-bench` measured 3107–3206 t/s prefill and
~94 t/s generation. `-ts` below is a **starting point** — verify.
**Gives up:** Q3 quality — and **nothing else.** ⚠️ This line used to claim "~20% generation
speed versus Q4_K_M (117 → 94 t/s)". That comparison crossed a fine-tune boundary. Measured
against Q4 quants of the *same* base model, generation is within **1.0%** and this file has the
**fastest prefill of the three** at 3.8 GiB less VRAM. See
[chapter 16 §16.3](16-best-commandlines.md).

```powershell
cd S:\OsDevelop\llamacpp

.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf" `
  --alias qwen3.6-35b `
  --host 0.0.0.0 --port 9010 `
  -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 1,2 -fa on `
  -b 2048 -ub 512 `
  -fit off `
  -cram 24576 `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

No `-ctk`/`-ctv`: **f16 KV fits here, and f16 is the fastest and most accurate KV there is.**
Only quantise when you must. Your proven `-ts 1,3` also works; `1,2` relieves the display
GPU, which is the binding constraint.

---

## What *not* to do

| Anti-pattern | Why | Ref |
| --- | --- | --- |
| `$env:GGML_SCHED_MAX_COPIES = "1"` | **No effect.** Compile-time define | §6.2 |
| Mixing KV types (`-ctk q8_0 -ctv q4_0`) | Asymmetric → attention leaves the GPU, −89% | §6.3 |
| `-ctk q5_1` / `q5_0` / `q4_1` / `iq4_nl` | Not a supported FA type at all, −93% | §6.3 |
| `-ncmoe 2` to make something fit | −69% prefill for 5% of the model | §6.7 |
| `-fa off`, or omitting `-fa` | 9 GiB of extra compute buffers | §3.3 |
| `-ub 1024` with `q8_0` at 115k–123k | Saturates the display GPU → spills to system RAM, 1036 t/s | §6.5 |
| `-np 4` with `-c 130048` for one user | Each request capped at 32512 tokens | §2.4 |
| `-mg 1` with `-sm layer` | No effect | §3.1 |
| Assuming a smaller quant is faster | Sometimes true, but ⚠️ **not for the reason §6.1 gives** — same-base-model quants generate within 1.0%, and Q3_K_XL prefills *fastest* | [§16.3](16-best-commandlines.md) |
| Tuning before the sysmem policy is set | You will measure overflow, not speed | §5.3 |

---

## 7.5 The rebuild that would fix all of this

Both remaining awkwardnesses — relying on an allocation failure for speed, and being unable
to use asymmetric KV — are build-time limitations:

> ⚠️ **The obvious command does not work on this machine.** An earlier version of this section
> said to run `cmake -B build -DGGML_CUDA=ON ...`, which fails twice here before producing
> anything: CMake defaults to a Visual Studio 2026 *preview* generator that crashes `nvcc`'s
> `cudafe++`, and forcing the VS2022 generator then fails with `No CUDA toolset found` because
> the CUDA installer registered its MSBuild integration only for the preview.

The combination that actually works uses the **Ninja** generator inside a `vcvars64.bat`
environment, with both GPU architectures named explicitly — `86` for the 3070 and **`120`**
for the 5060 Ti (Blackwell, *not* `89`/Ada):

```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
cd /d S:\OsDevelop\llamacpp\llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON ^
      -DCMAKE_CUDA_ARCHITECTURES="86;120" ^
      -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build -j
```

Configure and build must be in the **same** script — environment variables from
`vcvars64.bat` do not survive into a separate shell. Ninja is single-config, so binaries land
flat in `build\bin\`, not `build\bin\Release\`. Requires `pip install ninja`.

Easier: use the `building-llamacpp-cuda` skill, which detects the toolchain, derives the
architectures from `nvidia-smi`, and smoke-tests the result:

```powershell
.\.claude\skills\building-llamacpp-cuda\scripts\build-llamacpp.ps1 `
  -CMakeExtra '-DGGML_SCHED_MAX_COPIES=1','-DGGML_CUDA_FA_ALL_QUANTS=ON'
```

- `GGML_CUDA_FA_ALL_QUANTS=ON` enables `-ctk q8_0 -ctv q4_0`: precise keys, compact values.
  **Measured at ×30.5 prefill, and free thereafter** — see [chapter 14](14-build-experiment.md).
- `GGML_SCHED_MAX_COPIES=1` frees ~845 MiB **at `-ub 512` only** — at `-ub 1024` both builds
  peak identically and the saving disappears (§14.9). Do not budget for it.

> 🚫 **Neither flag makes anything faster.** Best config against best config, at matched key
> precision, both at `-ub 1024`: control `q8_0`/`q8_0` = 1257.91 t/s / 22876 MiB, treatment
> `q8_0`/`q4_0` = 1255.55 t/s / **21860 MiB**. Identical throughput, ~1 GiB less memory. The
> whole return on this rebuild is VRAM headroom. If you want prefill throughput on this model,
> **set `-ub 1024`** — worth 3.1% on the binaries you already have, no compiler required.

> ⚠️ **Correction (1 of 2).** This section used to say `GGML_SCHED_MAX_COPIES=1` "makes the fast
> non-pipelined path deterministic rather than a side effect of running out of memory."
> [Chapter 14](14-build-experiment.md) measured that and **it is wrong.** There is no speed
> gain — the treatment build was 3.2–3.4% *slower* on symmetric KV (inside the ±8% noise floor,
> but 4 of 4 measurements in the same direction). The earlier observation that the automatic
> non-pipelined fallback "measured faster" was an artefact: that fallback fires when an
> allocation *fails*, so what was being measured was a run that had stopped spilling into
> host RAM. The flag buys you the ~845 MiB that caused that difference — not the speed.
>
> ⚠️ **Correction (2 of 2).** This section also expected the flag to "free enough to run `q8_0`
> KV at `-ub 1024` at full 130k". It does not need to: [§14.9](14-build-experiment.md) shows the
> **control** already runs `-ub 1024` at full 130k in 22876 MiB, and gains 3.1% doing so. There
> was no locked door for the freed memory to open.

Cost: **10.4 min** to compile on this machine, and `ggml-cuda.dll` grows 84.2 → 116.2 MB.
`FA_ALL_QUANTS` was expected to dominate the compile and did not, so this is much cheaper to
try than this section originally assumed.

---

## Verifying a profile after launch

```powershell
curl.exe http://localhost:9010/health          # alive?
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv   # inside physical VRAM?
```

Check the server log for `n_ctx_slot` — it should be ~130048, not 32512 (that would mean
`-np` divided it). Then send a real request and read the timings llama.cpp prints:

```
prompt eval time = ... tokens per second      <- real prefill
       eval time = ... tokens per second      <- real generation
```

**Those two lines are the only numbers that matter.** Everything in chapter 6 exists to make
them good.

On a second, similar request, look for:

```
selected slot by LCP similarity, f_sim_best = 0.987, f_keep = 0.995
```

Above 0.95 means prompt caching is working and your effective prefill wait has collapsed by
an order of magnitude. If the line never appears, check `-np` first.

Finally, to confirm long-context quality on your own settings (`-Model` is mandatory; add
`-OutDir wiki\benchmarks` to keep results in the wiki's evidence trail):

```powershell
.\.claude\skills\tuning-llamacpp-configs\scripts\needle-test.ps1 -Label mine -Ctk q8_0 -Ctv q8_0 -Ub 512 `
  -Model "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf" `
  -OutDir "S:\OsDevelop\llamacpp\wiki\benchmarks"
```

---

Next: [Chapter 8 — Troubleshooting](08-troubleshooting.md).
