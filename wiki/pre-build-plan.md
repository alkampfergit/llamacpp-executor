# Pre-build plan: what would actually differ from the baseline

Written **before** any compile, so the experiment can be judged on its design rather than on
whatever number comes out. Nothing in this repository has ever been measured on a
self-compiled binary — all 73 recorded runs used the shipped baseline.

---

## The baseline we would be departing from

| Property | Baseline (shipped binaries in the repo root) |
| --- | --- |
| Version | `0.1.2-dev (build 10509, commit fe8156f78)` |
| Source date | 2026-08-19 |
| Compiler | **Clang 20.1.8** for Windows x86_64 |
| CUDA runtime | **13.3.0** (`cudart64_13.dll` FileVersion `6,14,11,13030`) |
| CUDA archs | SASS for `sm_86`, `sm_89`, `sm_120a`, `sm_121a` (141 cubins each); PTX for `compute_75/80/90` |
| `GGML_CUDA_FA_ALL_QUANTS` | **OFF** → only 4 symmetric KV pairs have fast kernels |
| `GGML_SCHED_MAX_COPIES` | **4** (default) |

---

## 1. Intended changes — the three things I am actually testing

| Change | From → To | Hypothesis | Evidence today |
| --- | --- | --- | --- |
| `GGML_CUDA_FA_ALL_QUANTS` | OFF → **ON** | Asymmetric `-ctk q8_0 -ctv q4_0` stops falling off the flash-attention path | Source reading + issue #24485 (96→2361 t/s). **Never run here.** |
| `GGML_SCHED_MAX_COPIES` | 4 → **1** | Makes the lean non-pipelined path deterministic instead of dependent on an allocation failing | We measured the *automatic* fallback as faster; the flag itself is untested |
| `CMAKE_CUDA_ARCHITECTURES` | shipped list → **`86-real;120a-real`** | Shorter compile, smaller binary. **No runtime gain expected** | Neither card JITs today, so this is a build-cost change only |

Only the first has a concrete, falsifiable prediction. The second is plausible. The third
should change nothing measurable and is included to keep the compile affordable.

---

## 2. Unintended changes — the confounds I would be introducing

This is the part that matters, and it is easy to overlook.

| Axis | Baseline | Our rebuild | Status |
| --- | --- | --- | --- |
| **CUDA** | 13.3.0 | 13.1.80 → **13.3.x** | **RESOLVED** — toolkit 13.3 being installed |
| **Compiler (CUDA backend)** | MSVC via `vcvarsall.bat` | MSVC 19.44.35228 | **NOT a confound** — see below |
| **Compiler (host/CPU side)** | Clang 20.1.8 | MSVC 19.44.35228 | Real but irrelevant to GPU throughput |
| **Source** | `fe8156f78` (Aug 19) | `e85caa81e` (Aug 22) — **+73 commits** | **ACCEPTED** as a known variable |
| **Archs dropped** | `sm_89`, `sm_121a`, all PTX | absent | Harmless — no such GPU, and no JIT today |
| **Build shape** | `GGML_BACKEND_DL=ON`, `GGML_CPU=OFF`, target `ggml-cuda` only | monolithic, all targets | Minor structural difference |

### Correction: the compiler axis is much smaller than it first appeared

An earlier draft of this plan said the baseline used Clang 20.1.8 and that switching to MSVC
was a major confound. **That was wrong**, and reading llama.cpp's own release workflow settles
it. The Windows release is assembled from two separately-compiled halves:

| Component | Compiler | Evidence |
| --- | --- | --- |
| `ggml-cuda.dll` — the CUDA backend | **MSVC** | `build-cuda-windows.yml`: `call vcvarsall.bat` + `Ninja Multi-Config`, **no** Clang toolchain file, `--target ggml-cuda` |
| `llama-server.exe`, `llama.dll`, `ggml-cpu-*.dll` | **Clang** | `build-cpu.yml`: `-DCMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake`, which sets `CMAKE_C_COMPILER clang` |

So `llama-server --version` reporting *"built with Clang 20.1.8"* describes how the
**executable** was compiled, not the CUDA kernels. Since every number we care about is
GPU-bound, building locally with MSVC is **closer** to the baseline than the version string
suggests — the CUDA half was always MSVC.

Residual: MSVC 19.44.35228 here versus whatever the `windows-2022` runner shipped for build
10509 (likely a nearby 19.4x), and `-DGGML_BACKEND_DL=ON -DGGML_CPU=OFF` with only the
`ggml-cuda` target upstream versus a monolithic build here.

### What remains

**One variable: 73 commits of source drift**, knowingly accepted. A control build removes even
that, and is cheap because it omits `FA_ALL_QUANTS` — the expensive part. Recommended but no
longer essential.

---

## 3. The fix: the control must be a rebuild, not the baseline

> **Build twice from identical source, compiler and CUDA — once with default flags, once with
> the flags under test.** Compare those two. The shipped baseline stays untouched as a
> historical reference for the 73 recorded runs, not as an experimental control.

```
Build CONTROL   : e85caa81e + MSVC + CUDA 13.1, default flags
Build TREATMENT : e85caa81e + MSVC + CUDA 13.1, FA_ALL_QUANTS=ON, SCHED_MAX_COPIES=1
Compare         : CONTROL vs TREATMENT     <- attributable to the flags
Do NOT compare  : baseline  vs TREATMENT   <- confounded by source+compiler+CUDA
```

Cost: two compiles instead of one. `FA_ALL_QUANTS` makes the treatment build much longer
(49 `fattn-vec*.cu` translation units instead of 4), so trimming targets is worth doing.

The cheaper alternative — `git checkout fe8156f78` in the submodule to match the baseline
source — removes only *one* of the three confounds and detaches the submodule HEAD, which the
parent repo then wants to record. Not worth it.

---

## 4. The commands

```powershell
# CONTROL — default flags, same source/compiler/CUDA as the treatment
.\.claude\skills\building-llamacpp-cuda\scripts\build-llamacpp.ps1 `
  -BuildDir build-control -CudaArch '86-real;120a-real' `
  -CMakeExtra '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_UI=OFF'

# TREATMENT — the two flags under test
.\.claude\skills\building-llamacpp-cuda\scripts\build-llamacpp.ps1 `
  -BuildDir build-fa -CudaArch '86-real;120a-real' `
  -CMakeExtra '-DGGML_CUDA_FA_ALL_QUANTS=ON','-DGGML_SCHED_MAX_COPIES=1', `
              '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_UI=OFF'
```

Deliberately **not** passed, and why:

| Flag | Why not |
| --- | --- |
| `GGML_CUDA_FA=ON`, `GGML_CUDA_GRAPHS=ON` | Already ON by default — inert |
| `GGML_CUDA_FORCE_MMQ` | Unreachable on SM 86/120 (`turing_mma_available` returns true first) |
| `GGML_CUDA_FORCE_CUBLAS` | Kills MMQ on both cards and inflates compute buffers against ~760 MiB of headroom |
| `GGML_NATIVE=OFF` | A CI flag; only guards `CMAKE_CUDA_ARCHITECTURES=native`, which we always set |
| `LLAMA_BUILD_SERVER=OFF` | Would also drop `llama-cli` in this tree |
| `LLAMA_BUILD_TOOLS=OFF` | Carries `llama-bench`, `llama-batched-bench`, `llama-fit-params` — the whole harness |
| `-DLLAMA_SCHED_MAX_COPIES` | **Not an option.** It is `GGML_SCHED_MAX_COPIES`; the `LLAMA_` name is not forwarded and CMake only *warns* |

---

## 5. Verification, in order, before believing any benchmark

1. **Did CMake ignore a flag?** The build script now greps for
   `Manually-specified variables were not used by the project` and prints a red block. A
   mistyped `-D` is only a warning, so this check is the difference between a real test and a
   silently defaulted one.
2. **Did the arch list take?** `cuobjdump --list-elf` on the new `ggml-cuda.dll` — expect
   `sm_86` and `sm_120a` only, and **no PTX**.
3. **Did `FA_ALL_QUANTS` take?** The decisive, seconds-long check:
   ```powershell
   .\llama-fit-params.exe -m <model> -c 130048 -fa on -ctk q8_0 -ctv q4_0 -fitp on
   ```
   Baseline reports **14507 MiB** of host compute for that pair. A working build should report
   *hundreds*. If it still says thousands, the flag did nothing — stop and diagnose.
4. **Sanity-check provenance:** `llama-server.exe --version` should show `e85caa81e`, not
   `fe8156f78`.
5. **Only then benchmark**, CONTROL vs TREATMENT, on a config we already have baseline numbers
   for — and label every row with which build produced it.

---

## 6. Risks worth naming up front

- **CUDA 13.3 → 13.1 is a regression risk** on Blackwell. If TREATMENT looks slower than the
  shipped baseline, this is the first suspect, not the flags. Having a CONTROL build makes that
  distinguishable.
- **Disk and OneDrive.** Two CUDA build trees are several GB each. `build*/` is gitignored, but
  this folder is inside OneDrive, which will try to sync every object file. Consider pausing
  sync, or building outside OneDrive entirely.
- **Compile time.** `FA_ALL_QUANTS` × 2 architectures is the dominant cost. Expect this to be
  the long pole; the control build is much faster.
- **The baseline stays untouched.** Per `CLAUDE.md`, new binaries run from
  `llama.cpp/build-*/bin/`. The build script refuses to write into the root, and that guard is
  tested.

---

## 7. What this experiment can and cannot settle

**Can settle:** whether `FA_ALL_QUANTS` makes asymmetric `q8_0`/`q4_0` viable, and whether
`GGML_SCHED_MAX_COPIES=1` reproduces the fallback path deterministically.

**Cannot settle:** whether a self-compiled binary beats the shipped one. Too many axes differ,
and one of them (CUDA) is worse. Anyone wanting that answer needs Clang 20.1.8 and CUDA 13.3
locally, which we do not have.

**Still open even on success:** whether `q8_0`/`q4_0` is actually *better* than symmetric
`q8_0` once it runs at full speed. It saves ~1 GiB and protects keys, but on this model we
measured `q4_0/q4_0` as no faster than `q8_0/q8_0` — so the win may be memory only, and we have
no quality evidence either way.
