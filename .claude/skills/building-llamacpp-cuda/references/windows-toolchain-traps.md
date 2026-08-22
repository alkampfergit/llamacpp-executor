# Windows CUDA build traps: failure signatures and fixes

Match the error text you actually got, then apply that fix. Do not change several things at
once — each of these has a single distinct cause.

## Contents

1. [`cudafe++ died with status 0xC0000005`](#1-cudafe-died-with-status-0xc0000005)
2. [`No CUDA toolset found`](#2-no-cuda-toolset-found)
3. [`Does not match the generator used previously`](#3-does-not-match-the-generator-used-previously)
4. [The binary exits instantly with no message](#4-the-binary-exits-instantly-with-no-message)
5. [`build\bin\Release\` does not exist](#5-buildbinrelease-does-not-exist)
6. [Wrong CUDA architecture — the silent one](#6-wrong-cuda-architecture--the-silent-one)
7. [Harmless warnings](#7-harmless-warnings)
8. [Verified toolchain on this machine](#8-verified-toolchain-on-this-machine)

---

## 1. `cudafe++ died with status 0xC0000005`

```
CUDACOMPILE : nvcc error : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)
...BuildCustomizations\CUDA 13.1.targets(803,9): error MSB3722: The command
  ""...\nvcc.exe" -gencode=arch=compute_75,code=\"sm_75,compute_75\" ... exited with code 5.
  Please verify that you have sufficient rights to run this command.
```

**Cause:** CMake picked its *default* generator, which on this machine is **"Visual Studio 18
2026"** — a preview install sitting alongside VS2022. `nvcc` cannot drive that MSVC toolset
(MSVC 14.51), and it crashes on the very first CUDA file, which is CMake's own compiler-ID
probe. Note the misleading `compute_75` and the "sufficient rights" text: neither is the
problem.

**Confirm:** `cmake --help` lists generators with `*` on the default.

**Fix:** do not use the default. Use the Ninja generator (see trap 2's fix) — forcing
`-G "Visual Studio 17 2022"` runs straight into the next trap.

---

## 2. `No CUDA toolset found`

```
CMake Error at .../CMakeDetermineCompilerId.cmake:714 (message):
  No CUDA toolset found.
```

…even though configuration *found* the toolkit a few lines earlier:

```
-- Found CUDAToolkit: .../CUDA/v13.1/include (found version "13.1.80")
-- CUDA Toolkit found
```

**Cause:** those two statements are about different things. CMake found the CUDA *toolkit*,
but the **VS/MSBuild integration** is missing. The CUDA installer copies

```
CUDA 13.1.props   CUDA 13.1.targets   CUDA 13.1.xml   Nvda.Build.CudaTasks.v13.1.dll
```

into the `BuildCustomizations` folder of whichever Visual Studio it detects **at install
time** — here, only the 2026 preview:

```
C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Microsoft\VC\v180\BuildCustomizations\
```

VS2022's equivalent (`...\2022\Enterprise\MSBuild\Microsoft\VC\v170\BuildCustomizations\`) has
nothing CUDA-related.

**Do not fix it by copying the files.** Writing into `Program Files` needs admin (a plain copy
fails with `Permission denied`), and it leaves the machine in a hand-patched state.

**Fix — use Ninja instead.** Ninja calls `cl.exe` and `nvcc.exe` directly and never consults
MSBuild or those `.props`/`.targets` files, so the whole problem disappears:

```powershell
pip install ninja      # not bundled with VS or CMake; user-space, no admin
```

then configure **and** build inside one `vcvars64.bat` session:

```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
cd /d <repo>
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;120"
cmake --build build -j
```

Both commands must be in the same script: **environment variables set by a batch file do not
survive into a separate shell invocation.** Running `vcvars64.bat` in one call and `cmake` in
the next silently loses the environment.

---

## 3. `Does not match the generator used previously`

```
CMake Error: Error: generator : Ninja
Does not match the generator used previously: Visual Studio 17 2022
Either remove the CMakeCache.txt file and CMakeFiles directory or choose a different
binary directory.
```

**Cause:** CMake cannot reconfigure an existing build tree with a different generator.

**Fix:** delete the build directory. Typically hit right after switching away from a failed
Visual Studio attempt.

```powershell
Remove-Item -Recurse -Force <repo>\build
```

---

## 4. The binary exits instantly with no message

Builds fine, links fine, runs and returns nothing at all — no error, no usage text.

**Cause:** missing CUDA **runtime** DLLs. The CUDA 13.1 installer placed them in

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\
```

— `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`, `cusparse64_12.dll`,
`cusolver64_12.dll`, … — while only `...\v13.1\bin\` was on `PATH`. Windows fails the load
before `main`, so nothing is printed.

**Fix (session):**

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;$env:PATH"
```

**Fix (permanent):** add that directory to the user `PATH`, or copy the DLLs next to the
`.exe` — which is what shipped llama.cpp release zips do, and why they "just work".

**Diagnose:** `dumpbin /dependents <exe>` lists what it wants to load.

---

## 5. `build\bin\Release\` does not exist

**Cause:** you are following a guide written for the Visual Studio generator. VS is
multi-config, so it emits `build\bin\Release\`. **Ninja is single-config**, so everything lands
flat:

```
build\bin\llama-server.exe          <-- Ninja
build\bin\Release\llama-server.exe  <-- Visual Studio generator only
```

`-DCMAKE_BUILD_TYPE=Release` is what selects the configuration for Ninja, and `--config
Release` on the build command is a no-op there.

---

## 6. Wrong CUDA architecture — the silent one

This one produces **no error at all**, which makes it the most dangerous entry here.

A prior build on this machine used:

```
-DCMAKE_CUDA_ARCHITECTURES="86;89"
```

`86` is right for the RTX 3070 (Ampere 8.6). **`89` is Ada — the RTX 5060 Ti is Blackwell,
compute 12.0.** CMake said so in the same log while being told otherwise:

```
-- Replacing 120-real in CMAKE_CUDA_ARCHITECTURES_NATIVE with 120a-real
-- Using CMAKE_CUDA_ARCHITECTURES=86;89 CMAKE_CUDA_ARCHITECTURES_NATIVE=86-real;120a-real
```

A card with no matching SASS in the binary falls back to JIT-compiling from PTX at load: slow
startup, and potentially slower kernels. The program still runs and still produces correct
output, so nothing draws your attention to it.

**Fix — derive it, never type it from memory:**

```powershell
nvidia-smi --query-gpu=index,name,compute_cap --format=csv
# 0, NVIDIA GeForce RTX 3070,    8.6   -> 86
# 1, NVIDIA GeForce RTX 5060 Ti, 12.0  -> 120
```

**Every installed card must appear in the list.** With a mixed pair like this one, omitting
either is a silent performance loss on that card. Correct value here: **`"86;120"`**.

Do not infer the number from the marketing name; RTX 50-series is Blackwell (12.x), not Ada.

---

## 7. Harmless warnings

These are all expected and safe to ignore:

| Message | Meaning |
| --- | --- |
| `Could NOT find NCCL` / `performance for multiple CUDA GPUs will be suboptimal` | NCCL is for multi-GPU collectives; llama.cpp's layer split does not need it. |
| `Could NOT find OpenSSL` / `HTTPS support disabled` | `llama-server` serves plain HTTP. Fine for localhost. |
| `Warning: ccache not found` | Optional compile cache. `-DGGML_CCACHE=OFF` silences it. |
| `Policy CMP0194 is not set: MSVC is not an assembler for language ASM` | CMake policy noise. |
| Many `warning #177-D`, `#221-D` from `mmq.cuh` / `common.cuh` | Routine CUDA template-instantiation warnings; there are hundreds. |

---

## 8. Verified toolchain on this machine

The combination that produced a working CUDA build:

| Component | Version / path |
| --- | --- |
| CUDA | 13.1.80 (`nvcc` release 13.1) |
| Host compiler | MSVC 19.44.35228 (VS2022 Enterprise, toolset 14.44.35207) |
| `vcvars64.bat` | `C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\` |
| CMake | 4.4.0 |
| Generator | **Ninja** (`pip install ninja`) |
| Architectures | `86;120` (RTX 3070 + RTX 5060 Ti) |
| Windows SDK | 10.0.26100.0 |
| Also installed | Visual Studio "18" 2026 **preview** — the source of traps 1 and 2 |

Build produced 389 Ninja targets ending in `bin\llama-*.exe`, with `llama.dll`,
`llama-common.dll` and `ggml-cuda.dll` alongside.

A longer narrative of the original debugging session, written up as a guide, is at
`S:/Develop/diffusionGemma/diffusiongemma-windows-guide.md`.
