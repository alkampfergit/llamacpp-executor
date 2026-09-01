---
name: building-llamacpp-cuda
description: Build llama.cpp from source on Windows with CUDA, using the exact generator and toolchain combination that actually works on this machine. Use whenever the user wants to compile, build, or rebuild llama.cpp or ggml, needs a custom build option such as GGML_CUDA_FA_ALL_QUANTS or GGML_SCHED_MAX_COPIES, wants to build a PR or feature branch, or hits any of these failures: "No CUDA toolset found", cudafe++ died with ACCESS_VIOLATION 0xC0000005, error MSB3722, "Does not match the generator used previously", or a freshly built .exe that exits instantly with no message. Do NOT use it to tune an already-built binary's runtime flags (-ngl, -ub, -ctk, tensor split) — that is the tuning-llamacpp-configs skill.
argument-hint: "[cmake options, e.g. -DGGML_CUDA_FA_ALL_QUANTS=ON]"
shell: powershell
allowed-tools: PowerShell Bash Read Write Edit Glob Grep
---

# Building llama.cpp with CUDA on Windows

## Detected GPUs (these set the CUDA architectures)

?`nvidia-smi --query-gpu=index,name,compute_cap --format=csv 2>&1`

Convert each `compute_cap` by removing the dot: `8.6` → `86`, `12.0` → `120`. Pass the set as
`-DCMAKE_CUDA_ARCHITECTURES="86;120"`. **Never guess these** — see the warning in §3.

---

## This is a narrow bridge: run the script, do not improvise

The plain instructions in llama.cpp's own docs —

```powershell
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # DOES NOT WORK HERE
cmake --build build -j --config Release
```

— **fail on this machine**, twice, for two unrelated reasons, before producing anything. Both
are documented with their exact error text in
[`references/windows-toolchain-traps.md`](references/windows-toolchain-traps.md).

Use the bundled script. It encodes the working combination and validates each step:

```powershell
./.claude/skills/building-llamacpp-cuda/scripts/build-llamacpp.ps1
./.claude/skills/building-llamacpp-cuda/scripts/build-llamacpp.ps1 -CMakeExtra '-DGGML_CUDA_FA_ALL_QUANTS=ON','-DGGML_SCHED_MAX_COPIES=1'
./.claude/skills/building-llamacpp-cuda/scripts/build-llamacpp.ps1 -Repo S:/Develop/somewhere/llama.cpp -Targets llama-server
```

**Run it; do not read it and hand-type the steps.** The ordering, the single-process
environment, and the generator choice are all load-bearing.

---

## 1. Why the default path fails (know this before you deviate)

| Attempt | Failure | Root cause |
| --- | --- | --- |
| No `-G` flag | `nvcc error : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)` then `error MSB3722` | CMake's default generator is a **Visual Studio 2026 preview** ("18 Community") installed alongside VS2022. CUDA's `nvcc` cannot drive that MSVC toolset. |
| `-G "Visual Studio 17 2022"` | `CMake Error ... No CUDA toolset found.` | The CUDA installer copies its MSBuild integration (`CUDA <ver>.props/.targets/.xml`, `Nvda.Build.CudaTasks.*.dll`) only into the VS instance it detected at install time — the 2026 preview. VS2022's `BuildCustomizations` folder has nothing CUDA-related. Copying them needs **admin rights**. |
| **Ninja + `vcvars64.bat`** | works | Ninja invokes `cl.exe` and `nvcc.exe` directly, so MSBuild and the CUDA `.props`/`.targets` integration are never involved. Sidesteps both problems. |

> **Teaching point.** The winning move was not fixing the broken path, it was **choosing a
> build system that does not need the broken part**. When a toolchain integration is missing
> and repairing it needs admin, prefer the generator that bypasses it.

---

## 2. The recipe, in one process

Two non-obvious requirements:

- **`vcvars64.bat` must be sourced in the same process as `cmake`.** Environment variables set
  by a batch file do not survive into a separate shell invocation, so configure *and* build
  must run inside one script. The bundled script writes a temporary `.bat` for exactly this.
- **Ninja is not bundled** with Visual Studio or CMake. Install it in user space —
  `pip install ninja` — rather than touching `Program Files`.

```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d <repo>
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON ^
      -DCMAKE_CUDA_ARCHITECTURES="86;120"
cmake --build build -j
```

Adjust the `vcvars64.bat` path for your edition (Community / Professional / Enterprise). The
script auto-detects it.

---

## 3. Always set `-DCMAKE_CUDA_ARCHITECTURES` explicitly, and get it right

Auto-detection has misbehaved on this machine, and a wrong value is **silent**: the binary
builds and runs, but a GPU with no matching SASS falls back to JIT-compiling from PTX, which
costs startup time and can lose performance.

> ### ⚠️ A previous build on this machine used `"86;89"` — that is WRONG
> `86` is correct for the RTX 3070 (Ampere, 8.6). But the RTX 5060 Ti is **Blackwell, compute
> 12.0**, not Ada 8.9. CMake's own native detection reported `86-real;120a-real` while the
> build was being told `86;89`. Use **`"86-real;120-real"`** on this hardware.

**Prefer `-real` whenever you know which GPUs will run the binary**, which on a fixed machine
is always. The suffixes are: `-real` = SASS only, `-virtual` = PTX only, **no suffix = both**.
A bare `86;120` therefore ships a complete second copy of every kernel as PTX that the driver
will never reach for, because SASS for both architectures is already present — 141 redundant
blobs per architecture in a default build, 186 with `GGML_CUDA_FA_ALL_QUANTS=ON`.
Do **not** write `120a-real` yourself: `ggml/src/ggml-cuda/CMakeLists.txt` rewrites `12X` to
`12Xa` and preserves the suffix. Omit `-real` only if you deliberately want a JIT fallback for
some *other* GPU. This script appends `-real` automatically to any list it derives from
`nvidia-smi`; an explicit `-CudaArch` is passed through verbatim.

Derive it from live data, never from the GPU's marketing name:

```powershell
$arch = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader | ForEach-Object { $_.Trim() -replace '\.','' } |
         Sort-Object -Unique) -join ';'
```

---

## 4. Where the binaries land

**Ninja is a single-config generator**, so output is flat:

```
build\bin\llama-server.exe          <-- correct
build\bin\Release\llama-server.exe  <-- does NOT exist with Ninja
```

The `Release\` subfolder only appears with the Visual Studio generator. Guides that reference
it assume the MSBuild path.

---

## 5. After building: the silent-exit trap

A freshly built binary can exit instantly with **no error message at all**.

Cause on this machine: the CUDA 13.1 installer placed the runtime DLLs (`cudart64_13.dll`,
`cublas64_13.dll`, `cublasLt64_13.dll`, `cusparse64_12.dll`, `cusolver64_12.dll`, …) in
`CUDA\v13.1\bin\x64\`, but only `CUDA\v13.1\bin\` was on `PATH`.

Fix by either putting the DLL directory on `PATH` for the session, or copying the needed DLLs
next to the `.exe`:

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;$env:PATH"
```

**Always smoke-test after a build** — the script does this automatically:

```powershell
.\build\bin\llama-server.exe --version    # prints build/commit + lists both CUDA devices
```

If it prints nothing, suspect missing runtime DLLs before suspecting the build.

---

## 6. Build options worth knowing

| Option | Effect |
| --- | --- |
| `-DGGML_CUDA=ON` | The whole point. Omit for a CPU-only build. |
| `-DCMAKE_BUILD_TYPE=Release` | Mandatory. A Debug CUDA build is unusably slow. |
| `-DCMAKE_CUDA_ARCHITECTURES="86-real;120-real"` | See §3. `-real` = SASS only; a bare number also emits unusable PTX. |
| `-DGGML_CUDA_FA_ALL_QUANTS=ON` | Compiles the **full** flash-attention KV matrix. Without it only `f16/f16`, `bf16/bf16`, `q8_0/q8_0`, `q4_0/q4_0` exist and any other pair silently costs ~14×. Long compile, much larger binary. |
| `-DGGML_SCHED_MAX_COPIES=1` | Uses one scheduler copy at compile time (it is a `#define`, **not** an env var). Primarily a memory optimisation; controlled `ub512` tests on this pair found <2% throughput change. |
| `-DGGML_CCACHE=OFF` | Silences the "ccache not found" warning. |
| `--target llama-server` | Build one binary instead of everything — much faster when iterating. |

Expected warnings that are **not** problems: `NCCL not found` (multi-GPU perf note only),
`OpenSSL not found` (disables HTTPS in the server), `ccache not found`, `Policy CMP0194`, and
a large volume of `warning #177-D` / `#221-D` from CUDA template instantiation.

---

## 6b. Four traps the script now handles for you

Each of these cost a real build or a wrong conclusion. They are handled automatically, but know
they exist so you recognise the symptoms.

**`GGML_*` not `LLAMA_*`.** `-DLLAMA_SCHED_MAX_COPIES=1` is **not an option** — the real name is
`GGML_SCHED_MAX_COPIES` and the `LLAMA_` spelling is not forwarded. CMake treats an unrecognised
`-D` as a *warning*, compiles with the default, and exits 0. A mistyped flag therefore costs a
full build and silently changes nothing. The script greps for
`Manually-specified variables were not used by the project` and prints a red block.

**The CUDA toolkit must be pinned, not inherited.** A newer toolkit can be installed and still
lose, because `CUDA_PATH` and the leading PATH entry live in the *user registry* and are written
by whichever installer ran last. CMake resolves nvcc from `$ENV{CUDA_PATH}/bin` then PATH. The
script picks the newest installed toolkit (numeric version sort — a string sort puts `v13.1`
after `v13.3`), announces it, warns when it is overriding, and pins it inside the generated
`.bat`. Override with `-CudaToolkit`.

**Run new binaries from their own `bin/`.** ggml resolves backend DLLs relative to the working
directory. Observed: a fresh build loaded `ggml-rpc.dll` and `ggml-cpu-haswell.dll` from the
**baseline** folder, because those filenames differ from the new build's `ggml-cpu.dll`. A
benchmark run that way silently mixes two builds. The smoke test now runs from `bin/` and warns
if any backend resolves from outside it.

**`--version` is not a health check.** `llama-bench` has no `--version` at all, and
`llama-server --version` exits *before* backends initialise, so it never lists devices. Use
`--version` for build/commit provenance and `--list-devices` for hardware. And keep diagnostics
out of the exit code: an invalid probe flag once made a successful build report failure.

## 6c. Provenance: every build gets a pushed branch

Before configuring, the script creates `build/<timestamp>-<label>` in the submodule, writes
`PROVENANCE.md` (base commit, flags, archs, host compiler, CUDA toolkit, GPUs), commits it, and
pushes to the fork. `-Label` names it; `-NoProvenance` skips it for throwaway builds.

Always a **new branch, never `master`** — `master` must stay a clean mirror so
`git rebase upstream/master` keeps working.

> **Do not name the manifest `BUILD-*`.** llama.cpp's `.gitignore` contains `/build*`, and
> Windows git matches ignore patterns **case-insensitively**, so `BUILD-PROVENANCE.md` was
> silently swallowed by a rule meant for the `build/` directory. The branch pushed with no
> record of the flags at all. `git check-ignore -v <file>` names the offending line.

Moving the submodule onto a build branch changes the **parent** repo's gitlink. Mention it;
don't commit it silently.

## 7. Standing rules

### 🚫 Never touch the baseline binaries

Prebuilt binaries sitting **in the repository root** are the **baseline** and are immutable.
Never overwrite, replace, delete, or copy over them, and never copy a fresh build on top of
them — not even temporarily.

Every recorded measurement in `wiki/benchmarks/` was produced by those binaries. Swapping them
out makes old and new numbers silently incomparable, and nothing in the data would reveal it.

**Build in place and run from llama.cpp's standard output directory instead:**

```
<repo>/llama.cpp/build/bin/        <- run new builds from HERE
<repo>/*.exe                        <- baseline, never touched
```

To compare a rebuild against the baseline, run **both** and label which binary produced which
numbers. Report the build with every result — `llama-server.exe --version` prints
`version: … (build N, commit …)`.

- **Never delete a working binary set before the new build passes its smoke test.** Build into
  a separate tree and switch over only after verifying.
- **A generator change requires a clean build directory.** `Does not match the generator used
  previously` means delete `build/` — CMake will not reconfigure across generators.
- **Never write into `Program Files`** to fix CUDA/VS integration. Use Ninja instead.
- **Report build time and the target list**; a full CUDA build is long, and
  `GGML_CUDA_FA_ALL_QUANTS=ON` makes it much longer.
- **If a PR or branch is the goal**, check out the branch *before* configuring, and confirm
  with `git log -1` that you are on it.
- After replacing binaries used by tuned server configs, **re-verify throughput** — a build
  change can move the memory numbers those configs sit against.

---

## References

- [`references/windows-toolchain-traps.md`](references/windows-toolchain-traps.md) — every
  failure signature with its exact error text, cause, and fix.
- Prior worked build on this machine, including a full narrative:
  `S:/Develop/diffusionGemma/diffusiongemma-windows-guide.md`

## Bundled script

- `scripts/build-llamacpp.ps1` — detects vcvars/ninja/CUDA arch, configures with Ninja inside
  the VS environment, builds, then smoke-tests the result. **Run it, don't read it.**



