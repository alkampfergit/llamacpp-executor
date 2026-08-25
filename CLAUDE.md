# CLAUDE.md — llamacpp-executor

Prebuilt Windows CUDA llama.cpp binaries in the root (gitignored), a tuning wiki in `wiki/`,
and the upstream source as a submodule in `llama.cpp/`.

Hardware: RTX 3070 (8 GiB, device 0) + RTX 5060 Ti (16 GiB, device 1, **drives the display**).
Compute capabilities `8.6` and `12.0` → `-DCMAKE_CUDA_ARCHITECTURES="86-real;120-real"`.
**Use the `-real` suffix.** A bare `86;120` also emits PTX for those same two architectures,
which neither card can ever use because both have native SASS — measured at 141 redundant PTX
blobs per architecture in `build-control` and 186 in `build-fa`. ggml rewrites `120-real` →
`120a-real` itself, so do not type the `a`. The build script now appends `-real` to any list it
derives from `nvidia-smi`. See `wiki/14-build-experiment.md` §14.2.

## When a benchmark result looks strange, read the source

**A surprising measurement is a signal to go read llama.cpp's source in `llama.cpp/`, not to
theorise about it.** Guessing at causes has produced wrong conclusions in this repo twice, and
both were settled in minutes by reading the code:

| Symptom | Wrong guess | What the source said |
| --- | --- | --- |
| KV cache `q8_0`/`q4_0` was 14× slower than `q8_0`/`q8_0` | "4-bit KV is slow" | `ggml/src/ggml-cuda/fattn.cu`: `if (K->type != V->type) return BEST_FATTN_KERNEL_NONE;` — only four *symmetric* pairs are compiled. Attention was leaving the GPU. (That guard is `#ifndef GGML_CUDA_FA_ALL_QUANTS`; `wiki/14-build-experiment.md` measures the rebuild at ×30.5.) |
| Setting `$env:GGML_SCHED_MAX_COPIES=1` appeared to gain 13% | "it disables pipeline parallelism" | `ggml/src/ggml-backend.cpp`: it is a compile-time `#define`, never read from the environment. The gain was noise. |

Applies to: a result that contradicts a documented claim, a flag that "works" but is
inexplicably slow, non-monotonic behaviour, a silent fallback, or any gain near the ±8% noise
floor. Prefer `Grep` over the submodule to a web search — the checked-out code is the code
that produced the measurement.

## Prefer a matched local build for causal testing

For investigations into *why* a result occurs, prefer a verified build under `llama.cpp/` over
the immutable root baseline. A local build is usually newer, its source is available, and narrow
logging, counters or assertions can be added around the suspected decision. Verify the build's
commit rather than choosing by timestamp, and run it from its own `bin/`.

Instrumentation is an experiment: build a control and an instrumented variant from the same
commit, toolchain and CMake options, use identical runtime arguments, and keep them in separate
build directories. Change only the diagnostic code. A comparison against the root binaries does
not isolate instrumentation because source, compiler and flags may also differ. Never modify or
replace the root binaries while doing this, and do not commit diagnostic source edits unless the
user asks to retain them.

## Keep the submodule current before reading it

Stale source explains nothing about a binary built from a newer tree. Before investigating,
fetch and rebase:

```powershell
cd S:\OsDevelop\llamacpp\llama.cpp
git fetch upstream
git rebase upstream/master          # NOTE: master, not main
```

- `upstream` = `https://github.com/ggml-org/llama.cpp.git` (push URL deliberately disabled).
  `origin` = the user's fork, `alkampfergit/llama.cpp`.
- **The default branch is `master`.** `upstream/main` does not exist.
- If the rebase would not fast-forward, stop and report rather than resolving conflicts
  unasked — local commits on the fork may be deliberate.
- Moving the submodule's checkout changes the gitlink in the parent repo and needs its own
  commit there. Mention it; don't commit it silently.
- **Check the source you read matches the binary you measured.** `llama-server.exe --version`
  reports `version: 0.1.2-dev (build 10509, commit fe8156f78)` — compare that commit against
  the submodule's `git log -1`. If they diverge, say so before attributing observed behaviour
  to the code you just read. (`llama-bench` has no `--version`; use `llama-server` or
  `llama-cli`, or read the `build:` line at the end of normal `llama-bench` output.)

## NEVER modify the executables in the root

The `*.exe` and `*.dll` in `S:\OsDevelop\llamacpp\` are the **baseline** — build
`fe8156f78 (10509)`. They are immutable.

- **Never** overwrite, replace, delete, or copy over them.
- **Never** copy freshly built binaries into the root, not even "just to try it".

Every one of the 73 measurements in `wiki/benchmarks/results.tsv` was taken with these
binaries. Replacing them silently invalidates the entire recorded dataset, because a later
run would no longer be comparable to an earlier one and nothing in the data would say so.

**When a rebuild is needed**, build in place and run from llama.cpp's standard output
directory:

```
S:\OsDevelop\llamacpp\llama.cpp\build\bin\        <- new builds run from HERE
S:\OsDevelop\llamacpp\*.exe                        <- baseline, never touched
```

Ninja is single-config, so binaries land flat in `build\bin\` — there is no `build\bin\Release\`.

To compare a rebuild against the baseline, run **both** and label which binary produced which
numbers; never swap one in for the other. State the build in any result you report:
`llama-server.exe --version` gives `version: 0.1.2-dev (build 10509, commit fe8156f78)` for the
baseline.

## Scripts live in ONE place — `.claude/skills/tuning-llamacpp-configs/scripts/`

There is no `wiki/scripts/` directory. Every operative PowerShell script — the benchmark
harness, the MTP and needle quality gates, the `serve-*` launchers, and the deep-probe /
resume-benchmarks helpers — lives under `.claude/skills/tuning-llamacpp-configs/scripts/`.
The wiki's chapters only *reference* these scripts by path; they do not carry their own copy.

This used to not be true. Three scripts (`bench-harness.ps1`, `mtp-test.ps1`,
`needle-test.ps1`) existed twice — once under `wiki/scripts/`, once under the skill — and the
two copies diverged, silently, twice (see the historical scars below). That duplication has
been removed. **Do not recreate it** by copying a script back into `wiki/` "just for the
wiki" — add a reference to the skill copy instead.

**`bench-harness.ps1`, `mtp-test.ps1` and `needle-test.ps1` are portable, not this-repo-specific**:
`-Model` is mandatory (there is no built-in default model path), `llama.cpp` is auto-located
from the script's own path, and results default to `bench-results/` next to the binaries
rather than `wiki/benchmarks/`. That default is right for a copy run from an arbitrary
location, but **it is not this repo's own workflow** — `wiki/benchmarks/` is the durable,
committed evidence trail (see Conventions below). All three accept `-OutDir` to override the
default; pass `-OutDir S:\OsDevelop\llamacpp\wiki\benchmarks` whenever a run in *this*
repo should land in the wiki's evidence trail instead of `bench-results/`. Don't let evidence
start landing in the portable default silently — say where it went. (`resume-benchmarks.ps1`
already defaults `-Model`/`-OutDir` to this repo's own values; the `serve-*.ps1` and
`deep-probe-q38.ps1` scripts were always this-repo-specific and hardcode their own paths.)

> **Historical scars — why "fix one copy" was never good enough, back when there were two:**
> - The MTP detector once grepped a hardcoded `unused tensor blk.40`. The MTP layer sits at
>   `blk.<n_layer>`, which is model-specific — `blk.40` on one model, `blk.64` on another — so
>   the hardcoded version reported "MTP loaded" for *every* run on any other model, baselines
>   included. It was fixed in one copy, left broken in the other, and had to be fixed a third
>   time in a doc comment before both copies matched. It now matches
>   `unused tensor blk\.\d+\.nextn`, which is model-agnostic.
> - `llama-bench --version` (not a valid flag) was documented as wrong in this file, then
>   reintroduced in the build script's smoke test, where it made a **successful** build report
>   exit code 1.
>
> Both were caught only by grepping the whole repo for the pattern after fixing one instance.
> **Standing rule, and it survives the duplication going away: after fixing any bug, grep the
> whole repo for siblings before declaring it done.** A fix applied to one instance of a
> repeated pattern is not a fix.
>
> ```powershell
> # always do this after a fix
> Get-ChildItem -Recurse -Include *.ps1,*.md | Select-String -Pattern '<the bad pattern>'
> ```

## A measurement is only valid if all of these hold

Throughput numbers in this repo are worthless without these preconditions. Check them before
recording anything.

1. **The NVIDIA sysmem-fallback policy covers the binary being measured.** It is keyed on the
   full executable path, so covering `llama-server.exe` does nothing for
   `llama-batched-bench.exe` — which produced all 73 recorded runs. Proven here: the policy was
   set two hours before a campaign and every run still spilled silently. See
   `wiki/nvidia-sysmem-fallback-paths.md`. Without it, "slow" and "overflowing" are
   indistinguishable.
2. **New builds run from their own `bin/`.** ggml resolves backend DLLs relative to the working
   directory. A fresh build launched from the repo root loaded `ggml-rpc.dll` and
   `ggml-cpu-haswell.dll` out of the **baseline** folder, because those filenames differ from
   the new build's `ggml-cpu.dll`. That silently mixes two builds.
3. **Every number is labelled with the build that produced it.** Baseline is
   `build 10509, commit fe8156f78, Clang 20.1.8, CUDA 13.3`. Anything else must say so.
4. **The result row exists.** A run that logged `out of memory` but still produced a row
   succeeded via llama.cpp's one-copy fallback — it is `OK`, not `OOM`. New result files record
   `execution_mode=fallback-single-copy`; a retry with no result is `fallback-failed` and remains
   `OOM`. Do not infer a speedup from the mode label: chapter 18 isolates the tensor split as the
   large effect.

## "What should I run for model X?" — one file answers this

**`.claude/skills/tuning-llamacpp-configs/references/best-commandlines.md` is the canonical
table of best-known command lines per model**, two to four per model, with prefill/generation/
peak-VRAM and a confidence grade on every row. Read it before composing a launch command by
hand, and update it whenever a campaign produces a better config.

- Grades: **A** = repeated, logged, rows in a `results.tsv`. **B** = single logged run, trust
  the config not the digits. **C** = prose in a wiki chapter, no row — a starting point only.
- `wiki/16-best-commandlines.md` is the companion chapter. It carries provenance, confounds and
  the unmeasured-model inventory, and **deliberately does not repeat the commands** — one
  source only, for the reason in "Scripts live in ONE place" above.
- Only two models have grade-A/B data (Qwen3.8-27B dense, Qwopus3.6-35B-A3B MoE). A third is
  grade C. **A dozen more GGUFs are on disk with no measurements at all** — do not invent a
  command line for those; §4 of the reference gives the derivation order.

## Skills

- `tuning-llamacpp-configs` — find optimal runtime flags (context, ubatch, split, KV type).
  Its `references/best-commandlines.md` is the lookup table described above.
- `building-llamacpp-cuda` — build from source. The plain `cmake -B build -DGGML_CUDA=ON …`
  from llama.cpp's docs **fails on this machine**; use the skill.

## Building

- **Use the `building-llamacpp-cuda` skill.** The documented `cmake -B build -DGGML_CUDA=ON …`
  fails twice on this machine.
- **ggml options are `GGML_*`, not `LLAMA_*`.** `-DLLAMA_SCHED_MAX_COPIES=1` is not an option;
  the real name is `GGML_SCHED_MAX_COPIES`. CMake treats an unrecognised `-D` as a **warning**,
  compiles with the default and exits 0 — so a mistyped flag costs a full build and changes
  nothing. The build script now greps for `Manually-specified variables were not used`.
- **The CUDA toolkit must be pinned, not inherited.** 13.3.73 is installed but `CUDA_PATH` and
  PATH still say 13.1, so CMake would silently pick the older one. The script selects the newest
  and says when it is overriding. Baseline binaries were built against **13.3**.
- **Every compile pushes a provenance branch** to the fork: `build/<timestamp>-<label>` carrying
  `PROVENANCE.md` (base commit, flags, archs, compiler, toolkit, GPUs). Always a *new* branch,
  never `master` — `master` stays a clean mirror so `git rebase upstream/master` keeps working.
  Note the manifest must **not** be named `BUILD-*`: llama.cpp's `.gitignore` has `/build*` and
  Windows git matches case-insensitively, which silently swallowed the first one.
- **Compare a rebuild against a CONTROL rebuild, never against the shipped baseline.** The
  baseline differs in source (+73 commits, `0.1.2-dev` → `0.2.0-dev`) and host compiler as well
  as flags. See `wiki/pre-build-plan.md`.

## Conventions

- **Commit unsigned** here (`commit.gpgsign false` is set repo-locally); the user signs at PR
  close.
- **Never commit** `*.gguf`, or the root `*.dll` / `*.exe` (668 MB of CUDA runtime).
- **Benchmark evidence is committed on purpose** — `wiki/benchmarks/results.tsv`, the per-run
  logs, and `needle-prompt.txt` (marked `-text`, since line-ending normalisation would change
  its token count). Flush each run to disk as it completes; GPU-heavy runs here have crashed.
- Differences under **~8%** are noise when a GPU also drives the display.
