# Chapter 14 — The build experiment: compiling the flags, then measuring them

Chapter 12 audited a *proposal* to recompile, and settled most of it with `cuobjdump` and
`grep` — **no build**. This chapter is what happened when we finally did the build.

It is the first chapter in this wiki whose numbers were **not** produced by the shipped
baseline binaries. Everything here comes from two binaries we compiled ourselves, and the
whole design exists to make that comparison mean something.

---

## 14.1 The design: compare a build against a build, never against the shipped baseline

The tempting experiment is "rebuild with the flags, compare against the binaries in the repo
root". That experiment cannot answer anything, because the shipped baseline differs from any
local build in at least four ways at once:

| Axis | Shipped baseline | Any local build |
| --- | --- | --- |
| Source | `fe8156f78` (build 10509) | `c80c4c8a8` (+73 commits, `0.1.2-dev` → `0.2.0-dev`) |
| Host compiler | Clang 20.1.8 | MSVC 19.44.35228 |
| Build shape | `GGML_BACKEND_DL=ON`, `GGML_CPU=OFF`, target `ggml-cuda` only | monolithic, all targets |
| Flags under test | OFF | ON |

Attribute a slowdown to the flags there and you may in fact be measuring 73 commits of source
drift. So the experiment is **two local builds**, identical in everything except the two flags:

```
CONTROL    build-control/bin   base c80c4c8a8, MSVC 14.44.35207, CUDA 13.3.73, default flags
TREATMENT  build-fa/bin        base c80c4c8a8, MSVC 14.44.35207, CUDA 13.3.73,
                               -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_SCHED_MAX_COPIES=1
```

The treatment's base commit is the control's provenance commit, so the two trees differ by a
single `PROVENANCE.md` file and **no compiled source whatsoever**. Same compiler, same
toolkit, same architecture list (`86;120`), same `LLAMA_BUILD_TESTS=OFF
LLAMA_BUILD_EXAMPLES=OFF`. Two variables changed, and both are named.

### The two flags are separable, which is what makes this one experiment instead of a muddle

Changing two things at once is normally a design error. Here it is not, because the source
says the flags act on **disjoint** configurations. From
`ggml/src/ggml-cuda/CMakeLists.txt:115`:

```cmake
if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB SRCS "template-instances/fattn-vec*.cu")   # all 49
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    list(APPEND GGML_SOURCES_CUDA                        # exactly 4
        template-instances/fattn-vec-instance-f16-f16.cu
        template-instances/fattn-vec-instance-q4_0-q4_0.cu
        template-instances/fattn-vec-instance-q8_0-q8_0.cu
        template-instances/fattn-vec-instance-bf16-bf16.cu)
endif()
```

Those four **symmetric** pairs are compiled in *both* builds, and the dispatch path for them
is not guarded by the macro. So:

- On `q8_0/q8_0` and `q4_0/q4_0`, `FA_ALL_QUANTS` is **inert**. Any control-vs-treatment
  difference on those rows belongs to `GGML_SCHED_MAX_COPIES=1`. → **H2**
- On `q8_0/q4_0`, the control has no kernel at all, so that row is dominated by
  `FA_ALL_QUANTS`. → **H1**

One campaign, two separable answers.

---

## 14.2 Verify the flags took, before believing any number

A mistyped `-D` is only a CMake **warning**: it compiles with the default and exits 0. So the
flags get verified structurally first, and none of these checks need a benchmark.

**1. CMake did not ignore anything.** The build script greps for
`Manually-specified variables were not used by the project`. It did not fire, and the cache
agrees:

```
build-control/CMakeCache.txt : GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF   GGML_SCHED_MAX_COPIES:STRING=4
build-fa/CMakeCache.txt      : GGML_CUDA_FA_ALL_QUANTS:BOOL=ON    GGML_SCHED_MAX_COPIES:STRING=1
```

**2. The compile really did widen.** Counting the objects each build produced is a harder fact
than reading the flag back:

| | `fattn-vec*.obj` | `ggml-cuda.dll` |
| --- | --- | --- |
| CONTROL | **4** | 84.2 MB |
| TREATMENT | **49** | 116.2 MB |

**3. The architecture list took.** `cuobjdump --list-elf` / `--list-ptx`:

| Build | Arch list passed | SASS (ELF) | PTX |
| --- | --- | --- | --- |
| shipped baseline | (llama.cpp default) | `sm_86`, `sm_89`, `sm_120a`, `sm_121a` — ×141 each | `sm_75`, `sm_80`, `sm_90` — ×141 each |
| CONTROL | `86;120` | `sm_86` ×141, `sm_120a` ×141 | `sm_86` ×141, `sm_120a` ×141 |
| TREATMENT | `86;120` | `sm_86` ×186, `sm_120a` ×186 | `sm_86` ×186, `sm_120a` ×186 |

+45 cubins per architecture in the treatment — the 45 extra translation units, present as real
device code for both cards. **Neither GPU JITs**: SASS exists for `sm_86` (3070) and `sm_120a`
(5060 Ti), and the driver always prefers SASS over PTX.

### ⚠️ But the PTX in our two builds is dead weight, and that is our mistake

Read the table again. The baseline's PTX is for `compute_75/80/90` — architectures it has **no**
SASS for. That is PTX doing its job: a JIT fallback for GPUs not on the list.

Our builds' PTX is for `sm_86` and `sm_120a` — *the exact architectures they already have SASS
for*. It can never be used. It is a second, redundant copy of every kernel.

The cause is a missing suffix. We passed `-DCMAKE_CUDA_ARCHITECTURES="86;120"`, and a bare
number means **both** SASS and PTX. The suffixes control this:

| Form | Emits | Use when |
| --- | --- | --- |
| `86` | SASS **+ PTX** for compute_86 | you want a JIT fallback for *newer* GPUs |
| `86-real` | SASS only | **you know exactly which GPUs will run it** |
| `86-virtual` | PTX only | you want JIT and nothing else |

llama.cpp's own default list uses these deliberately —
`75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real 121a-real`
(`ggml/src/ggml-cuda/CMakeLists.txt:31-54`) — SASS for the cards it targets, PTX only for the
compute capabilities it does *not* enumerate. That is why the shipped baseline has no
`sm_86` PTX and ours does.

[`pre-build-plan.md`](pre-build-plan.md) §1 specified `86-real;120a-real` for exactly this
reason, and called the effect correctly: *"Shorter compile, smaller binary. **No runtime gain
expected** — neither card JITs today, so this is a build-cost change only."* **We did not pass
it.** The build script derives its arch list from `nvidia-smi` and emits bare numbers, so the
`-real` intent was lost.

Worse, **the wiki already knew.** [Chapter 12 §"`-real` is real"](12-build-flags-analysis.md)
says in as many words: *"Our current invocation passes `"86;120"` — no suffix — so nvcc emits
both PTX and device code for each"*, and predicts the penalty grows with `FA_ALL_QUANTS`'s 45
extra translation units. It was right on both counts (141 → 186 PTX blobs per architecture).
So this was not an unknown — it was a chapter contradicting an earlier chapter, which is worse
than not having looked. The bad regex is what let the contradiction through unnoticed.

**What this does and does not affect:**

- **Not the measurements.** Nothing in §14.5 changes. The redundant PTX is never executed, so
  every throughput and VRAM number below stands as recorded. The plan predicted no runtime
  effect, and there is none to find.
- **Binary size and compile time only.** Both builds are larger and took longer than necessary.
- The fix, for any future build:

```powershell
.\.claude\skills\building-llamacpp-cuda\scripts\build-llamacpp.ps1 `
  -BuildDir build-fa-real -CudaArch '86-real;120-real' -Label fa-all-quants-real `
  -CMakeExtra '-DGGML_CUDA_FA_ALL_QUANTS=ON','-DGGML_SCHED_MAX_COPIES=1'
```

> **The lesson that generalises:** `cuobjdump --list-ptx` names its entries `sm_86.ptx`, not
> `compute_86.ptx`. An earlier revision of this chapter grepped the output for `compute_` ,
> found nothing, and reported "no PTX" for builds that are in fact half PTX. Two architectures
> of dead device code hid behind one wrong regex. **Check a tool's output format before
> filtering it** — and prefer counting whole records (`\.ptx$`) over matching a substring you
> merely expect to be there.

**4. The decisive one, and it costs seconds.** `llama-fit-params` only reads GGUF headers and
*predicts* memory; it never allocates. Ask all three builds to plan the asymmetric pair:

```powershell
llama-fit-params -m Qwen3.8-27B-Q4_K_M.gguf -c 130048 -fa on -ctk q8_0 -ctv q4_0 -fitp on
```

Estimated memory, MiB (`device, model, context, compute`):

| Build | CUDA0 compute | CUDA1 compute | **Host compute** |
| --- | --- | --- | --- |
| shipped baseline | 403 | 1114 | **14507** |
| CONTROL | 403 | 1114 | **14507** |
| TREATMENT | 715 | 715 | **147** |

Two things to read here. First, **the control reproduces the baseline exactly** — same 14507
MiB — which is the evidence that the control is a valid stand-in and that the 73 commits of
source drift changed nothing about this behaviour. Second, the treatment plans **147 MiB** of
host compute instead of 14507: a 98.7× reduction, and the GPU-side buffers become symmetric
at 715 MiB each. That is attention returning to the GPU, visible before a single token is
generated.

---

## 14.3 The precondition: the sysmem-fallback policy, verified rather than assumed

`wiki/nvidia-sysmem-fallback-paths.md` warns that the NVIDIA "Prefer No Sysmem Fallback"
setting is keyed on the **full executable path**, so a new build directory is a new path that
starts uncovered — and that until it is covered, a slow result and a silently spilling one are
indistinguishable.

The driver profile database (`C:\ProgramData\NVIDIA Corporation\Drs\nvdrsdb0.bin`) contains
per-program entries for exactly six executables, all of them in the baseline root, and **none**
for `build-control\bin` or `build-fa\bin`. That looks like a blocker. It was not — because the
question "is the policy in effect for this binary?" is answerable by experiment rather than by
reading a config.

The probe: demand something that **cannot** fit, and see whether the driver refuses or
absorbs it. `-ts 100,0` forces 15.7 GiB of weights plus a 130k KV cache entirely onto the
8 GiB RTX 3070.

| Build | Result | Wall |
| --- | --- | --- |
| TREATMENT | `cudaMalloc failed: out of memory` → clean load failure | 2.6 s |
| CONTROL | `cudaMalloc failed: out of memory` → clean load failure | 3.6 s |

```
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 15088.32 MiB on device 0:
  cudaMalloc failed: out of memory
E alloc_tensor_range: failed to allocate CUDA0 buffer of size 15821250560
E llama_model_load: error loading model: unable to allocate CUDA0 buffer
```

With the fallback **enabled**, that 15 GiB request would have *succeeded* — this box has
63.9 GiB of system RAM, so there was ample room to spill into — and the run would have
completed slowly. It refused instead, in under four seconds, for both builds. **The policy is
active for both new paths**, so the throughput numbers below are not measuring a spill.

> The general lesson is worth more than the specific answer: a per-executable driver setting
> can be *verified from outside* by deliberately over-committing and checking whether the
> failure is loud. That is cheaper and more trustworthy than auditing a GUI.

---

## 14.4 A correction to the control build's own provenance record

`llama.cpp/PROVENANCE.md` on branch `build/20260822-165302-control` reports:

```
| CUDA toolkit | Cuda compilation tools, release 13.1, V13.1.80 |
```

**That line is wrong.** The control build used CUDA 13.3. The authoritative record is the
cache the compiler actually consumed:

```
build-control/CMakeCache.txt:
  CMAKE_CUDA_COMPILER:FILEPATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe
```

The cause: this machine's persisted user environment pointed `CUDA_PATH` and `PATH` at v13.1
while v13.3 was installed, and the manifest was generated by a revision of the build script
that read `nvcc --version` off **PATH** rather than off the toolkit it had just pinned. The
compile was pinned correctly; only the record of it was not. The treatment build's manifest,
written by the current script, reports `release 13.3, V13.3.73` correctly.

The pushed provenance commit has deliberately **not** been rewritten — annotating a wrong
record is safer than editing history that a branch name already points at. Two follow-ups came
out of this instead:

- The machine's user-scope `CUDA_PATH`, `CUDA_HOME` and `PATH` now point at v13.3, with
  `v13.3\bin` and `v13.3\bin\x64` ahead of everything else. Machine scope already said 13.3;
  it was the **user**-scope override that was winning. (`v12.5\bin` remains further down the
  PATH and was left alone.)
- Because `bin\x64` is now on PATH permanently, the "a freshly built exe exits instantly with
  no output at all" trap can no longer be triggered by a missing CUDA runtime.

> **Read the cache, not the manifest.** A provenance file is a claim; `CMakeCache.txt` is what
> the compiler was handed. When they disagree, the cache wins.

---

## 14.5 The campaign

Twelve runs: two builds × three KV configurations × two repetitions, all at the same operating
point. Rep 1 of everything ran before rep 2 of anything, so desktop-VRAM drift over the
campaign hits both builds rather than only the later one.

```
llama-batched-bench -c 130048 -npp 8192 -ntg 128 -npl 1
                    -ub 512 -ts 21,44 -fa on -ngl 999 -fit off
model: Qwen3.8-27B-Q4_K_M.gguf (15.7 GiB)
```

Raw rows: [`benchmarks/build-experiment/results.tsv`](benchmarks/build-experiment/results.tsv),
full logs beside them. They are deliberately **not** in `benchmarks/results.tsv` — that file's
74 rows were all produced by the shipped baseline binaries, and mixing self-compiled rows into
it would make two populations look like one.

### Prefill (t/s), mean of two runs

| KV cache | CONTROL | TREATMENT | Change |
| --- | --- | --- | --- |
| `q8_0`/`q8_0` — symmetric | 1219.89 | 1178.69 | −3.4% |
| `q4_0`/`q4_0` — symmetric | 1218.02 | 1178.46 | −3.2% |
| **`q8_0`/`q4_0` — asymmetric** | **38.87** | **1187.25** | **×30.5** |

### Generation (t/s) and peak VRAM (MiB, both cards summed)

| KV cache | CONTROL tg | TREATMENT tg | CONTROL VRAM | TREATMENT VRAM | ΔVRAM |
| --- | --- | --- | --- | --- | --- |
| `q8_0`/`q8_0` | 22.02 | 21.97 | 23202 | 22357 | **−845** |
| `q4_0`/`q4_0` | 21.94 | 21.98 | 21170 | 20324 | **−846** |
| `q8_0`/`q4_0` | **6.16** | **21.99** | 21384 | 21340 | −44 |

Reproducibility was excellent: the control's two runs of each symmetric config landed within
1.7 t/s of each other, and its peak VRAM within 4 MiB.

---

## 14.6 H1 — `GGML_CUDA_FA_ALL_QUANTS`: confirmed, and it is not a small effect

**Asymmetric `-ctk q8_0 -ctv q4_0` goes from 38.87 to 1187.25 t/s prefill — ×30.5 — and from
6.16 to 21.99 t/s generation.** Wall clock for the same 8192-token prefill plus 128 generated
tokens fell from ~246 s to 23 s.

The number that matters most, though, is not the ratio. It is this:

> **On the treatment build the asymmetric pair is no slower than the symmetric ones.**
> 1187.25 t/s against 1178.69 and 1178.46. The penalty is not reduced — it is *gone*.

So the asymmetric layout is now free, and free changes the recommendation. `q8_0` keys with
`q4_0` values was always the attractive 130k layout on this box — full precision where
attention is most sensitive, half the bytes where it is least — and the only reason not to use
it was that a stock build makes it 30× slower. That reason is now a build flag.

### The bonus: three more KV types stop being traps

This wiki has said, in several places, *"NEVER use `q5_1`, `q5_0`, `q4_1`, or `iq4_nl`"*. That
was correct advice for a stock build and is **not** a property of the types. From
`ggml/src/ggml-cuda/fattn.cu`, `ggml_cuda_fattn_kv_type_supported`:

```c
case GGML_TYPE_Q4_1:
case GGML_TYPE_Q5_0:
case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
    return false;                 // <- the ban, and it is conditional
#endif
case GGML_TYPE_Q4_0:
case GGML_TYPE_Q8_0:
case GGML_TYPE_BF16:
    return true;
```

With the macro defined, those three fall straight through to `return true`. `llama-fit-params`
confirms it — host compute buffer at `-c 130048`, in MiB:

| KV pair | CONTROL | TREATMENT |
| --- | --- | --- |
| `q8_0`/`q8_0` | 529 | **147** |
| `q4_1`/`q4_1` | 11459 | **147** |
| `q5_1`/`q5_1` | 13491 | **147** |
| `q8_0`/`q4_0` | 14507 | **147** |

Every pair reports the same 147 MiB on the treatment build: **there is no CPU attention path
left to fall onto.** The treatment compiles all 49 K/V combinations, so "only four symmetric
pairs work" simply is not true of it.

> The ban was never about the quantisation formats. It was about which kernels someone chose to
> compile — four of forty-nine, by default.

---

## 14.7 H2 — `GGML_SCHED_MAX_COPIES=1`: the hypothesis was wrong, and the flag is still worth having

The hypothesis was: *we measured llama.cpp's automatic non-pipelined fallback as faster, so
compiling pipeline parallelism out should reproduce that speed deterministically instead of by
accident.*

The mechanism is real. `ggml/src/CMakeLists.txt:4` compiles the value in as a define, and
`ggml/src/ggml-backend.cpp:1816` reads it:

```c
sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;
```

With the value set to 1, `n_copies` is 1 whether or not the caller asked for parallel.
(Compile-time only: an environment variable of that name does nothing.) Chapter 18 later found
one nuance: the context's `pipeline_parallel` boolean remains true in this build and still
guards a graph-reuse synchronization, while the runtime fallback clears it. The scheduler copy
count is equivalent; the complete context state is not literally identical.

**But the predicted speed-up does not exist.** On the two symmetric configurations — where
source inspection says `FA_ALL_QUANTS` is inert, so the flag under test is isolated — the
treatment is **3.2–3.4% slower**, not faster. That is inside this box's ±8% noise floor, so it
should not be called a regression; but it is 4 measurements out of 4 in the same direction,
against a control whose own spread was 1.7 t/s. If anything real is there, it is a small cost,
not a gain.

What the flag *does* buy is memory, and that signal is unambiguous:

| | CONTROL | TREATMENT | Saving |
| --- | --- | --- | --- |
| Peak VRAM, `q8_0`/`q8_0` | 23202 | 22357 | **845 MiB** |
| Peak VRAM, `q4_0`/`q4_0` | 21170 | 20324 | **846 MiB** |
| Per GPU | — | — | ~422 MiB on *each* card |
| Host compute buffer, `q8_0`/`q8_0` | 529 | 147 | 382 MiB |

The same number twice, on two unrelated KV configurations, split almost exactly evenly across
two cards of different sizes. That is not noise; that is four sets of split-input copies
becoming one.

> ⚠️ **But only at `-ub 512`.** §14.9 re-ran this at `-ub 1024` and the saving is **gone** —
> both builds peak at exactly 22876 MiB. So this is a `-ub 512` result, not a general property
> of the flag. Read §14.9 before acting on the 845 MiB.

### The correction this forces

Neither build logged `retrying without pipeline parallelism` in any of the twelve runs. So the
earlier observation — "the non-pipelined fallback measured faster" — was never evidence that
non-pipelined execution is fast. That fallback fires when an allocation *fails*, i.e. exactly
when the configuration was too big for VRAM; what was being measured was a run that had
stopped over-committing.

> **Restated with chapter 18's controlled Qwopus data:** the fallback was not materially faster
> because it was non-pipelined. It is a memory recovery. On Qwopus the large gain came from the
> `13,28` tensor split (+12.8%); fallback versus a one-copy build was only 1.6% apart.

Which is still a good trade on this box. 845 MiB is not a rounding error when the display GPU's
idle usage drifts by ~1 GiB between sessions, and it is close to the 1.1 GiB an MTP draft graph
needs.

---

## 14.8 What this settles, and what it does not

**Settled:**

- `GGML_CUDA_FA_ALL_QUANTS=ON` makes asymmetric `q8_0`/`q4_0` **free** on this hardware —
  ×30.5 prefill, ×3.6 generation, and indistinguishable from the symmetric pairs afterwards.
- It also removes the `q5_1` / `q5_0` / `q4_1` ban entirely. Only four of forty-nine kernels
  ship by default; that is the whole reason those types were traps.
- `GGML_SCHED_MAX_COPIES=1` is a **memory** optimisation worth ~845 MiB, not a speed one, and
  the reasoning that predicted speed was mistaken.
- Two self-compiled builds on this toolchain reproduce the baseline where they should: the
  control's asymmetric host-compute estimate is 14507 MiB, identical to the shipped binary's.
  73 commits of source drift changed nothing here.

**Not settled:**

- **Whether a self-compiled binary beats the shipped one.** Still not measured, still confounded
  by source drift and host compiler. Nothing in this chapter compares against the baseline's
  throughput, on purpose.
- **Whether `q8_0`/`q4_0` is *better* than symmetric `q8_0`/`q8_0`.** It is now equally fast and
  saves ~1 GiB, but the quality question is untouched: no needle test, no
  `--kl-divergence-base` run. Fast is not the same as good, and this wiki has no reasoning or
  code-edit quality evidence for *any* quantised KV layout.
- **Whether `q5_1` KV is worth using** now that it runs. It is unbanned, not recommended.
- **The ~3% symmetric prefill cost.** Below the noise floor, consistent in sign, unexplained.
  Worth a repeat-heavy run if it ever matters; not worth acting on now.

### Reproducing this

```powershell
# both builds, via the building-llamacpp-cuda skill
.\.claude\skills\building-llamacpp-cuda\scripts\build-llamacpp.ps1 `
  -BuildDir build-control -Label control `
  -CMakeExtra '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF'

.\.claude\skills\building-llamacpp-cuda\scripts\build-llamacpp.ps1 `
  -BuildDir build-fa -Label fa-all-quants `
  -CMakeExtra '-DGGML_CUDA_FA_ALL_QUANTS=ON','-DGGML_SCHED_MAX_COPIES=1', `
              '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF'
```

The treatment took **10.4 min**. `FA_ALL_QUANTS` was expected to dominate the compile and did
not, so the flag is far cheaper to try than this wiki previously assumed.

Then, per build, from the binary's own `bin/`:

```powershell
$bin = 'S:\OsDevelop\llamacpp\llama.cpp\build-fa\bin'
Set-Location $bin
. 'S:\OsDevelop\llamacpp\.claude\skills\tuning-llamacpp-configs\scripts\bench-harness.ps1' `
    -Model <model.gguf> -LlamaDir $bin -OutDir 'S:\OsDevelop\llamacpp\wiki\benchmarks\build-experiment'
Probe 'fa-q8q4' @('-ub','512','-ts','21,44','-ctk','q8_0','-ctv','q4_0',
                  '-fa','on','-ngl','999','-fit','off') -Ctx 130048 -Npp '8192'
```

`-LlamaDir` selects which build is measured, and the harness now pins the child process's
working directory to it. Without that, ggml resolves backend DLLs against the *caller's* cwd
and can load one build's `ggml-cpu-haswell.dll` into another build's run.

---

## 14.9 So did any of this make anything *faster*? No.

§14.6 and §14.7 answer the two hypotheses as posed, but neither answers the question a user
actually cares about: **is my setup faster than it was?** It is not, and it is worth being
blunt about why, because a ×30.5 in a table is easy to misread as a speed-up.

The ×30.5 is a *penalty removal on a configuration nobody was running*. `q8_0`/`q4_0` at
38 t/s was not a slow option anyone chose — §0 told readers never to use it. Removing a
penalty from an unused config gains you nothing until you switch to that config, and switching
to it is a memory decision, not a speed one.

### The freed memory does not buy speed either

The one remaining route to a real gain: `GGML_SCHED_MAX_COPIES=1` frees ~845 MiB at `-ub 512`,
and a bigger `-ub` is historically worth real prefill throughput. Does the headroom afford
`-ub 1024` at full 130k? Measured (single runs, `-ts 21,44`, `-c 130048`, `npp 8192`):

| Build | KV | `-ub` | Prefill t/s | Peak VRAM |
| --- | --- | --- | --- | --- |
| CONTROL | `q8_0`/`q8_0` | 512 | 1219.89 | 23202 |
| CONTROL | `q8_0`/`q8_0` | **1024** | **1257.91** | 22876 |
| TREATMENT | `q8_0`/`q8_0` | **1024** | 1255.73 | 22876 |
| TREATMENT | `q8_0`/`q4_0` | **1024** | 1255.55 | **21860** |
| TREATMENT | `q8_0`/`q4_0` | 2048 | 1128.67 | 23153 |

Three things fall out, and two of them contradict expectations set earlier in this chapter.

**1. The control did not need the freed memory.** It runs `-ub 1024` at full 130k perfectly
well — 22876 MiB, comfortably inside 24503 — and gains 3.1% doing so. There was no locked door
for the 845 MiB to open. This retires the hope recorded in §7.5, which expected
`SCHED_MAX_COPIES=1` to "free enough to run `q8_0` KV at `-ub 1024` at full 130k". It was
already possible.

**2. At `-ub 1024` the two builds are indistinguishable — in speed *and* in memory.** 1257.91
vs 1255.73 t/s is 0.17%. Peak VRAM is *identical*, 22876 both. So the 845 MiB saving of §14.7
and the 3.3% slowdown of §14.7 are **both** artefacts of `-ub 512`, and both vanish one
notch up the ubatch curve. The mechanism (`n_copies` 4 → 1) is real and the source is
unambiguous, but its observable effect at this context size is not the general constant §14.7
implied. Caveat: these are single runs against the campaign's two-run means.

**3. `-ub 2048` falls off**, 1128.67 t/s — matching the shape of the baseline ubatch curves in
`benchmarks/results.tsv`, where this model peaks at 1024 and regresses at 2048. Nothing about
the new build changes that shape.

### The honest bottom line

Best configuration each build can actually reach, at matched key precision:

| | CONTROL best | TREATMENT best | Difference |
| --- | --- | --- | --- |
| Config | `q8_0`/`q8_0` @ `-ub 1024` | `q8_0`/`q4_0` @ `-ub 1024` | — |
| Prefill | 1257.91 t/s | 1255.55 t/s | **−0.2% (noise)** |
| Generation | 21.87 t/s | 21.93 t/s | unchanged |
| Peak VRAM | 22876 MiB | **21860 MiB** | **−1016 MiB** |

> **The entire return on this build is ~1 GiB of VRAM at identical throughput and identical
> key precision.** Not one token per second.

And note *which* flag delivers it. The 1016 MiB is `q4_0` values instead of `q8_0` values —
the KV cache being literally smaller, which is arithmetic and does not evaporate at a different
ubatch. That is `GGML_CUDA_FA_ALL_QUANTS` doing the work. `GGML_SCHED_MAX_COPIES=1`, on this
evidence, contributes nothing at the operating point you would actually choose.

### Is ~1 GiB worth a rebuild?

On this box, plausibly yes — but for headroom, not speed:

- The display GPU's idle usage has been observed drifting 703 → 1844 MiB between sessions. A
  configuration with 1 GiB of slack survives that; one without it starts spilling silently.
- It is close to the ~1.1 GiB an MTP draft graph needs (§10.13), which is the one thing on the
  open list that *could* convert into real generation throughput.
- It buys `q8_0` keys at roughly the memory price of `q4_0` keys — a quality-per-byte
  improvement, if the quality holds. **Which is still unmeasured.**

If what you want is prefill throughput, the lesson is duller and cheaper: **set `-ub 1024`.**
That is worth 3.1% on the binaries you already have, needs no compiler, and this chapter would
have found it sooner had the campaign not fixed `-ub 512` throughout.
