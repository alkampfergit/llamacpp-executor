# Chapter 12 — Auditing a third-party *build* report

Chapter 11 audited a report about which **file** to download. This one audits a report about
how to **compile** — a set of `cmake` flags proposed for this exact machine (RTX 3070, SM 86,
device 0; RTX 5060 Ti, SM 120, device 1) with the promise that recompiling will make
Qwen3.8-27B faster.

It is a better report than chapter 11's. Nothing in it will destroy a load. But it has the
same shape of weakness, and this chapter exists to name it precisely:

> **A build flag is not advice until you know its default.** Roughly half of what this report
> recommends is *already what happens*. That is not harmless — a command line full of no-ops
> teaches you that you are in control of things you are not, and it hides the two flags that
> actually change the binary.

The verdicts used below:

| Verdict | Means |
| --- | --- |
| **ALREADY FOUND** | We established it independently. The chapter says where |
| **NEW** | Genuinely adds something this wiki did not have |
| **CONTRADICTED** | A measurement or the checked-out source says otherwise |
| **UNVERIFIABLE** | Neither we nor the report has evidence. Say so and stop |

Everything new here came from three cheap, local checks, **no build**:

- `cuobjdump --list-elf` / `--list-ptx` against the baseline `ggml-cuda.dll` — which asks the
  shipped binary what device code it actually contains, instead of reasoning about it.
- `grep` over the submodule at `llama.cpp/` (commit `e85caa81e`) for every option name and its
  default.
- Reading upstream's own `.github/workflows/release.yml`, which is the ground truth for "what
  the official Windows CUDA build does".

> ⚠️ **Version caveat, stated once.** The measurement baseline in the repository root is build
> `10509`, commit `fe8156f78`. The submodule is at `e85caa81e`, which is newer. Every source
> quotation below is from `e85caa81e`. Two findings depend on the newer tree and are flagged
> where they appear (§12.8 and §12.9).

---

## 12.1 The classification

Two of the report's items bundle two independent sub-claims, so they are split rather than
averaged.

| # | Claim | Verdict | Where |
| --- | --- | --- | --- |
| 1 | Recompiling for exact SM targets buys little; the official binary already ships device code for 86 and 120 | **NEW** | §12.2 |
| 2 | `GGML_CUDA_FA_ALL_QUANTS` defaults OFF and the official Windows release does not enable it | ALREADY FOUND | [§0 KEY FINDINGS](00-KEY-FINDINGS.md), [§6.3](06-results.md), [§11.3](11-quant-selection-qwen38.md) |
| 3 | Cited issue: quantized KV + FA went 96 → 2361 t/s PP on Blackwell after enabling all FA quants | ALREADY FOUND | [§6.3](06-results.md), [§11.3](11-quant-selection-qwen38.md); citation checked in §12.4 |
| 4a | Use the `-real` suffix — SASS only, no PTX | **NEW** | §12.3 |
| 4b | Use `120a` — Blackwell architecture-specific, needed for FP4 tensor-core instructions | ALREADY FOUND | [traps §6](../.claude/skills/building-llamacpp-cuda/references/windows-toolchain-traps.md); it is also automatic — §12.3 |
| 5 | `-DGGML_NATIVE=OFF`; the official Windows CUDA build uses CUDA 13.3 with `GGML_NATIVE=OFF` | **NEW** (fact) | §12.6 |
| 6 | `-DGGML_CUDA_FA=ON` as a distinct option | **NEW** (inert) | §12.7 |
| 7 | Do **not** set `GGML_CUDA_FORCE_MMQ=ON` | **NEW** (right advice, wrong mechanism) | §12.5 |
| 8 | Do **not** set `GGML_CUDA_FORCE_CUBLAS=ON` | **NEW** | §12.5 |
| 9 | Keep `GGML_CUDA_GRAPHS=ON` | **NEW** (inert) | §12.7 |
| 10 | `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON` to cut build time | **NEW** | §12.8 |
| 11 | Output at `build\bin\Release\llama-server.exe` "depending on generator" | ALREADY FOUND | [SKILL §4](../.claude/skills/building-llamacpp-cuda/SKILL.md), [traps §5](../.claude/skills/building-llamacpp-cuda/references/windows-toolchain-traps.md) — with one correction, §12.8 |
| 12a | CUDA peer access has limitations on Windows | **CONTRADICTED** | §12.9 |
| 12b | The 3070 can therefore bottleneck via PCIe | **UNVERIFIABLE** | §12.9 |
| 13 | Benchmark splits `70,30` / `75,25` / `80,20`; fill the 5060 Ti aggressively | **CONTRADICTED** | §12.10 |
| 14 | Only 16 conventional attention layers ⇒ generation is bandwidth-bound; compilation cannot fix it | ALREADY FOUND | [§10.2](10-results-qwen38-27b.md), [§10.3](10-results-qwen38-27b.md) |
| 15 | Impact table: exact-SM build small; FA_ALL_QUANTS potentially huge PP; avoiding CPU/shared-memory fallback enormous | ALREADY FOUND | [§0 KEY FINDINGS](00-KEY-FINDINGS.md) ranks these identically |
| 16a | Final command still uses `--cache-type-k q8_0 --cache-type-v q4_0` | ALREADY FOUND | [§11.3](11-quant-selection-qwen38.md) — and here it is *coherent*, see §12.10 |
| 16b | Final command still uses `--tensor-split 2,1` | **CONTRADICTED** | [§11.4](11-quant-selection-qwen38.md), §12.10 |

**Tally: 5 NEW, 2 CONTRADICTED, 1 UNVERIFIABLE, 8 already in this wiki.** Of the five NEW, one
is worth putting on a command line, one is worth putting on a command line *because* of the
first, and three are worth knowing precisely so that you do **not** type them.

---

## 12.2 What is actually inside the shipped DLL (claim 1 — NEW)

This is the report's best contribution, and it is checkable in four seconds without compiling
anything. `cuobjdump` will tell you exactly which architectures a CUDA binary carries:

```powershell
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cuobjdump.exe" `
    --list-elf S:\OsDevelop\llamacpp\ggml-cuda.dll
```

Run against the **baseline** `ggml-cuda.dll` (135.3 MB, build `10509`):

| Kind | Architectures present | Count each |
| --- | --- | --- |
| **SASS** (`--list-elf`) | **`sm_86`**, `sm_89`, **`sm_120a`**, `sm_121a` | 141 cubins |
| PTX (`--list-ptx`) | `compute_75`, `compute_80`, `compute_90` | — |

**Both of this machine's cards have native SASS in the shipped binary. Neither one JITs.**
Claim 1 is confirmed, and by the most direct evidence available.

That is not a coincidence. `ggml/src/ggml-cuda/CMakeLists.txt` picks the list when
`CMAKE_CUDA_ARCHITECTURES` is not defined, and upstream's release job does not define it:

```
75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real 121a-real     # CUDA >= 12.8
```

Line for line, that is the table above. The x64 Windows CUDA release is built with CUDA 13.3
(`release.yml`), which is `>= 12.8`, so `120a-real` is in.

> **The lesson, and it is a correction to how this wiki has been reading its own baseline.**
> [traps §6](../.claude/skills/building-llamacpp-cuda/references/windows-toolchain-traps.md)
> warns loudly that a wrong `-DCMAKE_CUDA_ARCHITECTURES` silently costs you JIT-from-PTX. That
> warning is correct **and it only ever applied to builds we made ourselves.** The `86;89`
> mistake it documents was ours. The prebuilt binaries every number in this wiki was measured
> with never had that problem. **Do not carry a warning about your own build process over to a
> vendor's binary without checking the binary.**

Practical consequence: **"recompile for my exact GPUs" is not a reason to rebuild.** There are
two good reasons to rebuild on this machine — `GGML_CUDA_FA_ALL_QUANTS` and
`GGML_SCHED_MAX_COPIES` — and architecture is not a third one.

There is one *non*-performance payoff, though, and it sets up the next section: **half the
device code in that DLL is dead weight here.** `sm_89` (Ada) and `sm_121a` match nothing in
this box. 141 cubins × 2 useless architectures is most of why the file is 135 MB.

---

## 12.3 `-real`, and why it matters more than it looks (claim 4a — NEW; 4b — already found)

Take the two halves separately, because only one of them does anything.

**`120a` is automatic.** `ggml/src/ggml-cuda/CMakeLists.txt` rewrites it for you,
unconditionally, over both the requested list and CMake's native detection:

```cmake
# Replace any plain 12X CUDA architectures with their "architecture-specific" equivalents 12Xa.
# Notably the Blackwell FP4 tensor core instructions are not forwards compatible and
# therefore need 12Xa.
if (ARCH MATCHES "^12[0-9](-real|-virtual)?$")
    string(REGEX REPLACE "^(12[0-9])((-real|-virtual)?)$" "\\1a\\2" FIXED_ARCH ${ARCH})
    message(STATUS "Replacing ${ARCH} in ${ARCHS} with ${FIXED_ARCH}")
```

So our `-DCMAKE_CUDA_ARCHITECTURES="86;120"` **already becomes `86;120a`**. This is the code
behind the `Replacing 120-real ... with 120a-real` line we saw in a build log and recorded in
traps §6 without understanding it. The report's reasoning about FP4 forward-incompatibility is
right; typing `120a` yourself changes nothing. Marked ALREADY FOUND, not because we knew the
mechanism, but because we had already recorded its output and the value we pass is already
correct.

**`-real` is real, and it is the one arch-flag change worth making.** From the same file's own
comment block:

```
XX-virtual == compile CUDA code as PTX, do JIT compilation to binary code on first run
XX-real    == compile CUDA code as device code for this specific architecture
no suffix  == compile as both PTX and device code
```

Our current invocation passes `"86;120"` — **no suffix** — so nvcc emits *both* SASS and PTX
for each. `"86-real;120a-real"` emits SASS only. The measurable effects:

| | Ours today (`86;120`) | Report's (`86-real;120a-real`) |
| --- | --- | --- |
| Architectures compiled | 2 | 2 |
| PTX also emitted | yes, for both | no |
| Binary size | larger | smaller |
| JIT fallback for a *future* GPU | yes | **none** |

On its own that is a rounding error. **It stops being a rounding error the moment you also
pass `GGML_CUDA_FA_ALL_QUANTS=ON`,** which is exactly what we are about to do:

```
ggml/src/ggml-cuda/template-instances/fattn-vec*.cu     49 files
of which compiled today                                  4 files
```

`GGML_CUDA_FA_ALL_QUANTS=ON` globs all 49 instead of naming 4 — **45 extra CUDA translation
units**, each of which nvcc must run through the full front end and then code-generate once per
architecture *and* once more for PTX. Dropping the PTX pass is a fraction of the build you no
longer pay 45 times over.

> **Be honest about what this is.** We have not built either way, so the compile-time saving is
> reasoned from the CMake and the file count, not measured. The *direction* is certain (strictly
> less code generation); the magnitude is not.
>
> **✅ Update — [chapter 14](14-build-experiment.md) built it, and this section was right.**
> Both builds were made with the bare `"86;120"` this section warns about, and `cuobjdump`
> confirms the predicted redundant PTX: **141 blobs per architecture** in the control and
> **186** with `GGML_CUDA_FA_ALL_QUANTS=ON` — i.e. the penalty does scale with the extra 45
> translation units, exactly as argued above. What was *not* confirmed is the compile-time
> story: the `FA_ALL_QUANTS` build took only **10.4 min** including the wasted PTX passes, so
> the "long pole" framing was too pessimistic. `-real` remains free on this machine and is now
> the documented default. What is certain and useful is the cost side:
> `-real` costs you nothing on this machine, because both cards have native SASS and neither
> will ever need to JIT. If you put a third, newer GPU in this box, `-real` is the flag that
> makes it fall over — and that is a fine trade to make explicitly.

---

## 12.4 The citation, checked (byproduct — not a report claim)

The report cites a llama.cpp issue reporting **96 → 2361 t/s** prompt processing after enabling
all quantized FA kernels. It checks out exactly:
[issue #24485](https://github.com/ggml-org/llama.cpp/issues/24485), opened 2026-06-11 — *"[FR]
Make GGML_CUDA_FA_ALL_QUANTS the default or add runtime warning for missing quantized FA
kernels"*. RTX 5070 Ti Laptop (**sm_120**, the same architecture as our 5060 Ti), Gemma 4 12B
QAT, `q8_0`/`q4_0`, CUDA 13.2, "~25x prefill improvement with FA_ALL_QUANTS=ON (96 -> 2,361
tok/s)". Compare our own 11.6× at identical `-ub 512` ([§11.3](11-quant-selection-qwen38.md)).

The mechanism was already ours. **Two things fall out of the citation that are genuinely new:**

- It is an independent sm_120 confirmation of the *throughput* figure that
  [§11.3](11-quant-selection-qwen38.md) explicitly admits we never measured on the dense model
  — we measured only the allocation signature there (14507 MiB of host compute buffer).
- **The issue was closed as "not planned", labelled stale.** So `GGML_CUDA_FA_ALL_QUANTS` will
  keep defaulting OFF, and **no runtime warning is coming**. This is not a bug awaiting a fix
  that a version bump will bring us. On this machine, asymmetric KV is a permanent
  build-it-yourself item.

---

## 12.5 Right advice, wrong mechanism: `FORCE_MMQ` vs `FORCE_CUBLAS` (claims 7, 8 — NEW)

Both options exist, both default OFF (`ggml/CMakeLists.txt:201-202`), and the report is right
that you should leave both alone. But it treats them as a symmetric pair of "don't touch these",
and in `ggml/src/ggml-cuda/mmq.cu` they are not remotely symmetric.

`ggml_cuda_should_use_mmq()` reads, in order:

```c
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;                                   // <-- first thing in the function
#endif
    ... type support check ...
    ... 48 KiB shared-memory check ...
    if (turing_mma_available(cc)) {
        return true;                                // <-- BEFORE the FORCE_MMQ check
    }
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) { return false; }
#ifdef GGML_CUDA_FORCE_MMQ
    return true;                                    // <-- unreachable on Turing+ hardware
#endif
```

| Option | Effect on **this** hardware | Why |
| --- | --- | --- |
| `GGML_CUDA_FORCE_MMQ=ON` | **essentially a no-op** | `turing_mma_available()` is true for both SM 86 and SM 120, so the function has already returned `true` before the `FORCE_MMQ` branch is reached |
| `GGML_CUDA_FORCE_CUBLAS=ON` | **disables MMQ everywhere** | it is the first statement in the function and short-circuits every type, every batch size, both cards |

So the report's stated reason for avoiding `FORCE_MMQ` — "forcing it can slow large-batch
processing, though it may cut VRAM" — **does not apply to a Turing-or-newer card.** That
reasoning is aimed at Pascal/Volta-class hardware, and there is a hint of its real audience in
`ggml-cuda.cu`, which suggests `FORCE_MMQ` only for *Turing cards without tensor cores*
(GTX 16xx). It is generic advice that happens to land on the right answer here for a reason the
report does not give.

`FORCE_CUBLAS`, by contrast, is a genuinely bad idea and the report understates it. With MMQ
off, every quantized matrix multiply must dequantize its weights to FP16 into a scratch buffer
before cuBLAS can touch them. On a box whose entire tuning story is a 760 MiB headroom at
`-c 130048` ([§10.1](10-results-qwen38-27b.md)), a systematically larger compute buffer is not
a "raises memory use" footnote — it is the difference between a config that loads and one that
does not.

> **The rule: read the order of the returns, not just the option list.** Two flags described
> identically in `--help` can differ by "no effect" versus "changes every matrix multiply in the
> model", and the only thing that tells you which is which is where the `return` sits.

Neither flag goes on our command line. `FORCE_MMQ` because it would do nothing; `FORCE_CUBLAS`
because it would do too much.

---

## 12.6 `GGML_NATIVE=OFF` is a CI artefact, not a tuning flag (claim 5 — NEW)

The report's factual claim is true, and worth having on record. From
`.github/workflows/release.yml`, the `windows-cuda` job for `cuda: '13.3', arch: x64`:

```bat
cmake -S . -B build -G "Ninja Multi-Config" ^
  -DGGML_BACKEND_DL=ON ^
  -DGGML_NATIVE=OFF ^
  -DGGML_CPU=OFF ^
  -DGGML_CUDA=ON ^
  -DLLAMA_BUILD_BORINGSSL=ON
cmake --build build --config Release -j %NINJA_JOBS% --target ggml-cuda
```

CUDA 13.3, `GGML_NATIVE=OFF`: confirmed. **But copying it here would be cargo-culting**, and
tracing *why* it is there is instructive:

**Reason it exists in CI.** In `ggml/src/ggml-cuda/CMakeLists.txt` the architecture default is
chosen like this:

```cmake
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" ...)
        set(CMAKE_CUDA_ARCHITECTURES "native")
```

A GitHub runner has no GPU, so `native` would produce garbage. `GGML_NATIVE=OFF` is what forces
the explicit `75-virtual;…;121a-real` list of §12.2. **It is a flag whose job is to make a
GPU-less machine build for all GPUs.**

**Why it does nothing for us.** That whole branch is inside `if (NOT DEFINED
CMAKE_CUDA_ARCHITECTURES)`, and our script always defines it. So `GGML_NATIVE` cannot affect our
CUDA architecture selection at all.

**What is left of it.** Only the CPU backend's instruction set. On MSVC:

| | `GGML_NATIVE=ON` (our default) | `GGML_NATIVE=OFF` |
| --- | --- | --- |
| Path taken | `FindSIMD.cmake` probes the host | `INS_ENB=ON`, explicit ISA options |
| Result on a Ryzen 5900X (Zen 3) | `/arch:AVX2` (no AVX-512 on Zen 3) | `/arch:AVX2` |
| `__BMI2__` defined | **no** (`GGML_BMI2` stays at its OFF default) | **yes** |

Identical `/arch:` level, and `GGML_NATIVE=OFF` incidentally picks up BMI2. So on this CPU the
flag is somewhere between "no effect" and "a marginal improvement to a code path we have
measured to be a disaster" — [`-ncmoe 2` costs 69% of
prefill](00-KEY-FINDINGS.md), and every recommended profile runs `-ngl 999`. **We have no
evidence either way about its effect on our throughput, and a strong reason to expect none.**

> **The lesson: read a vendor's CI as a *distribution* recipe, not a *performance* recipe.**
> `GGML_NATIVE=OFF`, `GGML_CPU=OFF`, `GGML_BACKEND_DL=ON` and `-G "Ninja Multi-Config"` are all
> there so that one artefact runs on every machine a stranger might own. Every one of those is a
> constraint you do not have when you are compiling for the box under your desk.

---

## 12.7 True, and inert: `GGML_CUDA_FA` and `GGML_CUDA_GRAPHS` (claims 6, 9 — NEW)

Both options exist, and both are already what the report asks for. This is the "half the report
is no-ops" part, stated concretely so nobody types them:

| Option | Where | Default | `=ON` does |
| --- | --- | --- | --- |
| `GGML_CUDA_FA` | `ggml/CMakeLists.txt:207` | **ON** | nothing |
| `GGML_CUDA_GRAPHS` | `ggml/CMakeLists.txt:209` | **ON** for a standalone llama.cpp build — the root `CMakeLists.txt:170` sets `GGML_CUDA_GRAPHS_DEFAULT ON` | nothing |

`GGML_CUDA_FA` is worth one more sentence, because its *inverse* is the interesting direction.
`GGML_CUDA_FA=OFF` compiles `GGML_CUDA_NO_FA`, i.e. no flash-attention kernels at all — the
build-time twin of the runtime `-fa off` we already measured at **+9 GiB of compute buffers**
([§0 KEY FINDINGS](00-KEY-FINDINGS.md)). Never set it. Note also that it is *not* the flag that
governs which K/V pairs exist; that is `GGML_CUDA_FA_ALL_QUANTS`, and confusing the two is easy
from the names alone.

---

## 12.8 Trimming the target list — and a new trap in the newer tree (claims 10, 11 — NEW / already found)

All three options in claim 10 exist and all three default to `${LLAMA_STANDALONE}` = ON
(`CMakeLists.txt:132-135`), so `TESTS=OFF EXAMPLES=OFF` really does remove work. This is a
better idea than our current `--target llama-server`, because it keeps the measurement tools.
But three caveats matter more than the flags do:

1. **Do not touch `LLAMA_BUILD_TOOLS`.** The report does not mention it, and it is the one that
   carries everything we measure with. `tools/CMakeLists.txt` builds `llama-bench`,
   `batched-bench`, `fit-params`, `imatrix`, `quantize` and `perplexity`; `LLAMA_BUILD_SERVER`
   additionally gates `tools/cli`, `tools/server` and `tools/ui`. **`llama-cli` disappears if
   you turn the server off** — a non-obvious coupling in the current tree.
2. **`LLAMA_BUILD_EXAMPLES=OFF` is safe** precisely because nothing we use lives in `examples/`
   any more. Verified against `tools/CMakeLists.txt`.
3. **The saving is modest.** Everything above is ordinary C++. The wall clock is
   `ggml-cuda` — 141 cubins per architecture in §12.2's dump, about to become considerably more
   with `FA_ALL_QUANTS`. Trimming tests and examples trims the part that was never the problem.

**A hazard the report could not have known about, because it postdates our baseline.** In
`e85caa81e`, `LLAMA_BUILD_SERVER=ON` pulls in `tools/ui`, whose asset provisioning
(`scripts/ui-assets.cmake`) tries, in order: an existing `dist/`, an `npm ci && npm run build`,
then a download from a Hugging Face bucket. It degrades to
`message(WARNING "UI: no assets available - building without an embedded UI")` rather than
failing, so it cannot break the build — but it will spend time on npm or the network for a Web
UI we have never used, since every measurement in this wiki drives `llama-server` over raw
HTTP. **`-DLLAMA_BUILD_UI=OFF` skips it entirely**, and is a better build-time flag than either
of the two the report suggests.

**Claim 11, and a small correction to our own notes.** Our
[traps §5](../.claude/skills/building-llamacpp-cuda/references/windows-toolchain-traps.md) says
`build\bin\Release\` "only appears with the Visual Studio generator". Not quite: upstream's
release job uses `-G "Ninja Multi-Config"` and packs from `.\build\bin\Release\ggml-cuda.dll`.
The accurate rule is **multi-config generator ⇒ `bin\Release\`; single-config ⇒ flat `bin\`**.
Our recipe uses plain `-G Ninja`, which is single-config, so
[SKILL §4](../.claude/skills/building-llamacpp-cuda/SKILL.md) is right about *our* build:
binaries land in `build\bin\`. The report's hedge — "depending on generator" — is correct and
useless, because it does not say which generator you will end up with.

---

## 12.9 Peer access is not this machine's bottleneck (claims 12a — CONTRADICTED, 12b — UNVERIFIABLE)

The report warns that Windows limits CUDA peer access, that this is not NVLink-equivalent, and
that the 3070 can therefore bottleneck over PCIe. Grep the tree for where peer access is
actually turned on:

```c
// ggml/src/ggml-cuda/ggml-cuda.cu, in ggml_cuda_init()
if (getenv("GGML_CUDA_P2P") != nullptr) {
    ...
    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
}
```

and, for VMM pool allocations:

```c
bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
    use_peer_access = true;
#endif
```

**Peer access is opt-in.** It happens only if `GGML_CUDA_P2P` is set in the environment, or if
the build found NCCL. Our builds print `Could NOT find NCCL` — it is in
[traps §7](../.claude/skills/building-llamacpp-cuda/references/windows-toolchain-traps.md) as an
expected warning — and NCCL is not available on Windows. **So llama.cpp is not using P2P on this
machine at all, and a Windows limitation on a facility nobody invokes cannot be a bottleneck.**
That is claim 12a: contradicted, on the checked-out source.

There is a second reason it is the wrong thing to worry about. With `--split-mode layer`, the
only cross-device traffic per token is the activation tensor at the split boundary — kilobytes,
once. Weights never move. `GGML_CUDA_PEER_MAX_BATCH_SIZE` (default 128) governs `-sm row`, which
we do not use.

And what our measurements actually say limits the 3070 is **capacity, not interconnect**:
[§10.6](10-results-qwen38-27b.md) shows `-ts 23,42` failing to allocate an 823 MiB compute
buffer on device 0 while `-ts 22,43` works, and prefill flat at 1208 vs 1219 t/s across splits
that move a whole layer between cards. If PCIe were the limit, moving a layer would move the
number.

**Claim 12b stays UNVERIFIABLE, and this is worth being strict about.** We have never measured
PCIe bandwidth or link utilisation on this box. We can say the peer-access mechanism the report
names is inactive, and that our split sweeps show a memory cliff rather than a transfer cost. We
cannot say "PCIe never bottlenecks the 3070", and we should not pretend to. The one place a
PCIe bottleneck is *definitely* real here is
[sysmem fallback](00-KEY-FINDINGS.md) — but that is a card spilling to host RAM, not two cards
talking to each other, and the report is not describing it.

---

## 12.10 The split numbers, for the third time (claims 13, 16b — CONTRADICTED; 16a — already found)

[§11.4](11-quant-selection-qwen38.md) already refuted `--tensor-split 2,1` on this box. This
report repeats it, and then proposes a benchmark sweep that is worse.

Both cards' identities, from every launch log: **`CUDA0` is the RTX 3070 (8 GiB). `CUDA1` is the
RTX 5060 Ti (16 GiB).** `--tensor-split` is in device order, so the first number is the small
card. Now read the report's own proposal against that:

| Source | Split | Fraction to the **3070** | Verdict |
| --- | --- | --- | --- |
| Measured optimum, [§10.6](10-results-qwen38-27b.md) | `22,43` | 34% | ✅ 7421 / 15175 MiB peak |
| Report's final command | `2,1` | 67% | ❌ [§11.4](11-quant-selection-qwen38.md): asks 13864 MiB of an 8 GiB card |
| Report's sweep | `70,30` | **70%** | ❌ worse than `2,1` |
| Report's sweep | `75,25` | **75%** | ❌ worse still |
| Report's sweep | `80,20` | **80%** | ❌ worst |

Every proposed split is on the far side of a cliff we have already located: `-ts 23,42` — 35% to
the 3070 — **already fails to load**. The sweep starts at double that and goes up.

What makes this more than a repeat of §11.4 is that the report states its own intent correctly
and then writes numbers that do the opposite: it says to *fill the 5060 Ti aggressively and give
the 3070 only what it needs*, which is exactly right, and is exactly what `22,43` does. The
numbers `70,30`/`75,25`/`80,20` implement the reverse of the sentence next to them.

> **The lesson, restated because two independent reports have now made the identical error:**
> **`--tensor-split` is an ordered vector over `CUDA0, CUDA1, …`, and device order is a property
> of the machine, not of the cards.** Most boxes put the big GPU first; this one does not. Read
> your own `using device CUDA{0,1}` lines before copying anybody's ratio, and never copy the
> numbers.

**Claim 16a, in fairness, is the one place this report is more coherent than chapter 11's.** It
recommends `--cache-type-k q8_0 --cache-type-v q4_0` *and* `GGML_CUDA_FA_ALL_QUANTS=ON`. Those
two together are legitimate — that pairing is the whole point of the rebuild, and it is
[§10.13](10-results-qwen38-27b.md)'s top open item. The trap
[§11.3](11-quant-selection-qwen38.md) documents (14507 MiB of host compute buffer, 27× the
symmetric figure) fires only if you run that flag pair against a binary that lacks the kernels
— which is what happens if you paste the command and forget to run the rebuilt binary from
`llama.cpp\build\bin\`. Marked ALREADY FOUND rather than CONTRADICTED, with one condition
attached: **it is still unmeasured for throughput on this machine, in either direction.**

---

## 12.11 Found while verifying: our own rebuild flag does not exist

This is not one of the report's claims. It is our error, it has been in this wiki since chapter
6, and looking up option names to grade someone else's flags is what surfaced it.

Every rebuild recommendation in this repository used to say:

```
-DLLAMA_SCHED_MAX_COPIES=1
```

**There is no such `LLAMA_` option.** The real option is:

```cmake
# ggml/CMakeLists.txt:188
set(GGML_SCHED_MAX_COPIES "4" CACHE STRING "ggml: max input copies for pipeline parallelism")
# ggml/src/CMakeLists.txt:4
add_compile_definitions(GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES})
```

It is **`GGML_`**, not `LLAMA_`. The `llama_option_depr()` table in the root `CMakeLists.txt`
forwards ten legacy `LLAMA_*` names to their `GGML_*` equivalents — `LLAMA_CUBLAS`,
`LLAMA_NATIVE`, `LLAMA_RPC` and so on — and `GGML_SCHED_MAX_COPIES` is **not one of them.**

What would have happened had we run the rebuild as documented: CMake accepts the unknown `-D`,
stores it in the cache, prints `Manually-specified variables were not used by the project:
LLAMA_SCHED_MAX_COPIES` as a *warning*, and compiles with `GGML_SCHED_MAX_COPIES=4`. The build
succeeds. The binary is pipelined exactly as before. And our build script's console filter
(`'^\[\d+/\d+\]|^-- |error|Error|FAILED|fatal'`) does not match the string `CMake Warning`, so
the only notice would have gone to the log file nobody reads on a green build.

> **This is the third instance of the same failure mode in this repository, and by now it has
> earned a name.** [§6.2](06-results.md) credited +13% to an environment variable that is a
> compile-time `#define`. [§11.3](11-quant-selection-qwen38.md) found a KV pair with no compiled
> kernel. Now a `-D` with no option behind it. In all three cases the software accepted the
> input, produced correct output, and reported nothing.
>
> **`CLAUDE.md` says to read the source when a measurement looks strange. Extend it: read the
> source before you write the flag down, too.** Confirming that a knob exists is cheaper than
> any measurement you would take with it — one `grep`, and it also hands you the default, which
> is the thing that decides whether the flag is worth passing at all.

**Twenty-nine occurrences** needed the substitution `LLAMA_SCHED_MAX_COPIES` →
`GGML_SCHED_MAX_COPIES`: **19 in `wiki/`** (chapters 0, 3, 4, 6, 7, 8, 10, 11 and, at the time
of this audit, `wiki/scripts/serve-qwopus-130k.ps1` — that script has since moved to
`.claude/skills/tuning-llamacpp-configs/scripts/serve-qwopus-130k.ps1`) and **10 in
`.claude/skills/`** (both skills, including
`building-llamacpp-cuda`'s SKILL front-matter, its options table, the build script's usage
banner, and `tuning-llamacpp-configs`' `bench-harness.ps1` console hint). **That correction is
deliberately not made
in this chapter**, whose brief was not to edit the other chapters; it is listed here so the fix
is a mechanical pass rather than a rediscovery.

---

## 12.12 The rebuild command line

Given that a rebuild is already planned for the two flags that actually change the binary, here
is what should and should not join them.

**Run this:**

```powershell
.\.claude\skills\building-llamacpp-cuda\scripts\build-llamacpp.ps1 -CudaArch '86-real;120a-real' `
  -CMakeExtra '-DGGML_CUDA_FA_ALL_QUANTS=ON', `
              '-DGGML_SCHED_MAX_COPIES=1', `
              '-DLLAMA_BUILD_TESTS=OFF', `
              '-DLLAMA_BUILD_EXAMPLES=OFF', `
              '-DLLAMA_BUILD_UI=OFF'
```

| Flag | Why it is on the line | Evidence |
| --- | --- | --- |
| `-DGGML_CUDA_FA_ALL_QUANTS=ON` | The reason to rebuild. Unlocks `-ctk q8_0 -ctv q4_0`; without it that pair puts attention on the host | [§6.3](06-results.md), [§11.3](11-quant-selection-qwen38.md), and now issue #24485 |
| `-DGGML_SCHED_MAX_COPIES=1` | The *other* reason to rebuild — **and note the corrected name**. Makes the lean path deterministic instead of a side effect of an allocation failing | [§6.2](06-results.md), §12.11 |
| `-CudaArch '86-real;120a-real'` | Two architectures instead of the four in the shipped DLL, and no PTX pass — across 45 extra FA translation units | §12.2, §12.3 |
| `-DLLAMA_BUILD_TESTS=OFF`, `-DLLAMA_BUILD_EXAMPLES=OFF` | Free, and keeps `llama-bench` / `batched-bench` / `fit-params`, which live under `tools/` | §12.8 |
| `-DLLAMA_BUILD_UI=OFF` | Skips an npm/network step for a Web UI this wiki has never used | §12.8 |

**Leave these off, and know why:**

| Flag | Why not |
| --- | --- |
| `-DGGML_CUDA_FA=ON` | Already ON. Its inverse costs 9 GiB of compute buffer |
| `-DGGML_CUDA_GRAPHS=ON` | Already ON for a standalone llama.cpp build |
| `-DCMAKE_CUDA_ARCHITECTURES` containing a bare `120` | Already rewritten to `120a` by llama.cpp itself. Passing `120a` is neither wrong nor useful |
| `-DGGML_CUDA_FORCE_MMQ=ON` | Unreachable on SM 86 / SM 120 — `turing_mma_available()` returns first |
| `-DGGML_CUDA_FORCE_CUBLAS=ON` | Disables MMQ for every matmul on both cards and inflates compute buffers, against a 760 MiB headroom |
| `-DGGML_NATIVE=OFF` | A CI flag for a GPU-less runner. Cannot affect our arch list, and on Zen 3 lands on the same `/arch:AVX2` |
| `-DLLAMA_BUILD_SERVER=OFF` | Would also delete `llama-cli` in this tree |
| Anything about peer access / P2P | Off by default; `-sm layer` moves one activation tensor per boundary |

**And the honest gaps.** We have no evidence either way on: how much wall clock `-real` plus the
trimmed target list actually saves; whether `GGML_NATIVE` moves any number we care about
(mechanism says no, and the CPU path is one we avoid on purpose); and — the big one — **whether
`-ctk q8_0 -ctv q4_0` on a `FA_ALL_QUANTS` build is actually faster than symmetric `q8_0`.** It
saves 1016 MiB at 130k, which is almost exactly what the MTP draft graph costs, and *that* is
the measurement the rebuild exists to make possible. It is not a measurement the rebuild makes.

> **Verify the rebuild before you benchmark it, in two commands.** Both are cheap and both
> catch a silent no-op of exactly the kind §12.11 describes:
>
> ```powershell
> # 1. architectures actually compiled -- expect sm_86 and sm_120a, and nothing else
> & "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cuobjdump.exe" `
>     --list-elf .\llama.cpp\build\bin\ggml-cuda.dll
>
> # 2. asymmetric KV now has a kernel -- Host compute in the hundreds of MiB, not thousands
> .\llama.cpp\build\bin\llama-fit-params.exe -m <model.gguf> -fitp on `
>     -c 130048 -fa on -ctk q8_0 -ctv q4_0
> ```
>
> If the second still reports five figures of host compute buffer, `FA_ALL_QUANTS` did not take
> — and you have found that out in seconds instead of attributing it to the model.

---

## 12.13 Scorecard

| | |
| --- | --- |
| **What the report adds** | That the shipped binary already carries SASS for both our cards, so "recompile for exact SM" is not a reason to build (§12.2). The `-real` suffix, which is worth real compile time once `FA_ALL_QUANTS` multiplies the translation units by twelve (§12.3). A verified citation for the FA-quants penalty on sm_120 hardware (§12.4). Precise reasons to leave `FORCE_MMQ`, `FORCE_CUBLAS`, `GGML_CUDA_FA` and `GGML_CUDA_GRAPHS` alone (§12.5, §12.7). Target-list trimming that keeps the measurement tools (§12.8) |
| **What it gets wrong** | Peer access as this box's constraint — llama.cpp does not enable P2P unless you ask for it, and our splits show a memory cliff, not a transfer cost (§12.9). Tensor splits of `70,30`–`80,20` and `2,1`, all of them on the dead side of the `23,42` cliff, and all of them the reverse of the report's own stated intent (§12.10) |
| **What was already here** | The `FA_ALL_QUANTS` default and its cost. The `120a` rewrite. `bin\Release\` versus flat `bin\`. The 16-attention-layer architecture and the bandwidth ceiling. The impact ranking, which matches [KEY FINDINGS](00-KEY-FINDINGS.md) row for row |
| **Unverifiable** | Whether PCIe ever bottlenecks the 3070. We have never measured link utilisation and should not claim to have (§12.9) |
| **Our own error, surfaced by the audit** | `-DLLAMA_SCHED_MAX_COPIES=1` is not an option in llama.cpp. Twenty-nine occurrences were corrected to `-DGGML_SCHED_MAX_COPIES=1` (§12.11) |

> **The closing point.** Chapter 11's report reasoned about quality per byte and treated fit as
> a detail. This one reasons about compiler flags and treats defaults as a detail — which is the
> same mistake at a different altitude, because a default *is* a measurement someone else
> already took. The most valuable thing in it turned out to be a claim about what we should
> *not* bother doing, and the way to check that claim was to ask the binary rather than the
> documentation. **`cuobjdump --list-elf` settled in four seconds a question a full rebuild
> would have answered in ninety minutes.**

---

Previous: [Chapter 11 — Choosing the file](11-quant-selection-qwen38.md) ·
Back to [README](README.md).
