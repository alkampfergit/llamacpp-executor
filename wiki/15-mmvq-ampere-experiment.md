# Chapter 15 — The MMVQ Ampere cap: a refuted hypothesis, and the instrument that could not have found it

This chapter records a negative result and a measurement bug, in that order of importance —
because the measurement bug is the part that generalises.

The chain started from an examination of **DFlash2**, the block-diffusion drafter currently
attracting attention. That examination is §15.1 and it is short, because the answer is
"unmerged upstream, and the impressive numbers are BF16". The interesting part was a side
finding in the pull request's discussion: a **kernel-selection cliff** at the exact batch sizes
speculative decoding uses. That looked like a real, cheap, upstreamable win for this box.

It was not. Chasing it produced:

| | |
| --- | --- |
| **The hypothesis** | Ampere has no tuned entry in `ggml_cuda_should_use_mmvq`, so it keeps using quantized GEMV up to a batch of 8, where MMQ is already faster. Lowering that cap should speed up MTP's verify batch |
| **The verdict** | **Refuted.** +0.1% where the comparison is valid |
| **What was actually wrong** | `mtp-test.ps1` could not resolve the effect. At `temperature 0.6`, **87% of throughput variance is draft-acceptance luck** (r² = 0.872, n = 12). The first sweep showed +15% and +22%; both were noise |
| **What survives** | Corrections to chapters 10 and 13, six harness fixes, and one reproducibility fact about llama.cpp's kernel tables |

---

## 15.1 DFlash2: real, unmerged, and much weaker at 4-bit than advertised

[PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) adds DFlash2 (grouped dynamic
depthwise convolution plus a candidate selector, over DFlash v1). Status as read on
**2026-08-23**: **open**, 20 files, +689/−83, with a member review saying it *"deviates quite
significantly from the current design"* and that *"some of the code appears to be
AI-generated"*. A collaborator locked the thread for noise, then unlocked it.
`--split-mode tensor` support was asked for and deferred to a follow-up. This is not
merge-imminent.

**The numbers that matter here are the author's own, on Qwen3.8-27B `Q4_K_M`** — the only quant
that fits 24 GiB:

| arm | speedup |
| --- | ---: |
| MTP `n_max` 2 | **1.30×** |
| MTP `n_max` 7 | **0.909×** — a loss |
| DFlash2 `n_max` 7 | **1.15×** |
| DFlash2 `n_max` 7, MMVQ ≤ 4 | 1.447× |

The author states it plainly: *"DFlash2 is slower than MTP, and here I reproduced it on
Qwen3.8-27B Q4_K_M."* The widely-quoted 2–3.4× figures are **BF16** (27 B at BF16 is ~54 GB,
off the table on 24 GiB) or SGLang server throughput. The 2.06× DFlash2 result is BF16.

### Two corrections to chapter 13

**§13.11 credits the report with "our baseline binary already accepts it".** That is true of the
*flag* and false of the *checkpoint*. `--spec-type` does list `draft-dflash`, and
`src/models/dflash.cpp` does handle DFlash v1 plus DSpark's Markov and confidence heads — but it
creates **no convolution or selector tensors**. A DFlash2 GGUF therefore leaves
`n_created < n_tensors`, and `llama_model_loader::done_getting_tensors` throws
`wrong number of tensors; expected N, got M`. **Loud, not silent** — but it does not load.
DFlash2 needs the PR built from source.

**There is no zero-rebuild path for this target.** The Hugging Face API lists v1 `DFlash`
drafters for Qwen3.6-27B, Qwen3.5-4B, gemma-4 and others, but for **Qwen3.8-27B only DFlash2
exists** (`z-lab/Qwen3.8-27B-DFlash2{,-GGUF}`). §13.10's experiment E6 is therefore gated on a
build, not just on a 1090 MiB download.

---

## 15.2 The hypothesis, and why it looked good

`ggml/src/ggml-cuda/mmvq.cu` selects between quantized GEMV (MMVQ) and the quantized GEMM
kernel (MMQ) with a **per-compute-capability table**:

```cpp
bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    ...
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {   // tuned on RTX 4090
        case GGML_TYPE_Q4_K: ... return ne11 <= 7;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {      // tuned on RTX 5090
        case GGML_TYPE_Q4_K: ... return ne11 <= 5;
    }
    ...
    return ne11 <= MMVQ_MAX_BATCH_SIZE;   // 8
}
```

Those are exact `==` comparisons against `890` and `1200`. **Ampere is 860 and matches
nothing**, so it falls through to 8. This machine straddles the gap:

| Device | Card | `cc` | Q4_K uses MMVQ up to |
| --- | --- | ---: | ---: |
| 0 | RTX 3070 (Ampere) | 860 | **8** (untuned fallback) |
| 1 | RTX 5060 Ti (Blackwell) | 1200 | **5** (tuned on a 5090) |

With `-sm layer`, a single verify batch of 8 therefore takes **a different matmul kernel on each
GPU**. The PR author's per-step timings on H200 `Q4_K_M` show why that could matter —
`B=7: 31.78 ms, B=8: 34.86 ms, B=9: 27.86 ms` — cost climbing steeply through 8, then *dropping*
when MMQ takes over at 9.

> `GGML_CUDA_FORCE_MMQ` cannot test this. It is a compile-time `#ifdef` (`mmq.cu:320`), not an
> env var — the same trap as `GGML_SCHED_MAX_COPIES` in chapter 14 — and it gates
> `ggml_cuda_should_use_mmq`, which the dispatcher reaches only *after* MMVQ has already claimed
> the batch.

### The patch: one binary, two behaviours

Rather than build a control and a treatment, the cap was made **runtime-settable**, so a single
binary provides both arms with no build-provenance difference at all:

```cpp
// Ampere (cc 800-869) has no tuned entry below and therefore falls through to
// MMVQ_MAX_BATCH_SIZE ... GGML_CUDA_MMVQ_MAX_BATCH sets the Ampere k-quant cap at
// runtime so one binary can measure both behaviours; unset keeps upstream behaviour.
if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_ADA_LOVELACE) {
    static const int ampere_cap = [] { ... getenv("GGML_CUDA_MMVQ_MAX_BATCH") ... }();
    if (ampere_cap > 0) {
        switch (type) {
            case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:  return ne11 <= ampere_cap;
            default:                                   return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
}
```

Only k-quants, matching the existing branches and the source's own reasoning ("k-quants cost
more to decode and mvq redoes that per column, so MMQ wins sooner"). A cap of **5** makes Ampere
behave exactly like Blackwell for Q4_K, so device 1 is untouched and the experiment has one
variable.

**Build:** `build-mmvq/bin`, 8.6 min, `build 10585, commit 95aab10a9`, MSVC 19.44.35228,
CUDA 13.3, `86-real;120-real`, default ggml flags, `--target llama-server`.

### Preconditions, checked not assumed

| CLAUDE.md precondition | How it was satisfied |
| --- | --- |
| 1 — sysmem fallback covers **this** binary | `-ts 100,0 -c 130048` → `cudaMalloc failed: out of memory` in **7.1 s**. The driver refuses; it does not absorb |
| 2 — new build runs from its own `bin/` | Server launched with `WorkingDirectory = build-mmvq\bin` |
| 3 — every number labelled with its build | Harness now prints and records `--version` per sweep |
| 4 — the knob is actually read | `strings ggml-cuda.dll` finds `GGML_CUDA_MMVQ_MAX_BATCH`. The `getenv` survived optimisation |

That last row exists because of chapter 14 §14.7. A flag that is never read produces a clean,
publishable, entirely fictional result.

---

## 15.3 The first sweep, and why it was wrong

`n_max` 3–8, both arms, one run each, `-c 65536 -ub 512 -ctk/-ctv q8_0 -ts 22,43`, 400-token
code generation at `temperature 0.6`:

| `n_max` | cap unset | cap 5 | Δ |
| ---: | ---: | ---: | ---: |
| baseline | 22.63 | 22.64 | +0.0% |
| 3 | 47.35 | 45.21 | −4.5% |
| 4 | 48.91 | 49.50 | +1.2% |
| 5 | 51.82 | 49.17 | −5.1% |
| 6 | **54.05** | 50.72 | −6.2% |
| 7 | 43.44 | 50.08 | **+15.3%** |
| 8 | 42.75 | **51.97** | **+21.6%** |

This looked like a finding: a −20% cliff between n6 and n7 in the unpatched arm, flattened
entirely by the cap. It was reported as one. **It was noise.**

The stated prediction had already failed, which was the first clue. The prediction was a clean
step — n5/n6/n7 change, n3/n4/n8 do not, because the verify batch is `n_max + 1`. Instead n8
moved most and n6 not at all. Reading `common/speculative.cpp` explains the mechanism honestly:
`chain_heads` is false for this model (one `nextn` layer), so `n_max` is never clamped, but the
draft length per round is **variable** — cut short by `p_min`. The verify batch is a
*distribution*, not a fixed value. A cap clips whatever fraction of rounds exceeds it, and that
fraction grows with `n_max`.

That is a perfectly good post-hoc story. It is also unnecessary, because the effect does not
exist.

---

## 15.4 Three repetitions kill it

Same points, arms **interleaved** so thermal drift could not masquerade as an arm effect:

| `n_max` | cap unset | cap 5 |
| ---: | --- | --- |
| 6 | 54.05 · 44.94 · 50.34 | 50.72 · 53.50 · 49.75 |
| 7 | 43.44 · 43.75 · 52.27 | 50.08 · 59.68 · 44.84 |
| 8 | 42.75 · 51.58 · 46.27 | 51.97 · 42.18 · 42.66 |

Within-arm spread reaches **33%** (cap 5, n7: 44.84 → 59.68). Mean TG: **48.19 vs 48.77,
+1.2%** — nothing. The n7 "cliff" did not reproduce; the n8 "gain" reversed.

### The acceptance column, which the harness did not have until today

| | |
| --- | --- |
| **TG vs draft acceptance** | **r = 0.934, r² = 0.872** across all 12 runs |
| Mean acceptance, cap unset | **0.5602** |
| Mean acceptance, cap 5 | **0.5587** |

Sorted by acceptance, the twelve runs line up almost monotonically in throughput *regardless of
arm* — 0.445 → 42.18 t/s at the bottom, 0.704 → 59.68 t/s at the top. And the two arms have
**the same acceptance distribution**, which is the correctness check: a kernel choice cannot
change which tokens the target accepts, because speculative decoding is exact.

> **At `temperature 0.6`, `mtp-test.ps1` measures sampling luck.** Each run samples a different
> token sequence, so a different fraction of drafts survives, so throughput moves by ±20% with
> nothing changed. The **baselines** were tight across all runs (21.95–22.68, ±1.6%) — the noise
> is specific to drafting, exactly as the mechanism predicts.

---

## 15.5 Greedy decoding, and the confound the output hash caught

`temperature 0` makes the token sequence — and therefore acceptance — repeatable. Two
repetitions per cell:

| `n_max` | cap unset | cap 5 | unset mean | cap-5 mean | Δ | output text |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 6 | 54.55 / 56.81 | 58.02 / 58.48 | 55.68 | 58.25 | +4.6% | **DIFFERENT** |
| 7 | 52.91 / 53.39 | 58.20 / 57.17 | 53.15 | 57.69 | +8.5% | **DIFFERENT** |
| 8 | 55.44 / 54.41 | 55.30 / 54.63 | 54.92 | 54.97 | **+0.1%** | **identical** |

Greedy fixed the instrument: acceptance is now bit-identical across repetitions in every cell,
and rep-to-rep spread fell to **0.8–4.1%** from 33%.

**The verdict is the n8 row.** It is the one cell where both arms provably computed the same
thing — same output hash, same acceptance 0.6044, same mean accepted run 5.78. The cap changes
throughput there by **+0.1%**.

**Where it still looked like a win, it was a different computation.** At n6 and n7 the output
hashes differ. Switching MMVQ→MMQ changes float accumulation order, which changed the greedy
token sequence, which changed acceptance (0.700 → 0.720 and 0.643 → 0.676) and mean accepted run:

| `n_max` | mean accepted | TG | residual after acceptance |
| ---: | ---: | ---: | ---: |
| 6 | +2.7% | +4.6% | +1.9% |
| 7 | +4.2% | +8.5% | +4.3% |
| 8 | +0.0% | +0.1% | +0.1% |

Both residuals sit inside the 0.8–4.1% rep-to-rep spread. The cap **is** faster in wall-clock on
this prompt at n6/n7 — but not because kernels got cheaper. It took a token path with better
acceptance. That is a coin flip on one prompt, not a reproducible win, and on another prompt the
numerics could land the other way.

### A reproducibility fact worth keeping

**llama.cpp's per-architecture kernel-selection tables change greedy output.** MMVQ and MMQ are
not bit-identical, so the same model, prompt and seed can produce different deterministic text
depending on which kernel a batch size selects. On this box **that already happens with no patch
at all**: the 3070 and the 5060 Ti take different branches in `ggml_cuda_should_use_mmvq` for
Q4_K at the same batch size, so with `-sm layer` the two cards' layers are not numerically
interchangeable. Anyone comparing greedy output across builds, GPUs, or split ratios should
expect divergence and not treat it as corruption.

---

## 15.6 A correction to chapter 10 §10.7

§10.7's `n_max` table is one run per point at `temperature 0.6`. By §15.4 that method resolves
nothing below roughly ±20% for drafted generation.

**What survives:** MTP on this model is worth about **2× generation**, and the baseline
reproduced at 22.42 / 22.55 / 22.63 / 22.64 / 21.95 / 22.68 t/s across five separate campaigns
and two different binaries. That is solid.

**What does not survive:** the *shape* of the curve. The "dip" at n5 (38.1 between 47.1 and
44.4), and the apparent optimum at n4/n8, are the acceptance lottery. On build 10585 at greedy,
n6/n7/n8 are 55.7 / 53.2 / 54.9 — flat within noise. **Do not tune `n_max` off that table.** If a
best `n_max` matters, re-measure at `-Temperature 0`.

This also retires, unresolved, the idea that §10.7's n5 dip was a kernel-selection artefact. It
was not evidence of anything.

---

## 15.7 The harness fixes, which are the real yield

All in `.claude/skills/tuning-llamacpp-configs/scripts/`.

| Fix | Why it mattered |
| --- | --- |
| **`-LlamaDir` on `mtp-test.ps1`** | It located `llama-server.exe` by walking up from its own path — which in this repo lands on the **baseline root**. Every "new build" measurement would silently have measured the baseline. Now matches the `-LlamaDir` contract `bench-harness.ps1` and `vram-budget.ps1` already used |
| **`-LlamaDir` on `needle-test.ps1`** | Found by grepping for siblings of the above, per the standing rule. Identical idiom, identical trap |
| **`-Temperature`** | Default stays 0.6. Pass 0 to resolve anything below ±20% |
| **acceptance / mean-accepted columns** | Parsed from the server's own `print_timing` line. Without them this chapter's entire conclusion is invisible. The first attempt failed silently because the log pads as `mean len =  4.81` — **two** spaces — and the pattern expected one |
| **output SHA column** | Caught the numerics confound in §15.5 on its first use |
| **per-env log filenames** | Two sweeps differing only by an env var overwrote each other's logs, and arm A's acceptance data was lost before anyone looked at it |

> **The lesson.** Chapter 14 asked "did this flag change the binary?" and answered it with
> `cuobjdump`. This chapter asked "did this flag change the speed?" and could not answer it,
> because the instrument's noise was four times the effect. **Before measuring a small effect,
> measure the instrument.** One repeat of one configuration would have exposed this in three
> minutes and saved two full sweeps. The tell was there from the start and was rationalised
> away: a 20% non-monotonic wiggle in the very first table, in a repo whose own CLAUDE.md lists
> "non-monotonic behaviour" as a signal to stop and go read something.

---

## 15.8 Scorecard

| | |
| --- | --- |
| **Refuted** | The Ampere MMVQ k-quant cap changes MTP throughput on this box. +0.1% where the comparison is valid. Branch `exp/mmvq-ampere-cap` is kept as a record; nothing was merged |
| **Corroborated** | The kernel-table asymmetry itself is real and untuned upstream for Ampere, and this box runs two different branches simultaneously under `-sm layer` |
| **Corrected** | §13.11 (DFlash2 checkpoint vs flag), §13.10 E6 (needs a build, not just a download), §10.7 (curve shape unreliable; headline intact) |
| **Untested** | Whether the cap matters at higher concurrency, where verify batches stack and the batch distribution shifts upward. Every measurement here is `-np 1` |
| **New** | llama.cpp's kernel-selection tables change greedy output, and already do so across this box's two GPUs |

> DFlash2 was the question. The answer is "unmerged, and 1.15× at the only quant that fits."
> The chapter is long anyway, because the *interesting* failure was ours: two sweeps and a
> custom CUDA patch spent on an effect that a single repeated measurement would have shown was
> below the noise floor. The patch works. The knob is read. The build is clean. None of that
> makes a 0.1% result into a finding.

---

Evidence: `wiki/benchmarks/mmvq-experiment/` (temperature 0.6) and
`wiki/benchmarks/mmvq-experiment-greedy/` (temperature 0). Each carries a
`PROVENANCE.md` naming the binary, the config, and **which logs are missing** — both
directories hold a partial log set, because the per-env filename fix in §15.7 landed
mid-experiment and never covered repetition collisions. The `mtp-results.md` in each
is complete; the logs are not, and pairing a log with a row by filename alone will
mislead you.

Previous: [Chapter 14 — The build experiment](14-build-experiment.md) ·
Back to [README](README.md).
