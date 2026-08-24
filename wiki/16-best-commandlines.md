# Chapter 16 — Best command lines, and what actually backs them

Every other chapter in this wiki argues toward a recommendation. This one audits the
recommendations themselves, and asks a question none of the others do: **how well is each one
actually evidenced?**

The answer turned out to be uneven enough to matter. Two models carry dozens of logged runs. A
third carries a single number, quoted as fact in five separate places, that had never been
written to a results file — and that turned out to be measuring something other than what it
claimed. A dozen more GGUFs sit on disk with no measurements at all.

## 16.1 The commands live in ONE file, and it is not this one

> **[`.claude/skills/tuning-llamacpp-configs/references/best-commandlines.md`](../.claude/skills/tuning-llamacpp-configs/references/best-commandlines.md)**

That file is canonical. This chapter deliberately does **not** repeat a single command line from
it, for the reason spelled out in `CLAUDE.md`: three scripts once existed in both `wiki/` and
the skill, the two copies diverged silently, twice, and a bug fixed in one stayed live in the
other. Documentation duplicates the same way. **If you want to know what to run, follow the
link. If you want to know how much to trust it, stay here.**

Every row over there carries a confidence grade:

| grade | means | what you may do with it |
| --- | --- | --- |
| **A** | Repeated runs, logged, rows in a `results.tsv` | Trust the number |
| **B** | Single logged run, row exists | Trust the config; treat the digits as ±10% |
| **C** | Prose in a chapter — no row, no log | Starting point only. Re-measure before quoting |

The grades are not decoration. §16.3 is what happens when a C is mistaken for an A.

---

## 16.2 The ledger

| model | rows | where | grade |
| --- | ---: | --- | :-: |
| `Qwen3.8-27B-Q4_K_M` (dense, 15.66 GiB) | 23 | `benchmarks/results.tsv` | A/B |
| | 16 | `benchmarks/build-experiment/` | A |
| | — | `benchmarks/mmvq-experiment*/` (server-side, chapter 15) | A |
| `Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M` (MoE, 20.21 GiB) | 51 | `benchmarks/results.tsv` | A/B |
| | 22 | `benchmarks/a3b-asym-kv/` | A |
| `Qwen3.6-35B-A3B` × 3 quants | 11 | `benchmarks/qwen36-quant-compare/` | A |
| **everything else on disk** | **0** | — | — |

Note what the ledger does *not* contain: a hardware column, or a build column. That is why every
non-baseline campaign gets its own directory rather than appending to the main
`results.tsv` — a rule chapter 14 established and chapters 15 and 16 both follow.

---

## 16.3 The confound: "Q3_K kernels are slower than Q4_K" was never a quant measurement

Chapter 6 §6.1 records:

> **This Q4_K_M model generates faster than the smaller Q3_K_XL** — 117 vs 94 t/s. Q3_K CUDA
> kernels are slower than Q4_K ones. A smaller file is not automatically a faster model.

That finding propagated. It appears in `00-KEY-FINDINGS.md`, in chapter 7's trap table, twice in
chapter 11, and in the header of `serve-qwen36-130k.ps1` as the reason to expect "~20% slower
generation". It reads like a measured kernel property.

**The two files it compares are different models.** 94 t/s came from
`unsloth/Qwen3.6-35B-A3B-UD-Q3_K_XL`; 117 t/s came from
`Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M` — a different fine-tune, from a different
publisher, with an extra MTP head. "Q3_K is slower" and "it is a different model" were never
separated. And neither number was ever written to a results file.

### The clean test

Three quants of **the same base model**, one identical config — `-c 32768 -ctk q8_0 -ctv q8_0
-ngl 999 -sm layer -ts 12,29 -fa on -b 2048 -ub 512 -fit off` — on the stock binaries, three
interleaved repetitions each:

| arm | file | prefill (3 reps) | mean | generation (3 reps) | mean | peak VRAM |
| --- | --- | --- | ---: | --- | ---: | ---: |
| **A** | unsloth **UD-Q3_K_XL** | 2748 · 2802 · 2783 | **2778** | 91.2 · 88.3 · 92.0 | **90.5** | **18273** |
| B | unsloth UD-Q4_K_S | 2525 · 2457 · 2305 | 2429 | 92.5 · 89.2 · 92.6 | 91.5 | 22035 |
| C | lmstudio-community Q4_K_M | 2079 · 1361 · 1297 | 1579 | 100.0 · 79.5 · 94.5 | 91.3 | 22414 |

A vs B isolates the quant with publisher and imatrix method held constant. B vs C isolates
publisher.

**Generation: 90.5 / 91.5 / 91.3 — a 1.0% spread.** There is no Q3-vs-Q4 generation penalty on
this hardware. Arms A and B varied by under 4.1% across reps, so this null result is solid, not
an artefact of noise swamping a small effect.

**Verdict: refuted.** Arm A reproduces chapter 6's ~94 t/s closely (90.5), but same-base-model Q4
gives **91**, not 117. The 23-point gap chapter 6 attributed to CUDA kernels is a **fine-tune
difference**. Qwopus really is faster — 105.3 t/s at a comparable c32768 / `q8_0` point — but
for reasons that have nothing to do with Q3_K.

### And the practical advice inverts

Q3_K_XL is not the compromise the wiki implies. It is the best of the three on every axis except
quality: **+14% prefill** over Q4_K_S, **+76%** over lmstudio's Q4_K_M, identical generation, and
**~3.8 GiB less VRAM**.

> The warning chapter 6 drew — *"a smaller file is not automatically a faster model"* — is still
> worth keeping, because it is true in general and Q3_K genuinely can be slower elsewhere. But
> the evidence offered for it here said the opposite of what it was read as saying, and on this
> box the smaller quant is *strictly* faster.

**One arm is unstable and should be avoided.** Arm C gave prefill 1304 / 1952 / 2079 / 1361 /
1297 across five runs (including a first, aborted campaign) and generation down to 79.5. It is
the largest of the three and sits nearest the VRAM wall at 22464 MiB. A and B never wobbled.

### A methodological note worth more than the finding

Two things nearly corrupted this measurement, and both were caught only by looking:

1. **The first run of any model is cold.** The initial campaign showed the same config at
   prefill 1305 then 1952 — **+50%** — purely because a 21 GB file was being read from disk the
   first time and from the OS file cache the second. A model comparison where each arm runs once
   measures disk caching, not quants. Three interleaved reps make rep 1 cold for every arm.
2. **PowerShell variables are case-insensitive.** The first script validated paths with
   `foreach ($m in $M) {...}`, which overwrote the array `$M` with the loop variable `$m`. The
   main loop then iterated a single hashtable, ran **one arm out of three**, reported no error,
   and produced a plausible-looking results file. The rewrite asserts `$MODELS.Count -eq 3`
   before starting.

Both belong in the same family as chapter 15's lesson: **the instrument is part of the
experiment.** A campaign that silently measures the wrong thing is worse than one that fails.

---

## 16.4 What has never been measured

Twelve-plus GGUF files are on disk with zero rows anywhere. The ones large enough to matter:

| size | model | why it is interesting |
| ---: | --- | --- |
| 21.2 GB | `Jackrong/Qwopus3.6-35B-A3B-v1-Q4_K_M` | Non-Coder Qwopus. Would isolate the fine-tune that §16.3 showed is worth ~15% |
| 20.9 GB | `unsloth/Qwen3.6-35B-A3B-UD-Q4_K_S` | now measured — see §16.3 arm B |
| 16.8 GB | `Jackrong/Qwopus3.6-27B-Coder-MTP-Q4_K_M` | A **dense** 27B with an MTP head. Would separate "dense" from "Qwen3.8" in chapter 10's MTP result |
| 16.8 GB | `lmstudio-community/gemma-4-26B-A4B-it-Q4_K_M` | A different architecture entirely — every `-ts` and `-ub` conclusion here is Qwen-shaped |
| 7.4 GB | `lmstudio-community/gemma-4-12B-it-Q4_K_M` | Small enough to fit one card; would test whether any of this survives without a split |

**Do not invent a command line for these.** §4 of the reference file gives the derivation order:
predict the fit with `vram-budget.ps1`, start from the nearest row *of the same shape* (dense vs
MoE — chapters 6 and 10 disagreed on five of six effects and two flipped sign), re-derive `-ts`,
sweep into a **new** results directory, gate lossy KV with `needle-test.ps1`, and never claim a
drafting speedup without `mtp-test.ps1 -Temperature 0`.

---

## 16.5 Scorecard

| | |
| --- | --- |
| **Refuted** | "Q3_K CUDA kernels are slower than Q4_K", as evidenced in chapter 6. Generation is quant-independent here to within 1.0%. The 94-vs-117 gap was a fine-tune difference |
| **Corrected** | `serve-qwen36-130k.ps1`'s "~20% slower generation" warning; chapter 6 §6.1; the trap table in chapter 7; two rows in chapter 11; `00-KEY-FINDINGS.md` |
| **New** | Q3_K_XL is the *fastest* of three Qwen3.6 quants on prefill (+76% over one Q4_K_M) and 3.8 GiB smaller, at equal generation. `-ts` is a **proportion**, not a layer count — llama.cpp: *"fraction of the model to offload to each GPU… e.g. 3,1"* |
| **Untested** | Every config in §16.4. The Qwen3.6 130k f16 row (measurements here are c32768 / `q8_0`). Whether arm C's instability is VRAM proximity or something else |

> **The point of this chapter.** A wiki that records only its conclusions decays into folklore,
> because a number quoted five times looks five times as solid as a number quoted once — and
> both may rest on one unlogged run of the wrong comparison. Grading every recommendation by its
> evidence is less satisfying than adding another finding, and it is what stopped this repo from
> continuing to tell people that a 3.8 GiB saving costs them 20% of their generation speed.

---

Previous: [Chapter 15 — The MMVQ Ampere cap](15-mmvq-ampere-experiment.md) ·
Back to [README](README.md).
