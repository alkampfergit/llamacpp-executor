# Running Local LLMs with Raw llama.cpp — A Practical Course

Welcome. This wiki teaches you how to serve a GGUF language model over HTTP using
**raw llama.cpp** — no LM Studio, no Ollama, no wrapper — and how to *tune* it so it
actually uses the hardware you paid for.

It is written for one specific machine, because vague advice is useless advice:

| Component | Detail |
| --- | --- |
| GPU 0 | NVIDIA RTX 3070, 8 GiB, Ampere (CC 8.6) |
| GPU 1 | NVIDIA RTX 5060 Ti, 16 GiB, Blackwell (CC 12.0) |
| Total VRAM | 24 GiB (heterogeneous!) |
| CPU | AMD Ryzen 9 5900X, 12 cores / 24 threads |
| RAM | 64 GiB |
| OS | Windows 11 Pro |
| llama.cpp | build `fe8156f78` (10509) |
| Display attached to | GPU 1 (the 5060 Ti) — this matters more than you think |

Everything here was **measured on this machine**, not copied from a forum post. Where a
number appears, there is a log file behind it in [`benchmarks/logs/`](benchmarks/logs/).

---

## How to read this wiki

> # → START WITH [🔴 KEY FINDINGS](00-KEY-FINDINGS.md) ←
> Every important discovery from the whole campaign, stated loudly and briefly: the silent
> Windows overflow that cost 10×, the KV cache pairing that costs 14× with no error message,
> why CPU offload destroys an MoE, why speculative decoding made this model *slower* despite
> 70% draft acceptance, and the one correction where this wiki was wrong.

The chapters below build on each other. If you are impatient, read chapter 7, copy a command,
and come back later when you want to know *why* it looks like that.

| # | Chapter | What you will learn |
| --- | --- | --- |
| **0** | [**🔴 KEY FINDINGS**](00-KEY-FINDINGS.md) | **Everything that matters, loudly. Read first.** |
| 1 | [The mental model](01-mental-model.md) | What llama.cpp actually is, which binary does what, and the single most important idea: **where every byte of VRAM goes** |
| 2 | [Serving a model over HTTP](02-serving-http.md) | `llama-server` from first launch to a working OpenAI-compatible endpoint, tested with `curl` |
| 3 | [The parameters that matter](03-parameters.md) | Every flag worth knowing, what it trades away, and which ones are traps |
| 4 | [Budgeting your VRAM](04-vram-budget.md) | The arithmetic of model + KV cache + compute buffers, and how to predict a fit in one second instead of one minute |
| 5 | [Benchmarking honestly](05-benchmarking.md) | `llama-bench` vs `llama-batched-bench`, and the four ways a benchmark will lie to you |
| 6 | [Measured results](06-results.md) | The full tuning campaign on this hardware, with the surprises spelled out |
| 7 | [Recommended configurations](07-recommended-configs.md) | Copy-paste launch commands for each goal |
| 8 | [Troubleshooting](08-troubleshooting.md) | Slow prefill, silent OOM, gibberish output, and the Windows driver trap that ruins everything |
| 9 | [Speculative decoding & MTP](09-speculative-decoding.md) | Making generation faster without changing the model |
| 10 | [Qwen3.8-27B results](10-results-qwen38-27b.md) | A **dense** model on the same box — where the MoE tuning advice does and does not transfer |
| 11 | [Choosing the *file*](11-quant-selection-qwen38.md) | Which GGUF quant to download, and an audit of a third-party report: what survived measurement, what cost 27× the host buffer, and the one thing it got right that we never tested |
| 12 | [Auditing a *build* report](12-build-flags-analysis.md) | Which `cmake` flags actually change the binary — and how `cuobjdump` settled in four seconds what a rebuild would have taken ninety minutes to answer. Half the flags in a third-party report are already the default; one of ours does not exist |
| 13 | [The speculative-decoding landscape](13-speculative-decoding-landscape.md) | **DFlash2, DSpark and PFlash.** What `--spec-type` really does with a list (a priority cascade, not a sum), why an n-gram outranks your MTP head, why DFlash2 cannot fit at 130k here, and the one thing that beats every speedup in this wiki: **a speedup that competes with your prompt cache looks identical to one that composes with it — until the second request** |
| 14 | [The build experiment](14-build-experiment.md) | **The first numbers in this wiki not produced by the shipped binaries.** Two self-compiled builds differing in exactly two flags: what `GGML_CUDA_FA_ALL_QUANTS` is worth (asymmetric KV, measured), what `GGML_SCHED_MAX_COPIES=1` is worth (not throughput — VRAM), and why the control must be a *rebuild* rather than the baseline |
| 15 | [The MMVQ Ampere cap](15-mmvq-ampere-experiment.md) | **A refuted hypothesis, and the instrument that could not have found it.** DFlash2 is unmerged upstream and worth 1.15× at the only quant that fits. Chasing a kernel-selection cliff instead produced a custom CUDA patch, two sweeps, and +0.1% — because **87% of this harness's throughput variance was draft-acceptance luck** (r² = 0.872). Corrects the *shape* of §10.7's `n-max` curve, and records that llama.cpp's per-GPU kernel tables change greedy output |
| 16 | [Best command lines, and what backs them](16-best-commandlines.md) | **The provenance ledger for every recommendation in this wiki.** The commands themselves live in one place — `.claude/skills/tuning-llamacpp-configs/references/best-commandlines.md` — graded A/B/C by how well each is actually evidenced. This chapter is what that grade *means*: which configs are repeated and logged, which are a single run, which are prose someone later quoted as fact, the twelve GGUFs on disk that have never been measured at all, and the fine-tune confound hiding inside chapter 6's "Q3_K kernels are slower than Q4_K" |

---

## The one thing to learn first

If you take away a single idea, take this one:

> **A model that does not fit in VRAM does not fail loudly on Windows. It gets slow.**

Since NVIDIA driver 536.40, a CUDA program that asks for more VRAM than exists does not
get an out-of-memory error on Windows. The driver quietly puts the overflow in system RAM
("Shared GPU Memory") and lets the program continue — at roughly a fifth of the speed.

This is why the same model on the same hardware can be measured at 300 tokens/second in
one program and 3000 tokens/second in another. Nothing is broken. One of them overflowed.

Chapter 8 shows you how to turn this silent failure back into a loud one, which is the
single highest-value change you can make before you start tuning anything.

---

## Repository layout

```
wiki/
├── README.md                     ← you are here
├── 00-KEY-FINDINGS.md            ← read this first
├── 01-mental-model.md
├── 02-serving-http.md
├── 03-parameters.md
├── 04-vram-budget.md
├── 05-benchmarking.md
├── 06-results.md
├── 07-recommended-configs.md
├── 08-troubleshooting.md
├── 09-speculative-decoding.md
├── 10-results-qwen38-27b.md
├── 11-quant-selection-qwen38.md
├── 12-build-flags-analysis.md
├── 13-speculative-decoding-landscape.md
├── 14-build-experiment.md        ← CONTROL vs TREATMENT, self-compiled
├── 15-mmvq-ampere-experiment.md  ← a REFUTED result, and how the harness was lying
├── 16-best-commandlines.md       ← provenance for the per-model command-line table
├── pre-build-plan.md             ← written BEFORE the build, so the design can be judged
├── nvidia-sysmem-fallback-paths.md
└── benchmarks/
    ├── results.tsv               ← every BASELINE throughput measurement, append-only
    ├── needle-results.md         ← long-context quality-gate results
    ├── mtp-results.md            ← speculative-decoding results
    ├── logs/                     ← full stdout+stderr per run
    └── build-experiment/         ← ch.14 only: self-compiled builds, kept SEPARATE
        ├── results.tsv           ←   so baseline rows and rebuild rows can never be
        └── logs/                 ←   silently averaged together
```

There is no `wiki/scripts/` — every operative script (the measurement harness, the
quality gates, the `serve-*` launchers) lives under the reusable **skill** at
[`.claude/skills/tuning-llamacpp-configs/scripts/`](../.claude/skills/tuning-llamacpp-configs/scripts/),
which this wiki only references. That skill also teaches the whole method to Claude Code, so
you can run `/tuning-llamacpp-configs <model.gguf>` on any future model or GPU and get the same
procedure applied automatically:

```
.claude/skills/tuning-llamacpp-configs/scripts/
├── bench-harness.ps1         ← the measurement harness (dot-source, call Probe)
├── resume-benchmarks.ps1     ← this repo's remaining measurement queue
├── needle-test.ps1           ← long-context QUALITY gate (run before trusting q4_0 KV)
├── mtp-test.ps1              ← speculative-decoding measurement over HTTP
├── deep-probe-q38.ps1        ← real deep-context HTTP numbers for Qwen3.8-27B
├── vram-budget.ps1           ← hardware discovery, -c x -ub feasibility grid
├── serve-qwopus-130k.ps1     ← Profile A: q8_0 KV, full 130k (quality-first)
├── serve-qwopus-q4-130k.ps1  ← Profile B: q4_0 KV, full 130k (speed-first)
├── serve-qwopus-fast.ps1     ← Profile C: q8_0 KV at 127k (best all-round)
├── serve-qwen36-130k.ps1     ← Profile D: Q3_K_XL with headroom
└── serve-qwen38-27b.ps1      ← the four measured profiles for Qwen3.8-27B Q4_K_M
```

All of them require `-Model`, and the benchmark scripts default their output to
`bench-results/` next to the binaries — pass `-OutDir wiki\benchmarks` (or the repo-specific
defaults already baked into `resume-benchmarks.ps1`) to keep this repo's own evidence trail
landing in `wiki/benchmarks/`.

---

## Models covered

Both live under `S:\HuggingFace\lmstudio\`:

| Model | Size | Notes |
| --- | --- | --- |
| `unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | 15.69 GiB | Comfortable fit. Room for a large context. |
| `Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF/…-Q4_K_M.gguf` | 20.22 GiB | Higher quality, includes an **MTP head** for self-speculative decoding. A very tight fit at 130k context — the interesting case. |

Both are `qwen35moe`: 35.5 B total parameters, ~3 B active per token, 256 experts with 8
used per token, and a **hybrid attention design** that makes long context unusually cheap.
Chapter 4 explains why that last point is the reason 130k context is possible at all.

