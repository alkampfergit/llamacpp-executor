# Chapter 6 — Measured Results

Every number here was measured on the machine in the [README](README.md), one fresh process
per data point. Raw data: [`benchmarks/results.tsv`](benchmarks/results.tsv),
[`benchmarks/needle-results.md`](benchmarks/needle-results.md), logs in
[`benchmarks/logs/`](benchmarks/logs/).

Model: **`Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf`**, 20.21 GiB, 35.51 B params.

`PP` = prefill tokens/second, `TG` = generation tokens/second. Differences under ~8% are
noise (§5.7).

> **Two kinds of number appear below.** *Synthetic* figures come from
> `llama-batched-bench` with an 8192-token prompt. *Real* figures come from `llama-server`
> over HTTP with a 107,743-token prompt. They differ a lot, and the real ones are the ones
> that predict your wait. Both are labelled.

---

## 6.1 The headline

| | Synthetic PP | Real PP @108k | TG shallow | TG @108k |
| --- | --- | --- | --- | --- |
| Peak possible (small context) | 2651 | — | 117 | — |
| **`q4_0` KV, `ub 1024`, 130k** | **2650** | **1758** | 105 | 46.6 |
| **`q8_0` KV, `ub 512`, 130k** | 1850 | 1414 | 98 | 47.5 |
| `q8_0` KV, `ub 512`, 127k | **2323** | — | 109 | — |
| What CPU offload would give | 826 | — | 88 | — |
| What LM Studio gave | ~300 | — | — | — |

**A 130k context now runs at essentially the same prefill speed as a tiny one** — 2650 vs
2651 synthetic. The context tax measured in §6.4 turned out to be an artefact of running out
of memory, not a law of nature.

Two other findings worth stating up front:

- **This Q4_K_M model generates faster than the smaller Q3_K_XL** — 117 vs 94 t/s. Q3_K CUDA
  kernels are slower than Q4_K. A smaller file is not a faster model.
  > ⚠️ **The observation holds; the explanation is wrong.** See
  > [chapter 16 §16.3](16-best-commandlines.md). These two files are different *fine-tunes*
  > (unsloth Qwen3.6 vs Jackrong Qwopus), so the 24% gap is a model difference, not a kernel
  > one. Three quants of the **same** base model generate at 90.5 / 91.5 / 91.3 t/s — a 1.0%
  > spread. Do not cite this line as evidence about Q3_K kernels.
- **Generation halves at depth.** 98–105 t/s on a short prompt, **~47 t/s** at 108k tokens.
  Every "tokens per second" claim you read online is a shallow-context number.

---

## 6.2 Correction: pipeline parallelism, and how I got it wrong

An earlier version of this wiki told you to set:

```powershell
$env:GGML_SCHED_MAX_COPIES = "1"
```

**That does nothing.** `GGML_SCHED_MAX_COPIES` is a **compile-time** `#define` in
`ggml/src/ggml-backend.cpp`:

```c
#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif
```

It is set at build time via CMake (`-DGGML_SCHED_MAX_COPIES=1`), and is never read from the
environment. The evidence was in my own logs: a run **with the variable set** still tried to
allocate 1407 MiB of pipelined buffers and then fell back. The earlier "+13%" I attributed to
it was run-to-run variance.

**The underlying effect is real, and larger than I first measured.** What actually happens:

```
graph_reserve: failed to allocate compute buffers
sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
```

When pipelined buffers don't fit, llama.cpp retries **without** pipeline parallelism — a mode
that is both smaller *and faster* on mismatched GPUs, because a pipeline runs at the speed of
its slowest stage and the 3070 is much slower than the 5060 Ti.

And it is reproducible. Three identical runs of `q4_0` KV at `-ub 1024`, 130k:

| Run | PP t/s | TG t/s |
| --- | --- | --- |
| 1 | 2757 | 105.7 |
| 2 | 2599 | 105.3 |
| 3 | 2594 | 104.9 |

Mean ~2650, spread ±3%. This is a stable operating point, not a fluke.

> **So the fast path is reached by memory pressure, not by a flag.** That is uncomfortable —
> you are relying on an allocation failing. It is reproducible while your VRAM conditions are
> stable, but if you freed ~1.4 GiB the pipelined path might succeed and run *slower*.
>
> **The deterministic fix is to rebuild:**
> ```
> cmake -B build -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1
> ```
> That guarantees the lean path, makes behaviour monotonic in `-c` and `-ub`, and would very
> likely let `q8_0` KV run at `-ub 1024` at full 130k — the best of both worlds. **Untested;
> it is the top open item.**

**The general lesson:** verify whether a tuning knob is runtime or compile-time before
believing a measurement you attribute to it. A 13% change on a machine with ±8% noise is not
evidence.

---

## 6.3 KV cache quantisation: pick a *compiled kernel pair*

At 130k, `-ts 12,29`:

| `-ctk`/`-ctv` | KV size | Best PP (synthetic) | Fast FA kernel? |
| --- | --- | --- | --- |
| `f16`/`f16` | 2622 MiB | does not fit | ✅ |
| **`q8_0`/`q8_0`** | 1422 MiB | **1850** (`ub 512`) | ✅ |
| **`q4_0`/`q4_0`** | ~720 MiB | **2650** (`ub 1024`) | ✅ |
| `q5_1`/`q5_1` | ~1100 MiB | 107 | ❌ |
| `q4_0`/`q8_0` | ~1050 MiB | 160 | ❌ |

### The actual rule

A stock CUDA build compiles flash-attention kernels for only **four symmetric K/V pairs**.
From `ggml/src/ggml-cuda/CMakeLists.txt`:

```
fattn-vec-instance-f16-f16.cu      fattn-vec-instance-q8_0-q8_0.cu
fattn-vec-instance-bf16-bf16.cu    fattn-vec-instance-q4_0-q4_0.cu
```

and `ggml/src/ggml-cuda/fattn.cu` enforces it:

```c
if (K->type != V->type) {
    return BEST_FATTN_KERNEL_NONE;      // when GGML_CUDA_FA_ALL_QUANTS is undefined
}
```

Supported types: **F32, F16, Q4_0, Q8_0, BF16**. So there are two independent ways to fall
off the fast path, and my two disasters hit one each:

| Config | Fault |
| --- | --- |
| `q4_0`/`q8_0` | **Asymmetric** — `K->type != V->type` |
| `q5_1`/`q5_1` | **Symmetric, but `q5_1` is not a supported type** |

**Symmetry is necessary but not sufficient.** When no kernel qualifies, attention runs on
another backend — which is why `-ctv q4_0` showed **5277 MiB in host RAM** instead of 793.
It was never "4-bit is slow"; it was "attention left the GPU".

> **Rule: use a symmetric pair from the compiled set — `f16/f16`, `bf16/bf16`, `q8_0/q8_0`,
> `q4_0/q4_0`. Never mix types; never use a type outside that list.**

**Want asymmetric KV?** `q8_0` keys with `q4_0` values is genuinely attractive — keys
tolerate quantisation worse than values — but you must build it:

```
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
```

---

## 6.4 Is `q4_0` KV actually lossy? — the needle test

`q4_0` KV buys 24% more prefill. Throughput benchmarks **cannot** tell you what it costs in
quality, so a separate gate is needed.

**Method** (reproducible via
[`needle-test.ps1`](../.claude/skills/tuning-llamacpp-configs/scripts/needle-test.ps1)): build a
deliberately boring 3000-record telemetry log, hide one unguessable fact in **RECORD 0001** —
the very start, the hardest place to retain — and ask for it at the very end.

```
RECORD 0001: The commissioning passphrase for reactor bay 7 is CRIMSON-PELICAN-4417.
RECORD 0002..3000: routine telemetry; coolant nominal; no anomalies ...
Question: what is the commissioning passphrase for reactor bay 7?
```

`CRIMSON-PELICAN-4417` is arbitrary nonsense on purpose: it cannot be guessed, inferred, or
recalled from training data. A correct answer proves genuine retrieval across ~108,000 tokens
of near-duplicate filler. Total prompt: **107,743 tokens**, exercising most of a 130k window.

**Results:**

| Config | Needle | Real PP | Real TG | Wall |
| --- | --- | --- | --- | --- |
| `q8_0`/`q8_0`, `ub 512` | **PASS** | 1414 t/s | 47.5 t/s | 76.5 s |
| `q4_0`/`q4_0`, `ub 1024` | **PASS** | 1758 t/s | 46.6 t/s | 61.6 s |

Both retrieved `CRIMSON-PELICAN-4417` exactly, and `q4_0` did it **15 seconds faster**.

> **Read this result honestly.** It shows `q4_0` KV does not destroy long-range *exact
> recall*. It does **not** prove `q4_0` is free. Needle retrieval is an easy task — the
> answer is verbatim in context. It does not test multi-step reasoning over long context,
> summarisation faithfulness, or code edits where a subtly-wrong token matters.
>
> So: `q4_0` is **validated for retrieval-style work** at this context length. If you are
> doing something where quality is paramount, prefer `q8_0` and accept ~24% less prefill —
> which is exactly the trade the "quality-first" profile in chapter 7 makes.

A methodological trap worth recording: the **first two attempts returned an empty answer for
both KV types**, which looked like catastrophic quality failure. The cause was using the raw
`/completion` endpoint — an instruction-tuned model emits EOS immediately on an untemplated
prompt. Use `/v1/chat/completions`. A "quality failure" that affects every configuration
equally is usually a bug in your harness.

---

## 6.5 Trading context for speed — the quality-first frontier

If you prefer `q8_0`, how much context can you keep? `q8_0`/`q8_0`, `-ub 512`, `-ts 12,29`:

| `-c` | Synthetic PP | TG |
| --- | --- | --- |
| 122,880 | 2225 | 106 |
| **126,976** | **2323** | **109** |
| 129,024 | 2101 | 106 |
| 130,048 | 1850 | 98 |

**Giving up 3072 tokens of context (2.4%) buys 26% more prefill.** The curve is steep right
at the ceiling, because that is where VRAM pressure starts forcing compromises.

And `q8_0` at `ub 1024` with reduced context is **not** the answer:

| `-c` | `-ub` | PP | TG | GPU1 used |
| --- | --- | --- | --- | --- |
| 114,688 | 1024 | **1067** | **56.7** | 16005 / 16311 MiB |
| 122,880 | 1024 | **1036** | **77.6** | 16011 / 16311 MiB |
| 98,304 | 1024 | 2296 | 103 | 15918 MiB |

Look at the GPU1 column. The two slow rows had the display GPU **completely saturated** —
that is the Windows shared-memory spill from §5.3, caught live. Not a kernel effect; an
overflow. This is exactly why peak VRAM must be recorded next to every throughput number.

---

## 6.6 The micro-batch curve

Small context, so memory is not the constraint:

| `-ub` | PP t/s | TG t/s | Compute buffers |
| --- | --- | --- | --- |
| 128 | 1368 | 106 | ~0.4 GiB |
| 256 | 2017 | 102 | ~0.8 GiB |
| **512** | **2651** | 105 | ~1.6 GiB |
| 1024 | 2615 | 106 | ~3.2 GiB |
| 2048 | OOM | — | ~6.5 GiB |

512 is the peak; 1024 doubles memory for nothing. **TG is flat across the whole range** —
generation has no batch, so never spend VRAM on `-ub` for a chat workload.

At 130k, where memory *is* the constraint, the ranking changes because `-ub 1024` triggers
the lean non-pipelined path (§6.2):

| KV | `-ub` | PP | TG |
| --- | --- | --- | --- |
| `q4_0` | 256 | 1616 | 102 |
| `q4_0` | 512 | 1787 | 104 |
| **`q4_0`** | **1024** | **2650** | **105** |
| `q4_0` | 1536 / 2048 | fails | — |
| `q8_0` | 256 | 1466 | 95 |
| `q8_0` | 512 | 1850 | 98 |
| `q8_0` | 1024 | fails | — |

---

## 6.7 CPU expert offload: the trap

`-c 130048`, `-ub 512`, `q8_0` KV:

| `-ncmoe` | Layers on CPU | PP t/s | TG t/s |
| --- | --- | --- | --- |
| 2 | 2 / 40 | 826 | 88 |
| 4 | 4 / 40 | 541 | 82 |
| 6 | 6 / 40 | 421 | 76 |
| 8 | 8 / 40 | 344 | 60 |

Against the all-GPU best of 2650 PP, `-ncmoe 2` costs **69% of your prefill for 5% of the
model**. Prefill activates nearly all 256 experts per batch; generation activates 8 — so the
CPU becomes the bottleneck while both GPUs idle.

**`-ncmoe` is a last resort, not a tuning knob.**

---

## 6.8 The split boundary

At small context, more layers on the smaller card kept helping (2244 → 2541 → 2883 for
25% → 31% → 36%). At 130k the memory limit arrives first:

| `-ts` | Result |
| --- | --- |
| 11,30 | fails |
| **12,29** | **works** |
| 13,28 | OOM |
| 14,27 | OOM |

`11,30` fails despite putting *less* on the small card — it moves that layer onto the GPU
driving the display, which has less free memory. **The optimum is a memory boundary, not a
compute one. Start from free VRAM, not capacity.**

Note that proportional-to-free-memory arithmetic suggests `15,26` here, three layers above
the measured optimum, because compute buffers do not split by the `-ts` ratio — every device
carries a near-full one. Sweep *downward* from the arithmetic suggestion.

---

## 6.9 Still open

| Experiment | Why it matters |
| --- | --- |
| **Rebuild with `-DGGML_SCHED_MAX_COPIES=1`** | Makes the lean path deterministic instead of accidental; likely enables `q8_0` at `ub 1024` at full 130k — best of both worlds. **Top priority** |
| ~~Rebuild with `-DGGML_CUDA_FA_ALL_QUANTS=ON`~~ **DONE** | Measured at ×30.5 prefill and then free — see [chapter 14](14-build-experiment.md) |
| `--spec-type draft-mtp`, `n-max 1…6` | The model ships an MTP head; ~1.17× claimed on 35B-A3B. Costs ~530 MiB. See [chapter 9](09-speculative-decoding.md) |
| Harder quality gates for `q4_0` | Needle retrieval passed; reasoning and code-edit fidelity untested (§6.4) |
| Prompt-caching measurement on real agent traffic | Per §5.5, this dominates everything else |

---

## 6.10 Complete measured table

Synthetic (`llama-batched-bench`, 8192-token prompt) unless marked *real*.

| Config | `-c` | PP t/s | TG t/s | Peak VRAM |
| --- | --- | --- | --- | --- |
| `llama-bench` ts 10/18 f16 ub512 | 4224 | 2883 | 117 | — |
| `q4_0` ub1024 (run 1) | 130048 | **2757** | 106 | 23456 |
| ub512 (small ctx) | 8448 | 2651 | 105 | 22109 |
| ub1024 (small ctx) | 8448 | 2615 | 106 | 22786 |
| `q4_0` ub1024 (run 2) | 130048 | 2599 | 105 | 23101 |
| `q4_0` ub1024 (run 3) | 130048 | 2594 | 105 | 23203 |
| `q8_0` ub512 | 126976 | 2323 | 109 | 23281 |
| `q8_0` ub1024 | 81920 | 2304 | 104 | 23225 |
| `q8_0` ub1024 | 98304 | 2296 | 103 | 23301 |
| `q8_0` ub512 | 122880 | 2225 | 106 | 23419 |
| `q8_0` ub512 | 129024 | 2101 | 106 | 23367 |
| ub256 (small ctx) | 8448 | 2122 | 99 | 21642 |
| `q8_0` ub512 | 130048 | 1850 | 98 | 23371 |
| `q4_0` ub512 | 130048 | 1787 | 104 | 23521 |
| ***real*** `q4_0` ub1024, 108k prompt | 130048 | **1758** | **46.6** | — |
| `q8_0` ub256 | 65536 | 1764 | 105 | 22565 |
| `q8_0` ub256 | 98304 | 1664 | 103 | 23097 |
| `q4_0` ub256 | 130048 | 1616 | 102 | 23150 |
| ***real*** `q8_0` ub512, 108k prompt | 130048 | **1414** | **47.5** | — |
| `q8_0` ub256 | 130048 | 1466 | 95 | 23617 |
| ub128 (small ctx) | 8448 | 1368 | 106 | 21584 |
| `q8_0` ub1024 | 114688 | 1067 | 57 | 23504 (spilled) |
| `q8_0` ub1024 | 122880 | 1036 | 78 | 23568 (spilled) |
| `q8_0` ub512 `-ncmoe 2` | 130048 | 826 | 88 | — |
| `q8_0` ub512 `-ncmoe 4` | 130048 | 541 | 82 | — |
| `q8_0` ub512 `-ncmoe 6` | 130048 | 421 | 76 | — |
| `q8_0` ub512 `-ncmoe 8` | 130048 | 344 | 60 | — |
| `q4_0`K/`q8_0`V ub512 | 130048 | 160 | 25 | 23158 |
| `q5_1`/`q5_1` ub512 | 130048 | 107 | 19 | 23086 |

---

Next: [Chapter 7 — Recommended configurations](07-recommended-configs.md).

