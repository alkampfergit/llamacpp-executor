---
name: tuning-llamacpp-configs
description: Find the fastest raw llama.cpp / llama-server configuration for a GGUF model on the GPUs actually present, deciding context size, ubatch, tensor-split, KV cache type and CPU offload from measurement instead of guesswork. Use this whenever the user asks how to run, load, host or speed up a local GGUF model, asks which parameters or flags to use, asks how much context fits in VRAM, wants to benchmark a model, mentions llama-bench / llama-batched-bench / llama-fit-params / llama-server, reports that a local model is slower than expected, or hits CUDA out-of-memory while loading — even if they never say "tune" or "benchmark". Do NOT use for choosing which model to download, for prompt engineering, or for Ollama / LM Studio / vLLM-specific settings.
argument-hint: "[path to .gguf]"
shell: powershell
allowed-tools: PowerShell Bash Read Write Edit Glob Grep
---

# Tuning a llama.cpp serving configuration

If a `.gguf` path was passed with the command, tune that model. Otherwise ask which model
and what the target context size is — those two answers determine everything below.

## Detected hardware

?`nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1`

Treat `memory.used` above as **already lost**. It is the desktop, browser and chat apps,
and it is usually concentrated on whichever GPU drives the display.

---

## The governing idea

VRAM is spent on exactly three things, and every knob trades between them:

| Consumer | Scales with | Knobs |
| --- | --- | --- |
| **Model weights** | quantisation | quant level, `-ngl`, `-ncmoe`, `-ot` |
| **KV cache** | `-c`, linearly | `-c`, `-ctk`, `-ctv` |
| **Compute buffers** | `-ub`, linearly | `-ub`, `-fa`, pipeline parallelism (build-time) |

Newcomers optimise the first and forget the third. The third is the most elastic — it can
exceed the size of a whole model, and `-fa off` alone can cost 9 GiB.

**Optimal is not one number.** Decide first which axis matters:

- **Prefill (PP)** — reading the prompt. Compute-bound, batch helps. Matters for long
  pasted contexts and agentic coding.
- **Generation (TG)** — writing the answer. Memory-bandwidth-bound, batch is irrelevant.
  Matters for chat responsiveness.

A change can improve one and do nothing for the other. Always report both.

---

## Procedure

Work through these in order. Do not skip to step 4 — measuring is the expensive step and
prediction eliminates most candidates for free.

### 1. Establish preconditions

On Windows, confirm the driver will not hide overflow. Since driver 536.40 a CUDA
allocation past VRAM does **not** fail — it silently spills to system RAM at ~1/5 speed.
Tell the user to set:

> NVIDIA Control Panel → Manage 3D Settings → **CUDA - Sysmem Fallback Policy** →
> **Prefer No Sysmem Fallback**

Until this is set, **every measurement is suspect**, because a slow result is
indistinguishable from an overflowing one. Say so plainly rather than tuning around it.

Then note what is holding VRAM:

```powershell
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
```

Closing GPU-using desktop apps is a **tuning action**, not housekeeping — it is the only
one that costs nothing.

### 2. Characterise the model

```powershell
./llama-fit-params.exe -m <model.gguf> -v 2>&1 |
  Select-String "n_layer|n_head|n_embd_head|n_ctx_train|n_expert|architecture|ssm_"
```

Record: layer count, `n_ctx_train` (the quality ceiling — do not exceed it), whether it is
MoE (`n_expert`), and whether it is hybrid/linear-attention (`ssm_*` keys present).

**Never trust a KV-size formula from the internet.** Hybrid models carry a fixed-size state
on most layers, so measured KV can be 4× smaller than `n_layer × n_head_kv × …` predicts.
Measure it in step 3 instead.

### 3. Predict before you measure

`llama-fit-params` answers "does this fit?" in under a second, without loading weights.
Use it to build the memory model and to eliminate candidates.

```powershell
./scripts/vram-budget.ps1 -Model <model.gguf>
```

That script discovers per-GPU free VRAM, sweeps `-c` × `-ub`, and prints which
combinations fit. Read its `model / context / compute` triple as exactly the three
consumers above.

Two rules when reading its output:

- **Reject any configuration whose emitted CLI contains `-ot "...=CPU"`.** That is the
  fitter telling you it does not fit; see the ladder below before accepting it.
- **Add ~250 MiB per device to its compute estimate.** It under-predicts. On the worked
  example it said 596 MiB where reality was 831 MiB — the difference between fitting and
  not.

### 4. Choose the candidate set

Constrain hard before spending GPU time:

- `-fa on` **always.** Never benchmark `-fa off`; it is not a real option.
- `-ngl 999`. If the model cannot fit that way, fix it via the ladder, not by leaving
  layers on the CPU.
- `-sm layer` for two GPUs.
- **KV type: a symmetric pair from the compiled set only** — `f16/f16`, `bf16/bf16`,
  `q8_0/q8_0`, `q4_0/q4_0`. Anything else falls off the flash-attention path and costs
  ~14×. See `references/known-traps.md`.
- `-ub` from {256, 512, 1024}. Below 256 collapses; above 1024 never pays.
- `-ts` starting from the **free-VRAM** ratio, then ±2 layers each way. Capacity ratio is
  the wrong starting point when one GPU drives the display.
- `-np 1` unless the user genuinely needs concurrent conversations.

### 5. Measure at the real context size

```powershell
. ./scripts/bench-harness.ps1
Probe "ub512 ts12,29" @('-ub','512','-ts','12,29','-ctk','q8_0','-ctv','q8_0',
                        '-fa','on','-ngl','999','-fit','off') -Ctx 130048 -Npp '8192'
```

Non-negotiable rules, each of which was learned by being burned:

- **Use `llama-batched-bench`, not `llama-bench`.** Only the former accepts `-c`, so only
  it allocates the KV cache the server will. `llama-bench` measures a different memory
  regime and will tell you a configuration works when it does not.
- **One fresh process per data point.** Multi-value sweeps inside one process suffer VRAM
  fragmentation; a config measured third can read 40% slower than the same config
  measured first.
- **Pass `-fit off`** so you measure what you asked for.
- **Record peak VRAM with every throughput number.** Throughput alone cannot distinguish
  fast from overflowing.
- **Append each result to disk before starting the next run.** Configurations near the
  VRAM ceiling crash; an in-memory results array loses the whole campaign.

The bundled harness enforces all five. Prefer running it over hand-rolling a loop.

### 6. Interpret honestly

- **Differences under ~8% are noise** when a GPU also drives a display. Do not conclude
  from a 3% gain without repeating.
- **Grep every log**, not just the results table, for `retrying`, `out of memory`,
  `failed to`, and `ignoring`. A run that silently fell back to a different execution
  mode is not the configuration you think you measured.
- **An OOM row is data.** It locates the boundary, which is what you are mapping.
- If a fallback path outperforms the intended path, **adopt the fallback** and make it
  explicit so the config is reproducible.

### 7. Validate over HTTP

Synthetic prefill is not the user's workload. Launch `llama-server` with the winner and
read its own per-request timings:

```
prompt eval time = ... tokens per second      <- real prefill
       eval time = ... tokens per second      <- real generation
```

Then send a **second, similar** request and look for prompt-cache reuse:

```
selected slot by LCP similarity, f_sim_best = 0.987, f_keep = 0.995
```

`f_sim_best` above 0.95 means the cache is working and the prefill number you spent the
campaign optimising has largely stopped mattering. If the line never appears, check `-np`
first — with statically divided caches each request only gets `-c / -np` tokens.

### 8. Quality-gate any aggressive KV setting

**Throughput benchmarks cannot see a quality regression.** Dropping the KV cache from `q8_0`
to `q4_0` buys real speed, and a benchmark will call that a pure win even if the model has
started losing the middle of long documents. Never adopt a lossy KV setting on speed evidence
alone.

```powershell
./scripts/needle-test.ps1 -Model <model.gguf> -Label q8 -Ctk q8_0 -Ctv q8_0 -Ub 512
./scripts/needle-test.ps1 -Model <model.gguf> -Label q4 -Ctk q4_0 -Ctv q4_0 -Ub 1024
```

It hides one unguessable fact at the *start* of a ~108k-token monotonous document and asks
for it at the end, so only genuine long-range retrieval can pass. It also yields **real**
deep-context prefill and generation numbers over HTTP, which no synthetic benchmark gives.

Two rules:

- **Use the chat endpoint.** On raw `/completion` an instruction-tuned model emits EOS
  immediately and returns nothing — which mimics total quality collapse. A "failure" that
  affects every configuration equally is a bug in the harness, not a finding.
- **Report what the test proves.** Passing means long-range *exact recall* survives. It does
  not test multi-step reasoning, summarisation faithfulness, or code-edit fidelity. Say
  "cleared for retrieval-style work", not "lossless".

When the user cares about quality, offer the pair — a `q8_0` profile and a `q4_0` profile —
with the measured speed difference stated, and let them choose. Often the best answer is
**`q8_0` with slightly less context**, which can beat `q4_0` at full context on both axes.

### 9. Decide on speculative decoding — by measurement, never by assumption

Only ever evaluate it **after** the placement/memory work above, because it competes for the
same VRAM. It cannot be measured with either bench tool (neither supports `--spec-type`), so
it needs `llama-server` over HTTP with a **generation-heavy** request — short prompt, long
output — since it never accelerates prefill.

```powershell
./scripts/mtp-test.ps1 -Model <model.gguf>       # baseline, then draft-mtp n-max 1..6
```

Judge it on **end-to-end tokens/second versus a no-drafting baseline on the same prompt.**
Draft acceptance is not the metric — 67–72% acceptance still lost 29% on a sparse MoE
(see trap 6). Rough priors:

- **Large dense model** → likely a win; the expensive per-token pass amortises the draft.
- **Sparse MoE, few active params** → likely a loss; measure before enabling.
- **`ngram-*`** → 0 VRAM, ~0 risk. Test on *edit-style* work, where output copies input.

Also check the model even has a draft head: `n_layer_all > n_layer` and
`nextn_predict_layers` in the metadata. Activating it loads that layer and costs real VRAM
(~530 MiB on the reference model), which a tight configuration may not have.

### 10. Deliver

Emit a single copy-paste launch command, grouped by intent, plus the measured PP/TG and
peak VRAM, plus what was given up. State explicitly which numbers are measured and which
are starting points.

---

## The ladder: how to free VRAM, cheapest first

When a configuration does not fit, work **down** this list. The ordering is the core
judgement this skill encodes — the bottom two look attractive and are traps.

| # | Action | Speed cost |
| --- | --- | --- |
| 1 | Close GPU-using desktop apps | **none** |
| 2 | `q8_0/q8_0` instead of `f16/f16` | ~2% |
| 3 | `q4_0/q4_0` instead of `q8_0/q8_0` (symmetric — still fast) | **negative — it is faster**, but **quality-gate it** |
| 4 | Reduce `-c` a little (e.g. −2%) | **negative — prefill speeds up** |
| 5 | `-ub` down one step | ~−20 to −30% prefill |
| 6 | Rebuild `-DLLAMA_SCHED_MAX_COPIES=1` | **negative — it is faster** |
| 7 | `-ncmoe N` (experts to CPU) | **~−69% prefill for 2 of 40 layers** |

Steps 3, 4 and 6 are counter-intuitive and are usually the answer:

- **A smaller KV cache can be faster**, because it frees the compute-buffer headroom that
  lets you raise `-ub`. On the reference box `q4_0/q4_0` enabled `-ub 1024` and took prefill
  from 1850 to 2650 t/s. Only ever use a *symmetric compiled pair* (see traps).
- **Context taxes prefill near the memory ceiling.** Dropping 2.4% of the window bought 26%
  more prefill. Right-sizing `-c` gives memory back *and* speeds prefill up.
- **Pipeline parallelism is a loss on mismatched GPUs** — the pipeline stalls on the slower
  card, and its 4 activation copies are often exactly what breaks the fit. But note:
  `GGML_SCHED_MAX_COPIES` is a **compile-time** define, **not** an environment variable.
  Setting `$env:GGML_SCHED_MAX_COPIES` does nothing. Either rebuild with
  `-DLLAMA_SCHED_MAX_COPIES=1`, or raise `-ub` until the pipelined reserve fails and
  llama.cpp falls back to the lean path on its own (see traps).

---

## Standing cautions

- **`GGML_SCHED_MAX_COPIES` is not a runtime environment variable.** It is a compile-time
  `#define` (default 4). Never tell the user to `$env:`-set it, and never credit a
  measurement to it. To control it: `cmake -B build -DGGML_CUDA=ON -DLLAMA_SCHED_MAX_COPIES=1`.
- **Speculative decoding must be justified per model, never assumed.** On a sparse MoE it
  measured a **net loss** (−7% at `n-max 1`, −29% at `n-max 2`) *despite 67–72% draft
  acceptance* — each drafted token costs about a full forward pass, and a ~3 B-active model
  has nothing to amortise. **High acceptance is not a win; only end-to-end tokens/second is.**
  Expect it to help on large dense models and to hurt on sparse MoE. `ngram-*` drafters cost
  no VRAM and measured ~0%, so they are a safe bet but not a free win.
- **Commit benchmark inputs, don't just regenerate them.** A fixed long-context prompt must be
  byte-identical across configurations and machines; line-ending normalisation alone changes
  its token count. Store the artifact and mark it `-text` in `.gitattributes`.
- **Verify whether a knob is runtime or compile-time before believing a result you attribute
  to it.** A gain smaller than the machine's noise floor is not evidence of anything.
- `-mg` does nothing with `-sm layer`. Delete it if you see it.
- `failed to fit params ... n_gpu_layers already set by user` is **not an error** — you set
  `-ngl`, so the fitter stood down. Look for `model loaded` below it.
- `CUDA error: the resource allocation failed` at `cublasCreate` **is** out-of-memory
  wearing a different hat.
- `model has unused tensor blk.N... -- ignoring` on an MTP/draft model is expected; those
  weights cost nothing until a `--spec-type` activates them, and then they cost real VRAM.
- Do **not** assume a smaller quant is faster. Q3_K CUDA kernels are slower than Q4_K, so
  a smaller file can generate more slowly.
- Speculative decoding (`--spec-type`) cannot be measured with either bench tool. It needs
  `llama-server` over HTTP, and its benefit is workload-dependent.

---

## References

Load only what the current question needs.

- `references/parameter-effects.md` — measured curves for `-ub`, `-c`, `-ts`, `-ncmoe`, and
  what each knob buys on which axis.
- `references/known-traps.md` — the five silent failures: sysmem fallback, KV kernel
  pairs, CPU expert offload, pipeline parallelism, and the `-np` context divisor.
- `references/worked-example.md` — a full campaign on an RTX 3070 + 5060 Ti with a 20.2 GiB
  MoE model, from first probe to final config.

## Bundled scripts

- `scripts/vram-budget.ps1` — hardware discovery, `-c` × `-ub` feasibility grid, candidate
  shortlist. **Run it; do not read it.**
- `scripts/bench-harness.ps1` — dot-source, then call `Probe`. Fresh process per run, peak
  VRAM sampling, append-only results, full log per run.
- `scripts/needle-test.ps1` — long-context quality gate over HTTP. Run it before adopting
  any lossy KV setting, and to get real deep-context throughput.

Requires: llama.cpp CUDA build with `llama-fit-params.exe`, `llama-batched-bench.exe` and
`llama-server.exe`; `nvidia-smi` on PATH; PowerShell 7+.
