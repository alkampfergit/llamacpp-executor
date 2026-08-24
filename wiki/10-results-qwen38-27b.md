# Chapter 10 — Measured Results: Qwen3.8-27B (dense, hybrid, MTP)

Every number here was measured on the machine in the [README](README.md), one fresh process
per data point. Raw data: [`benchmarks/results.tsv`](benchmarks/results.tsv),
[`benchmarks/deep-results.md`](benchmarks/deep-results.md),
[`benchmarks/needle-results.md`](benchmarks/needle-results.md),
[`benchmarks/mtp-results.md`](benchmarks/mtp-results.md), logs in
[`benchmarks/logs/`](benchmarks/logs/).

Model: **`Qwen3.8-27B-Q4_K_M.gguf`**, 15.66 GiB, 27.32 B params, architecture `qwen35`.
Build `fe8156f78` (10509), prebuilt Windows CUDA.

`PP` = prefill tokens/second, `TG` = generation tokens/second. Differences under ~8% are
noise (§5.7).

> **Two kinds of number appear below.** *Synthetic* figures come from
> `llama-batched-bench` with an 8192-token prompt. *Real* figures come from `llama-server`
> over HTTP with a **107,763-token** prompt — the same haystack chapter 6 used. They differ
> a lot, and the real ones are the ones that predict your wait. Both are labelled.

> **This chapter is deliberately not a copy of chapter 6.** Almost every conclusion there came
> from a *sparse MoE* with ~3 B active parameters. This model is *dense*. Four of chapter 6's
> headline findings do not survive the change, and one of its open questions is now answered
> with a 2x speedup. §10.11 lists the contradictions in one table.

---

## 10.1 The headline

| | Synthetic PP | Real PP @108k | Shallow TG | TG @108k | Peak VRAM |
| --- | --- | --- | --- | --- | --- |
| `q8_0` KV, 130k, **no drafting** | **1208** | 830 | 22.5 | 14.7 | 22453 |
| **`q8_0` KV, 130k, `draft-mtp` n4** | — | 600 | **47.5** | **21.0** | 23560 |
| `q4_0` KV, 130k, `draft-mtp` n4 | 1187 | 607 | **47.4** | **25.7** | 22893 |
| `q4_0` KV, **192k**, no drafting | 1211 | — | 22.0 | — | 22126 |
| `f16` KV — best that fits | 1243 @ **72k** | *haystack does not fit* | 22.5 | — | 22472 |

Three things to take away before any of the detail:

- **≥130k with everything on the GPU: yes, comfortably.** `-c 130048` with `q8_0` KV peaks at
  22453 MiB against 23215 MiB free — about 760 MiB of headroom. `q8_0` even reaches **136k**
  (§10.5). Priority achieved.
- **`--spec-type draft-mtp` is worth ~2x generation on this model.** 22.5 → 47.5 t/s shallow,
  14.7 → 21.0 t/s at 108k depth. On the MoE of chapter 9 the identical flag was a **loss**.
  This is the most important result in the chapter; §10.7 explains why the two outcomes are
  consistent rather than contradictory.
- **`f16` KV is not affordable here, and that is the opposite of what I predicted.** This
  model's KV cache is *three times larger* than the 20.2 GiB MoE's, not smaller. §10.2.

---

## 10.2 Why the KV cache is bigger, not smaller

The model looks like it should be easy. It is 15.66 GiB — 4.5 GiB *smaller* than the MoE of
chapter 6 — and it is hybrid (`ssm_d_conv`, `ssm_d_state`, `ssm_n_group` are all present), and
chapter 4 tells you hybrid models carry far less KV than the naive formula predicts. So `f16`
KV at 130k ought to be within easy reach.

It is not. Dump the tensor list and count what is actually there:

```powershell
./llama-fit-params.exe -m Qwen3.8-27B-Q4_K_M.gguf -v 2>&1 |
  Select-String 'create_tensor: loading tensor blk\.\d+\.(attn_k|ssm_conv1d)\.weight'
```

| | Count | Layer indices |
| --- | --- | --- |
| SSM (linear-attention) layers | **48** | everything else |
| **Full-attention layers** | **16** | 3, 7, 11, … 63 — every 4th |
| MTP / next-token-prediction head | 1 | `blk.64` (`n_layer_all` = 65) |

So one layer in four is a *real* attention layer, and each of those carries a real KV cache
with `n_head_kv = 4` and `n_embd_head_k = 256` — that is `n_embd_k_gqa = 1024`, which is wide.
The arithmetic that matters:

```
KV per 1024 tokens = 16 layers x 1024 x 2 (K and V) x bytes_per_element
                   = 64 MiB (f16)    34 MiB (q8_0)    18 MiB (q4_0)
```

| `-c` | `f16` KV | `q8_0` KV | `q4_0` KV |
| --- | --- | --- | --- |
| 65536 | 4096 MiB | 2176 MiB | 1152 MiB |
| **130048** | **8128 MiB** | **4318 MiB** | **2286 MiB** |
| 196608 | 12288 MiB | 6528 MiB | 3456 MiB |

Compare with chapter 6: the 20.2 GiB MoE needed **1422 MiB** for `q8_0` KV at the same 130k.
This model needs **4318 MiB** — 3.0x more, from a file 4.5 GiB smaller.

> **The lesson is not "hybrid models have small KV caches."** It is *count the attention layers
> and multiply*. "Hybrid" tells you that *some* layers are cheap; it tells you nothing about how
> expensive the remaining ones are. Here 25% of the layers carry a wide GQA cache, and 25% of a
> large number is still a large number.

The practical consequence: **`f16` KV cannot serve a long context on this box.** Measured
ceilings are in §10.5 — `f16` tops out around 72–82k, which is not even enough to hold the
107,763-token quality-gate prompt. If you want ≥100k here, `q8_0` is the *most* precise KV you
can have. That is a hard constraint, not a preference.

---

## 10.3 `-ub` does almost nothing

Chapter 6 found `-ub` to be the most valuable single knob on the MoE: 1368 → 2651 t/s going
from 128 to 512. Here, at a small context where memory is not the constraint:

| `-ub` | PP t/s | TG t/s | Peak VRAM |
| --- | --- | --- | --- |
| 256 | 1141 | 22.21 | 17518 |
| 512 | 1202 | 22.20 | 17907 |
| 1024 | **1222** | 22.20 | 18688 |
| 2048 | 1145 | 22.18 | 20274 |

The whole 256 → 2048 range spans 7%, which is inside the noise floor. Only at `-ub 128` does
it finally bite: **939 t/s at 130k**, −22%.

**Why the difference.** A dense 27 B model reads all 15.66 GiB of weights for every ubatch and
does roughly 9x the arithmetic per token that a 3 B-active MoE does. Prefill is therefore
*compute*-bound almost immediately, and a bigger micro-batch has nothing left to amortise. On
the MoE, prefill was bound by weight streaming and expert routing, which batching helps a lot.

> **This is a gift, not a curiosity.** On the MoE, `-ub` was a three-way fight between prefill
> speed, KV size and compute buffers. Here `-ub 512` costs ~540 MiB more than `-ub 128` and buys
> 22% prefill, and going *above* 512 buys nothing at all. Set it to 512 and spend the rest of
> your VRAM on context. Do not drop to 256 or 128 hoping to buy context — KV is only 34 MiB per
> 1024 tokens at `q8_0`, so the trade is bad in both directions.

---

## 10.4 KV quantisation buys VRAM here — and nothing else

Chapter 6's most-quoted number was that `q4_0` KV took prefill from 1850 to 2650 t/s, a 43%
win, because the freed memory allowed `-ub 1024`. That mechanism does not exist here, because
`-ub 1024` is worth nothing (§10.3). What is left is pure memory savings.

At `-c 130048`, `-ub 512`, `-ts 22,43`:

| `-ctk`/`-ctv` | KV size | Synthetic PP | Synthetic TG | Real PP @108k | Real TG @108k | Peak VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| `f16`/`f16` | 8128 MiB | **does not fit** | — | — | — | — |
| **`q8_0`/`q8_0`** | 4318 MiB | **1208** | 22.14 | **829** | **14.7** | 22596 |
| **`q4_0`/`q4_0`** | 2286 MiB | 1187 | 22.02 | 835 | 14.2 | 21348 |

(Peak VRAM in that column is the synthetic run's. Under `llama-server` the same `q8_0`
configuration peaked slightly lower, at 22453 MiB — §10.7.)

**`q4_0` is not faster than `q8_0` on this model — not synthetically, not at depth, not for
generation.** Every pair of numbers above sits inside the noise floor. What `q4_0` buys is
**2032 MiB**, and nothing else.

> **So the chapter-6 ladder step "use `q4_0`, it is faster" does not apply here.** Reach for
> `q4_0` only when you want the VRAM for something specific — a longer context (§10.5) or room
> for the MTP draft graph (§10.7). If you do not need the memory, keep `q8_0`: it is free.

The symmetric-pair rule from §6.3 still holds and was not re-tested — `f16/f16`, `bf16/bf16`,
`q8_0/q8_0`, `q4_0/q4_0` are the only compiled flash-attention pairs, and anything else costs
about 14x. Nothing about this model changes the CUDA kernel set.

---

## 10.5 How much context actually fits

`-ub 512`, `-fa on`, `-ngl 999`, `-sm layer`. Free VRAM at the time: 8018 MiB (3070) +
15023 MiB (5060 Ti) = 23215 MiB.

| KV | `-c` | `-ts` | Result | Synthetic PP | Peak VRAM |
| --- | --- | --- | --- | --- | --- |
| `f16` | 73728 | 22,43 | works | 1243 | 22472 |
| `f16` | 81920 | 22,43 | works, thin | 1244 | 23041 |
| `f16` | 90112 | 22,43 | works, **saturated** | 977 | 23554 |
| `q8_0` | **130048** | **22,43** | **works** | **1208** | **22596** |
| `q8_0` | 139264 | 21,44 | works, thin | 1215 | 22758 |
| `q8_0` | 147456 | 22,43 | fails to allocate | — | — |
| `q8_0` | 163840 | 22,43 | fails to allocate | — | — |
| `q4_0` | 130048 | 22,43 | works | 1187 | 21348 |
| `q4_0` | **196608** | 22,43 | **works** | **1211** | 22126 |

Read the three `f16` rows as one story: 72k is fine, 80k is thin, and by 90k the display GPU is
at 15923 / 16311 MiB and prefill has dropped 21%. That last row is the §5.3 saturation signature
caught live — **not a kernel effect, an overflow.** It is exactly why peak VRAM belongs next to
every throughput number.

Practical ceilings on this box:

| KV type | Largest context that fits with margin |
| --- | --- |
| `f16`/`f16` | **~72k** |
| `q8_0`/`q8_0` | **~130k** (136k works, thinly) |
| `q4_0`/`q4_0` | **~192k** |

`n_ctx_train` is 262144, so none of these is a model-quality limit; all three are pure VRAM
limits.

**And there is no context tax.** 1208 t/s at 130048 versus 1202 t/s at 8448 — prefill speed is
flat across the whole window. Chapter 6 measured a steep tax near the ceiling (2.4% less context
bought 26% more prefill); that was memory pressure interacting with `-ub`, and since `-ub` is
inert here, so is the tax.

---

## 10.6 The split, and pipeline parallelism

`vram-budget.ps1` suggested `-ts 23,42` from the free-VRAM ratio. As in §6.8, the measured
optimum sits *below* the arithmetic suggestion, and the boundary is memory, not compute:

| `-ts` | `-ub` | Result | Synthetic PP | Peak V0 / V1 |
| --- | --- | --- | --- | --- |
| 24,41 | 512 | **fails** — device 0 cannot fit the compute buffer | — | — |
| 23,42 | 256 | **fails** — device 0 cannot fit the compute buffer | — | — |
| **22,43** | **512** | **works, non-pipelined** | **1208** | **7421 / 15175** |
| 22,43 | 256 | works, non-pipelined | 1147 | 7523 / 15224 |
| 21,44 | 512 | works, pipelined | 1219 | 7597 / 15784 |
| 20,45 | 512 | works, pipelined, saturated | 943 | 7387 / **15987** |

Two separate things are going on, and it is worth pulling them apart.

**The RTX 3070 is capacity-bound at 22 layers.** At `-ts 23,42` and above, device 0 has room for
the weights but not for its compute buffer, and the load fails outright:

```
allocating 823.09 MiB on device 0: cudaMalloc failed: out of memory
graph_reserve: failed to allocate compute buffers
sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
allocating 611.77 MiB on device 0: cudaMalloc failed: out of memory
llama_init_from_model: failed to initialize the context
```

Note the shape of that log: the *pipelined* reserve fails, llama.cpp retries lean, and the lean
one fails too. When only the first one fails, you get a working non-pipelined context — which
is what `-ts 22,43` gives.

**But pipeline parallelism is no longer the deciding factor.** `-ts 22,43` takes the lean
fallback and gives 1208; `-ts 21,44` keeps the pipeline and gives 1219. Those are the same
number. Contrast §6.2, where the fallback was worth 43%. Prefill here is compute-bound (§10.3),
so removing a pipeline stall changes nothing measurable.

What the fallback still buys is **memory**: the non-pipelined path allocates a 715 MiB device-0
compute buffer instead of 1137 MiB. That is why `-ts 22,43` peaks 785 MiB *lower* than
`-ts 21,44` while putting *more* work on the small card.

> **Recommendation: `-ts 22,43`.** Same speed as the alternatives, the lowest peak VRAM of any
> working split, and one layer clear of the cliff at 23,42. The 943 t/s row at `-ts 20,45` is
> not a pipeline penalty — it is the display GPU at 15987 / 16311 MiB, the same overflow visible
> in the `f16` 90k row of §10.5.

`GGML_SCHED_MAX_COPIES` is still a compile-time define and still does nothing as an environment
variable (§6.2). It was not set for any run here.

---

## 10.7 `--spec-type draft-mtp`: a 2x generation win

This is the finding the chapter exists for.

**The prediction going in.** Chapter 9 measured `draft-mtp` on the MoE and it *lost*: −7% at
`n-max 1`, −29% at `n-max 2`, despite 67–72% draft acceptance. The explanation offered there was
that speculative decoding pays for itself by amortising an expensive target forward pass, and a
3 B-active MoE has no expensive forward pass to amortise. If that explanation is right, then a
**dense** model with an MTP head is exactly the case that should win. The vendor claims 1.73x on
the 27 B dense against 1.17x on the MoE — the same shape of claim.

**The measurement.** `mtp-test.ps1` (`.claude/skills/tuning-llamacpp-configs/scripts/`), `-c 65536`, `-ub 512`, `q8_0` KV, `-ts 22,43`,
400-token code-shaped generation, one fresh server per point:

| `--spec-draft-n-max` | TG t/s | vs baseline | Draft acceptance | Mean accepted run |
| --- | --- | --- | --- | --- |
| *baseline (none)* | **22.4 – 22.6** | — | — | — |
| 1 | 35.2 | **+56%** | 0.891 | 1.89 |
| 2 | 41.1 | **+82%** | 0.901 | 2.80 |
| 3 | 45.0 | **+100%** | 0.805 | 3.40 |
| **4** | **47.1** | **+110%** | 0.739 | 3.95 |
| 5 | 38.1 | +70% | 0.594 | 3.95 |
| 6 | 44.4 | +98% | 0.686 | 5.12 |
| 8 | 47.2 | **+111%** | 0.522 | 5.12 |

The baseline reproduced at 22.42 and 22.55 t/s across three separate campaigns, so the ~2x is
not a baseline artefact. The curve rises steeply to `n-max 3–4` and then plateaus; the n5 row is
7% low and n6 is 6% low, both within the noise floor of a machine that is also driving a display.

> ### ⚠️ The SHAPE of this curve is not trustworthy — see [chapter 15 §15.6](15-mmvq-ampere-experiment.md)
> Every row above is **one run at `temperature 0.6`**. Chapter 15 measured the run-to-run spread
> of exactly this harness and configuration and found **±20%**, with
> **r² = 0.872 between throughput and draft acceptance** — because a different sampled token
> sequence accepts a different fraction of drafts. The `~2x` headline and the baseline are
> solid (the baseline has now reproduced six times across two binaries, 21.95–22.68 t/s). The
> per-`n-max` ordering, the n5 dip, and the apparent optimum at n4/n8 are **sampling luck**.
> Re-measured at greedy on build 10585, n6/n7/n8 came out 55.7 / 53.2 / 54.9 — flat within
> noise. **Do not pick `n-max` off this table**; if it matters, re-run with `-Temperature 0`.

**Acceptance falls as `n-max` rises, and that is fine.** From n1 to n8, acceptance drops from 89%
to 52% while the mean accepted run grows from 1.9 to 5.1 tokens. End-to-end throughput — the only
thing that matters — plateaus. This is exactly why the skill insists you judge drafting on
tokens/second and never on acceptance rate: the n1 configuration has the *best* acceptance in the
table and the *worst* throughput.

**Verify the head actually loaded.** With drafting off, the log says this fifteen times:

```
model has unused tensor blk.64.attn_q.weight (size = 35389440 bytes) -- ignoring
```

With `--spec-type draft-mtp` those lines vanish and you get instead:

```
common_speculative_init_result: creating MTP draft context against the target model
slot print_timing: draft acceptance = 0.73945 ( 298 accepted / 403 generated), mean len = 3.95
```

> **A harness trap.** `mtp-test.ps1` checks for `unused tensor blk.40` to decide whether the MTP
> head loaded — correct for the 40-layer MoE it was written against, wrong here, where the head
> is `blk.64`. Its "MTP loaded" column therefore reads `True` even for the baseline row. Grep the
> log for your own model's last block index instead of trusting that column.

### What it costs

Nothing in quality — speculative decoding is exact. The target model verifies every drafted
token, so the output distribution is unchanged, and the needle test passed on all seven MTP
configurations tested. But it costs two real things:

| Cost | Measured |
| --- | --- |
| **Prefill** | 830 → 596 t/s at 108k depth, **−27%** |
| **VRAM** | 22453 → 23560 MiB, **+1107 MiB** |

The prefill penalty is intrinsic, not an overflow artefact: the `q4_0` run has 670 MiB more
headroom than the `q8_0` run and prefills at 607 versus 596 t/s — the same. The MTP head has to
be run across the whole prompt too.

The VRAM cost is **not** the draft KV cache, which is what I assumed. The logs show a *second*
graph reservation, for the draft context, with its own compute buffers:

```
allocating 710.06 MiB on device 1: cudaMalloc failed: out of memory
sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
```

Two attempts to shrink it both failed, and both are worth recording so nobody repeats them:

| Attempt | Peak VRAM | Effect |
| --- | --- | --- |
| baseline MTP, `-ub 512` | 23560 | — |
| `-ctkd q8_0 -ctvd q8_0` (quantise the *draft* KV) | 23663 | **none** |
| `-ub 256` | 23527 | none (−33 MiB) |
| `-ub 128` | 23459 | none (−101 MiB) |

Without drafting, `-ub 128` saves 542 MiB versus `-ub 512`. With drafting it saves 101 MiB — the
draft graph expands into whatever the main graph gives up. **There is no knob that makes MTP
cheap.** Budget 1.1 GiB and plan the context around it.

### Real deep-context numbers

`llama-server` over HTTP, `-c 130048`, `-ts 22,43`, the 107,763-token haystack, long answer:

| Config | Shallow TG | Real PP @108k | Real TG @108k | Needle | Peak VRAM |
| --- | --- | --- | --- | --- | --- |
| `q8_0`, no drafting | 22.5 | **830** | 14.7 | PASS | **22453** |
| `q8_0`, `draft-mtp` n4 | 44.8 / 47.5 | 596 / 604 | 20.9 / 21.0 | PASS | 23560 / 23579 |
| `q8_0`, `draft-mtp` n4, `-ub 256` | 45.9 | 595 | 22.5 | PASS | 23527 |
| `q4_0`, `draft-mtp` n4 | 47.4 | 607 | **25.7** | PASS | 22893 |
| `q8_0`, `draft-mtp` n4, `-c 114688` | 44.1 | 615 | 20.9 | PASS | 23317 |

Two honest caveats on that table:

- **The gain shrinks with depth.** Shallow it is ~2.0x; at 108k it is 1.42–1.53x. Attention cost
  grows with context while the drafting overhead stays constant, so amortisation gets worse the
  deeper you are. The advertised 1.73x is a shallow-context number.
- **The `q4_0` row's 25.7 t/s is partly luck.** Temperature was 0, but changing the KV precision
  changes the logits, which changes the generated text, which changes what the drafter can
  predict: acceptance was 0.597 there versus 0.461 on the `q8_0` runs. Treat the `q8_0` / `q4_0`
  deep-TG gap as an artefact of different output text, **not** as evidence that `q4_0` KV decodes
  faster — §10.4 shows it does not.

---

## 10.8 `ngram-simple`: nothing, again

Same harness, same context, `q8_0` KV:

| Config | TG t/s | vs baseline |
| --- | --- | --- |
| baseline | 22.42 | — |
| `ngram-simple` n4 | 22.33 | −0.4% |
| `ngram-simple` n8 | 22.35 | −0.3% |

Identical to the MoE result in chapter 9: no gain, no loss. An n-gram drafter can only predict
text that literally repeats earlier in the window, and a freshly written Python module mostly does
not. It costs no VRAM, so it is harmless — but on a model that ships a real MTP head there is no
reason to choose it.

---

## 10.9 The quality gate

Method and haystack exactly as §6.4 — the same byte-identical
[`benchmarks/needle-prompt.txt`](benchmarks/needle-prompt.txt), `CRIMSON-PELICAN-4417` planted in
RECORD 0001 and asked for after ~108k tokens of near-duplicate telemetry. Chat endpoint,
`--temp 0`, `enable_thinking = false`.

| Config | `-c` | Needle | Real PP | Real TG | Wall |
| --- | --- | --- | --- | --- | --- |
| `q8_0`/`q8_0`, `ub 512` | 130048 | **PASS** | 829.2 | 14.7 | 130.9 s |
| `q4_0`/`q4_0`, `ub 512` | 130048 | **PASS** | 834.5 | 14.2 | 130.1 s |
| all seven `draft-mtp` variants of §10.7 | 130048 / 114688 | **PASS** | — | — | — |

Both KV precisions retrieved the passphrase exactly, and so did every MTP configuration.

> **Read this exactly as narrowly as chapter 6 did.** It shows that neither `q8_0` nor `q4_0` KV
> destroys long-range *exact recall* at 108k, and that MTP drafting does not corrupt output
> (which theory already guarantees — the target model verifies every token). It does **not** test
> multi-step reasoning over long context, summarisation faithfulness, or code-edit fidelity.
> Cleared for retrieval-style work; not certified lossless.
>
> There is a second, quieter reason to prefer `q8_0` that the needle test cannot see: `q4_0` is
> the only precision that reaches 192k (§10.5), and there is no quality evidence at all beyond
> 108k. Do not assume the pass transfers.

An observation worth more than the test itself: **`f16` KV cannot run this gate at all.** Its
ceiling is ~82k and the prompt is 107,763 tokens. If a configuration cannot hold your quality
gate's prompt, it cannot serve your workload either.

---

## 10.10 Prompt caching still dominates

Per §5.5 and skill step 7, the second identical request:

```
slot get_availabl: selected slot by LCP similarity, f_sim_best = 1.000 (> 0.100 thold), f_keep = 0.998
```

Measured on all seven deep runs without exception: `f_sim_best = 1.000`, and the re-prefill was
**4 tokens** instead of 107,763. A 130-second first request became roughly a 14-second one.

> **Put that next to §10.7's −27% prefill.** The MTP prefill penalty is paid once per *new*
> context. In an agentic loop where each turn appends to the same conversation, you pay it on the
> first turn and then generate at double speed for every turn after. That is what makes MTP an
> easy call despite the prefill cost — and it is why `-np 1` matters: with `-np 2` each slot gets
> only `-c / 2` tokens (see `references/known-traps.md`).

---

## 10.11 What contradicted chapter 6

| Chapter 6 finding (MoE, 20.2 GiB, ~3 B active) | On this model (dense, 15.66 GiB, 27 B active) |
| --- | --- |
| Hybrid ⇒ KV cache far smaller than the formula predicts | **KV is 3.0x larger** than the MoE's at the same context — 16 wide-GQA attention layers (§10.2) |
| `-ub` is the most valuable knob: 1368 → 2651 t/s | **Flat from 256 to 2048.** Only `-ub 128` costs anything (§10.3) |
| `q4_0` KV is *faster* than `q8_0` (+43% prefill) | **Identical speed.** `q4_0` buys 2032 MiB and nothing else (§10.4) |
| The non-pipelined fallback is worth +43% prefill | **Worth 0% speed**, but still worth 422 MiB of compute buffer (§10.6) |
| Dropping 2.4% of context bought 26% more prefill | **No context tax at all** — 1208 t/s at 130k vs 1202 at 8k (§10.5) |
| `--spec-type draft-mtp` is a net loss (−7% / −29%) | **+56% to +111%.** The mechanism is confirmed, not refuted (§10.7) |
| `-ncmoe` is a trap costing 69% of prefill | **Not applicable** — `n_expert = 0`, there are no experts to offload |

The unifying explanation for the first five rows is the same one that explains the sixth: a dense
27 B forward pass is roughly 4–5x more expensive than a 3 B-active MoE one. That makes prefill
compute-bound (so batching and pipelining stop mattering), makes generation
memory-bandwidth-bound at 22 t/s instead of 105 (so drafting finally has something large to
amortise), and leaves the KV cache — whose size is set by attention geometry, not by parameter
count — as the dominant elastic consumer.

**Chapter 6 was not wrong. It was specific.** The transferable part of it is the *method*: one
fresh process per point, `llama-batched-bench` because `llama-bench` has no `-c`, peak VRAM
beside every throughput number, and a quality gate before adopting anything lossy. Every *number*
in it belongs to that model.

---

## 10.12 Recommended launch commands

All four were measured as written. `-fit off` is deliberate — it makes llama.cpp serve exactly
what you asked for instead of re-fitting.

### A. Daily driver — 130k, `q8_0` KV, MTP on

Best all-round: 47 t/s shallow, 21 t/s at 108k depth, the full 130k window, and `q8_0` KV.
**Caveat: peak 23560 MiB puts the display GPU at ~15.9 / 16.3 GiB.** It ran clean five times out
of five, but there is no room for a browser starting up on device 1. If that worries you, use
profile B.

```powershell
.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf" `
  --host 127.0.0.1 --port 8080 `
  -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 22,43 `
  -fa on -b 2048 -ub 512 `
  -ctk q8_0 -ctv q8_0 `
  --spec-type draft-mtp --spec-draft-n-max 4 `
  -fit off
```

### B. Same speed, real headroom — 130k, `q4_0` KV, MTP on

670 MiB more margin than A, identical shallow speed, needle PASS. The trade is KV precision,
which §10.4 shows costs nothing in *speed* — you are trading only the precision itself.

```powershell
.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf" `
  --host 127.0.0.1 --port 8080 `
  -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 22,43 `
  -fa on -b 2048 -ub 512 `
  -ctk q4_0 -ctv q4_0 `
  --spec-type draft-mtp --spec-draft-n-max 4 `
  -fit off
```

### C. Maximum safety margin — 130k, `q8_0` KV, no drafting

Peak 22453 MiB, ~760 MiB spare. Fastest prefill of the three (830 t/s at depth) and the slowest
generation (14.7 t/s at depth). Use it when the desktop is busy, or when the workload is
prefill-heavy — bulk document ingestion rather than conversation.

```powershell
.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf" `
  --host 127.0.0.1 --port 8080 `
  -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 22,43 `
  -fa on -b 2048 -ub 512 `
  -ctk q8_0 -ctv q8_0 `
  -fit off
```

### D. Maximum context — 192k, `q4_0` KV, no drafting

Synthetic only: 1211 t/s PP, 22.0 t/s TG, peak 22126 MiB. **Unvalidated beyond 108k** — the
quality gate does not reach that far (§10.9).

```powershell
.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf" `
  --host 127.0.0.1 --port 8080 `
  -c 196608 -np 1 `
  -ngl 999 -sm layer -ts 22,43 `
  -fa on -b 2048 -ub 512 `
  -ctk q4_0 -ctv q4_0 `
  -fit off
```

### Choosing between them

| If you care most about | Use | Because |
| --- | --- | --- |
| Interactive chat / agent loops | **A** | 2x generation, and prompt caching (§10.10) hides the prefill cost after turn 1 |
| Not having to think about VRAM | **B** | Same speed as A with 670 MiB more room |
| Reading big documents once | **C** | Highest prefill, biggest margin |
| Windows above 130k | **D** | Only `q4_0` reaches there |
| The most precise KV possible | `f16` at `-c 73728` | 1243 t/s PP, 22.5 t/s TG, peak 22472 — but read §10.2 before you want this |

Add `--jinja` if you need tool calling. It was **not** enabled for any measurement here.

---

## 10.13 Still open

| Experiment | Why it matters |
| --- | --- |
| ~~**Rebuild with `-DGGML_CUDA_FA_ALL_QUANTS=ON`**~~ — **DONE, see [chapter 14](14-build-experiment.md)** | `q8_0` keys + `q4_0` values would save ~1000 MiB at 130k — almost exactly the 1.1 GiB the MTP draft graph needs. Measured: asymmetric KV goes ×30.5 and then costs nothing. `GGML_SCHED_MAX_COPIES=1`, built at the same time, independently frees ~845 MiB |
| Why the MTP draft graph needs 1.1 GiB, and whether it can be capped | `-ub` and `-ctkd` both failed to shrink it (§10.7). A build-level answer may exist |
| `draft-mtp` x `n-max` at *deep* context | The n-max sweep was run at 65k. The plateau may sit elsewhere at 108k, where amortisation is worse |
| Quality above 108k | Profile D offers 192k and nothing validates it. The haystack would need to be ~1.8x longer |
| `-ub 1024` under drafting | Untested combination; `-ub` is free without drafting, but the draft graph reacts to it unpredictably |
| Reasoning / code-edit gates for `q4_0` | The same gap chapter 6 left open, and profile B now depends on it |
| Rebuild with `-DGGML_SCHED_MAX_COPIES=1` | Still the deterministic way to reach the lean path. Worth less here than on the MoE — it buys memory, not speed (§10.6) |

---

## 10.14 Complete measured tables

### Synthetic — `llama-batched-bench`, 8192-token prompt, 128-token generation

| KV | `-c` | `-ts` | `-ub` | PP t/s | TG t/s | Peak V0 | Peak V1 | Peak total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `f16` | 81920 | 22,43 | 512 | **1244** | 22.46 | 7439 | 15602 | 23041 |
| `f16` | 73728 | 22,43 | 512 | 1243 | 22.47 | 7247 | 15225 | 22472 |
| `q8_0` | 8448 | 23,42 | 1024 | 1222 | 22.20 | 6057 | 12631 | 18688 |
| `q8_0` | 130048 | 21,44 | 512 | 1219 | 22.03 | 7597 | 15784 | 23381 |
| `q8_0` | 139264 | 21,44 | 512 | 1215 | 22.02 | 7317 | 15441 | 22758 |
| `q4_0` | 196608 | 22,43 | 512 | 1211 | 22.01 | 7485 | 14641 | 22126 |
| `q4_0` | 130048 | 20,45 | 1024 | 1209 | 21.26 | 7387 | 15574 | 22961 |
| **`q8_0`** | **130048** | **22,43** | **512** | **1208** | **22.14** | **7421** | **15175** | **22596** |
| `q8_0` | 8448 | 23,42 | 512 | 1202 | 22.20 | 5857 | 12050 | 17907 |
| `q4_0` | 130048 | 22,43 | 512 | 1187 | 22.02 | 7209 | 14139 | 21348 |
| `q8_0` | 130048 | 22,43 | 256 | 1147 | 22.13 | 7523 | 15224 | 22747 |
| `q8_0` | 8448 | 23,42 | 2048 | 1145 | 22.18 | 6467 | 13807 | 20274 |
| `q8_0` | 8448 | 23,42 | 256 | 1141 | 22.21 | 5765 | 11753 | 17518 |
| `q8_0` | 139264 | 20,45 | 512 | 1008 *(saturated)* | 20.30 | 7555 | 15919 | 23474 |
| `f16` | 90112 | 22,43 | 512 | 977 *(saturated)* | 22.08 | 7631 | 15923 | 23554 |
| `q8_0` | 130048 | 20,45 | 512 | 943 *(saturated)* | 21.99 | 7387 | 15987 | 23374 |
| `q8_0` | 130048 | 22,43 | 128 | 939 | 22.13 | 7367 | 14544 | 21911 |
| `q8_0` | 147456 | 22,43 | 512 | **fails** | — | 6789 | 13977 | 20766 |
| `q8_0` | 163840 | 22,43 | 512 | **fails** | — | 7017 | 14837 | 21854 |
| `q8_0` | 130048 | 24,41 | 512 | **fails** | — | 7343 | 13652 | 20995 |
| `q8_0` | 130048 | 24,41 | 256 | **fails** | — | 7535 | 13692 | 21227 |
| `q8_0` | 130048 | 23,42 | 256 | **fails** | — | 6737 | 11107 | 17844 |

Peak VRAM on a failed row is the high-water mark reached before the allocation gave up — it maps
the boundary, which is the point of recording it.

> **Reading `results.tsv` on this model needs care.** The harness stamps a row `OOM` whenever the
> log contains `out of memory` — and on this model the *successful* `-ts 22,43` configurations
> contain exactly that string, because the pipelined reserve fails first and the lean retry
> succeeds. Four rows are marked `OOM` and have perfectly good PP/TG numbers next to them; those
> are working configurations. A row that genuinely failed has an **empty** PP column. Always read
> the status flag together with the throughput columns, never on its own.

### Real — `llama-server` over HTTP, 107,763-token prompt, `-ts 22,43`

| Config | `-c` | Spec | Shallow TG | Deep PP | Deep TG | Cached re-prefill | Needle | Peak total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `q8_0` `ub512` | 130048 | none | 22.5 | **830** | 14.7 | 4 tok | PASS | **22453** |
| `q8_0` `ub512` | 130048 | mtp n4 | 44.8 | 596 | 20.9 | 4 tok | PASS | 23560 |
| `q8_0` `ub512` (repeat) | 130048 | mtp n4 | 47.5 | 604 | 21.0 | 4 tok | PASS | 23579 |
| `q8_0` `ub256` | 130048 | mtp n4 | 45.9 | 595 | 22.5 | 4 tok | PASS | 23527 |
| `q8_0` `ub128` | 130048 | mtp n4 | 43.6 | 544 | 22.8 | 4 tok | PASS | 23459 |
| `q8_0` `ub512` `-ctkd q8_0` | 130048 | mtp n4 | 45.3 | 586 | 20.2 | 4 tok | PASS | 23663 |
| `q8_0` `ub512` | 114688 | mtp n4 | 44.1 | 615 | 20.9 | 4 tok | PASS | 23317 |
| **`q4_0` `ub512`** | 130048 | mtp n4 | **47.4** | 607 | **25.7** | 4 tok | PASS | 22893 |
| `q4_0` `ub512` | 130048 | none | — | 835 | 14.2 | — | PASS | — |

### Method notes

- `llama-bench` was not used at all: it has no `-c`, so it never allocates the real KV cache and
  cannot answer a memory question (§5.2).
- One fresh process per data point, enforced by
  [`bench-harness.ps1`](../.claude/skills/tuning-llamacpp-configs/scripts/bench-harness.ps1);
  every row was appended to `results.tsv` before the next run started.
- `--spec-type` cannot be measured by either bench binary. Speculative-decoding numbers come from
  [`mtp-test.ps1`](../.claude/skills/tuning-llamacpp-configs/scripts/mtp-test.ps1) (shallow) and
  [`deep-probe-q38.ps1`](../.claude/skills/tuning-llamacpp-configs/scripts/deep-probe-q38.ps1)
  (deep, plus peak VRAM and prompt-cache checks), both driving `llama-server` over HTTP.
- **Unverified precondition.** The NVIDIA *CUDA — Sysmem Fallback Policy* setting could not be
  confirmed from the command line. Device-0 failures were clean OOMs, and the rows marked
  *saturated* degraded 20–25% rather than failing — consistent with, but not proof of, the §5.3
  spill. Set **Prefer No Sysmem Fallback** before trusting any run that peaks above ~23.2 GiB.
- **Fluctuating baseline.** Idle usage on the display GPU moved between 703 and 1288 MiB during
  the campaign, so peaks near 23.5 GiB are only reproducible with a quiet desktop. Free VRAM at
  campaign start: 8018 + 15023 = 23215 MiB of 24503 MiB installed.
- **Where the results landed.** The skill's bundled scripts default to a `bench-results/`
  directory next to the binaries; this campaign pointed them at `wiki/benchmarks/` (a
  directory junction, since removed) so that every row lives in the same append-only
  `results.tsv` as chapter 6's. The haystack in `benchmarks/needle-prompt.txt` was reused
  byte-for-byte, not regenerated — the script only builds it when it is absent, which is what
  makes the two chapters' quality gates comparable.
- The 107,763-token prompt is 20 tokens longer than chapter 6's 107,743 because the final
  instruction differs: `deep-probe-q38.ps1` swaps the needle's one-line question for one that
  forces a long answer, so that deep-context TG is measured over 220 tokens instead of a
  handful.

---

Previous: [Chapter 9 — Speculative decoding](09-speculative-decoding.md) ·
Back to [README](README.md).

