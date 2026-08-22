# Worked example: a 20.2 GiB MoE model on 24 GiB of mismatched VRAM

A full campaign, including the wrong turns, so the reasoning is visible rather than just
the answer.

## Contents

- [The problem](#the-problem)
- [Step 1: the real budget](#step-1-the-real-budget)
- [Step 2: characterise the model](#step-2-characterise-the-model)
- [Step 3: predict](#step-3-predict)
- [Step 4: the first wrong turn](#step-4-the-first-wrong-turn--a-contaminated-sweep)
- [Step 5: the second wrong turn](#step-5-the-second-wrong-turn--cpu-offload)
- [Step 6: the breakthrough](#step-6-the-breakthrough)
- [Step 7: the third wrong turn](#step-7-the-third-wrong-turn--kv-quantisation)
- [Step 8: the accident](#step-8-the-accident-that-gave-the-best-result)
- [Result](#result)
- [Transferable lessons](#transferable-lessons)

---

## The problem

| | |
| --- | --- |
| GPU 0 | RTX 3070, 8 GiB, Ampere |
| GPU 1 | RTX 5060 Ti, 16 GiB, Blackwell — **drives the display** |
| CPU / RAM | Ryzen 9 5900X 12c/24t, 64 GiB |
| Model | 20.21 GiB Q4_K_M, 35.5 B total / ~3 B active, 256 experts (8 used), hybrid attention, 40 layers + 1 MTP head |
| Requirement | **≥130k context**, everything in VRAM, decent speed |

Starting point: another runtime was delivering ~300 t/s prefill on this hardware, with both
GPUs near-idle and the CPU at ~47%.

---

## Step 1: the real budget

```
GPU 0:   8192 −  407 (screen recorder)              =  7785 MiB
GPU 1:  16311 − 1658 (desktop, Slack, Signal, ...)  = 14653 MiB
                                                      ───────────
                                            usable  = 22438 MiB
```

**2.0 GiB gone before starting**, almost all of it on the display GPU. Against a 20.2 GiB
model that is decisive — and it explains why the naive capacity-ratio split is wrong.

---

## Step 2: characterise the model

```
architecture = qwen35moe      n_layer = 40    n_layer_all = 41
n_head = 16    n_head_kv = 2   n_embd_head_k = 256
n_expert = 256  n_expert_used = 8
ssm_d_conv = 4  ssm_n_group = 16          <- hybrid: not all layers are attention
n_ctx_train = 262144
nextn_predict_layers = 1                  <- MTP head, layer 40
```

The textbook KV formula predicted **10.2 GiB** at 131k. Measured: **2.56 GiB** — 4× less,
because `ssm_*` means most layers carry a fixed-size state. **This is the only reason a
130k window was affordable at all**, and it is not a fact to carry over to a dense model.

---

## Step 3: predict

```
llama-fit-params -m <model> -c 131072 -fa on -ub 512 -fitp on

        model  context  compute
CUDA0    6899      791      596
CUDA1   13003     1831     1041
Host      272        0      520
```

The three consumers, per device. Sweeping `-c` gave the real KV curve in ~5 seconds; a
`-ub` sweep gave the compute curve, including `-fa off` = 10.4 GiB and `-ub 2048` = 6.5 GiB.

Arithmetic for the target:

```
20174 (weights) + 1422 (q8_0 KV @130k) + 1637 (ub 512)  =  23233
available                                                =  22438
                                                            ─────
                                                    SHORT =    795 MiB
```

So the question became: **which 795 MiB do we give up?**

---

## Step 4: the first wrong turn — a contaminated sweep

A multi-value `-ts` sweep inside one process:

```
-ts  6/18   PP 2244        -ts 10/18   PP 1703   <- ???
-ts  8/18   PP 2541        -ts 12/18   FAILED TO LOAD
```

But `-ts 10/18` measured **alone** gave **2883 PP / 117 TG** — 69% higher. Nothing about
the configuration changed; it was simply the third load in one process, and VRAM
fragmentation degraded it.

**→ One fresh process per data point.** Every later number was collected that way.

---

## Step 5: the second wrong turn — CPU offload

The obvious way to free 795 MiB on an MoE model is to push experts to the CPU:

| `-ncmoe` | PP t/s | TG t/s |
| --- | --- | --- |
| 2 | 826 | 88 |
| 4 | 541 | 82 |
| 6 | 421 | 76 |
| 8 | 344 | 60 |

Moving **2 of 40 layers** — 5% of the model — cut prefill by 44%. Prefill activates nearly
all 256 experts per batch; generation activates 8. The CPU becomes the bottleneck while
both GPUs wait.

**→ CPU offload is a last resort, not a tuning knob.**

---

## Step 6: the breakthrough

Instead of moving weights off the GPU, shrink the *workbench*:

```
-c 130048 -ub 256 -ts 12,29 -ctk q8_0 -ctv q8_0 -ngl 999 -fa on -fit off
→ PP 1360–1466, TG 95–100, peak 23606 MiB, ZERO CPU offload
```

`-ub 256` costs ~31% of prefill; `-ncmoe 2` costs 44% **and** was slower on generation too.
The feasible window turned out to be a single point:

```
ub 128 ts13,28  OOM        ub 384 ts12,29  fails (cublasCreate)
ub 256 ts11,30  fails      ub 512 ts12,29  OOM
ub 256 ts12,29  WORKS      ub 512 ts11,30  OOM
ub 256 ts14,27  OOM        ub 512 ts13,28  OOM
```

Note `ts 11,30` failing while `12,29` succeeds — moving a layer *off* the small card puts
it on the display card, which has less free memory. The binding constraint was never the
8 GiB card.

---

## Step 7: the third wrong turn — KV quantisation

Going below `q8_0` should have bought the last 700 MiB:

| Pair | PP t/s | TG t/s |
| --- | --- | --- |
| `q8_0`/`q8_0` | 1466 | 95 |
| `q5_1`/`q5_1` | 107 | 19 |
| `q4_0`/`q8_0` | 160 | 25 |

Correct output, no warning, **14× slower**. The first explanation — "sub-q8 KV is slow" —
was wrong. The real cause is that stock CUDA builds compile flash-attention kernels for
only four *symmetric* pairs (`f16/f16`, `bf16/bf16`, `q8_0/q8_0`, `q4_0/q4_0`), and
`fattn.cu` rejects `K->type != V->type` outright. `q4_0/q8_0` failed on asymmetry;
`q5_1/q5_1` failed because `q5_1` is not a supported type at all. Attention left the GPU —
visible as 5.2 GiB appearing in host RAM.

**→ The lesson is not about bit-widths.** A memory optimisation that makes things
dramatically slower usually means the operation moved off the accelerator. Check the host
memory figure to confirm. And note the untested implication: `q4_0/q4_0` **is** on the fast
path, and would have been the right thing to try.

---

## Step 8: the accident that gave the best result

Two runs that made no sense together:

```
-c  98304 -ub 512   PP 1630
-c 122880 -ub 512   PP 2145     <- larger context, 32% faster?
```

The log explained it:

```
graph_reserve: failed to allocate compute buffers
sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
```

The larger context couldn't fit its pipeline-parallel buffers, fell back, and the fallback
was **faster**. On a mismatched pair the pipeline stalls on the slower card, and its 4
activation copies were what broke the fit.

### Step 8b: the wrong fix, and a lesson about attribution

The obvious next move was to make it deliberate:

```powershell
$env:GGML_SCHED_MAX_COPIES = "1"     # measured 1656 vs 1466 -- "+13%!"
```

**That was wrong.** `GGML_SCHED_MAX_COPIES` is a compile-time `#define`; nothing reads it
from the environment. The proof was already in the logs: a later run *with the variable set*
still allocated 1407 MiB of pipelined buffers before falling back. The "+13%" was variance on
a machine with an ±8% noise floor — and it got written into a wiki as fact.

**→ Before crediting a knob, confirm the knob exists at runtime. A gain near your noise floor
is not evidence.**

The effect was real; the mechanism was not what I claimed. Reaching the lean path deliberately
requires either a rebuild (`-DGGML_SCHED_MAX_COPIES=1`) or raising `-ub` until the pipelined
reserve fails on purpose.

### Step 9: the actual best configuration

Revisiting the KV table showed `q4_0/q4_0` was **on the fast kernel path and never tested**.
It is ~700 MiB smaller than `q8_0` — almost exactly the headroom `-ub 1024` needs. The two
effects compound:

| KV | `-ub` | PP t/s |
| --- | --- | --- |
| `q8_0` | 512 | 1850 |
| `q4_0` | 512 | 1787 |
| **`q4_0`** | **1024** | **2650** |

`-ub 1024` cannot fit its pipelined buffers, so it takes the lean path — 2650 t/s at 130k,
matching the 2651 t/s measured at *8k* context. The "context tax" largely evaporated once the
configuration actually fit.

### Step 10: the quality gate

`q4_0` is a lossy cache, and no throughput benchmark can detect that. So: hide
`CRIMSON-PELICAN-4417` in RECORD 0001 of a 3000-record monotonous log, ask for it after
~108,000 tokens, and see who remembers.

| Config | Needle | Real PP | Real TG |
| --- | --- | --- | --- |
| `q8_0`, `ub 512` | **PASS** | 1414 | 47.5 |
| `q4_0`, `ub 1024` | **PASS** | 1758 | 46.6 |

Both retrieved it exactly. Note the real numbers: prefill 25–35% below synthetic, and
**generation roughly halved at depth** (98 → 47 t/s).

The first two attempts returned empty answers for *both* configs — a raw `/completion`
prompt makes an instruction-tuned model emit EOS immediately. **A quality failure that hits
every configuration equally is a harness bug.**

---

## Result

Two profiles, because the honest answer depends on what you value:

```powershell
# QUALITY-FIRST -- q8_0 KV, full 130k
llama-server -m <model.gguf> -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 12,29 -fa on `
  -b 2048 -ub 512 -ctk q8_0 -ctv q8_0 -fit off -cram 24576
# 1850 t/s synthetic, 1414 t/s real, TG 98/47.5, needle PASS

# SPEED-FIRST -- q4_0 KV, full 130k
llama-server -m <model.gguf> -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 12,29 -fa on `
  -b 2048 -ub 1024 -ctk q4_0 -ctv q4_0 -fit off -cram 24576
# 2650 t/s synthetic, 1758 t/s real, TG 105/46.6, needle PASS
```

And a third that beats both if you can spare 2.4% of the window: `q8_0` at `-c 126976`
reaches **2323 t/s** — higher precision *and* faster than `q4_0` at `-ub 512`.

Versus the ~300 t/s starting point, that is up to **8.8×**.

But the finding that outranked all of it: with prompt caching working, a 57,559-token request
took 32.5 s cold, while the next request with the same prefix evaluated **790 tokens in
0.96 s** — a **34× reduction** in real wait, for zero VRAM.

---

## Transferable lessons

1. **Budget from free VRAM, not installed VRAM.** The display GPU is the constraint.
2. **Predict before measuring.** `llama-fit-params` eliminates most candidates in seconds.
3. **One process per data point**, or fragmentation invents results.
4. **Shrink the workbench before moving the weights.** `-ub` beats `-ncmoe` decisively.
5. **A benchmark tool without `-c` measures the wrong memory regime.**
6. **Grep logs, not just tables.** Silent fallbacks change what you measured.
7. **Suspect "off the accelerator" whenever a memory saving costs an order of magnitude.**
8. **Confirm a knob is runtime before crediting it.** Step 8b cost a wrong published claim.
9. **A smaller KV cache can be *faster*** — it buys the headroom for a bigger `-ub`.
10. **Quality-gate every lossy setting.** Speed benchmarks are blind to regressions.
11. **A failure affecting every configuration equally is a harness bug, not a finding.**
8. **Right-sizing context is free speed** — it returns memory *and* speeds prefill.
9. **Test `GGML_SCHED_MAX_COPIES=1` on any mismatched GPU pair.**
10. **Reusing work beats optimising work.** Prompt caching was worth more than every flag.

