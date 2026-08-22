# Chapter 9 — Speculative Decoding & MTP

Everything up to here made *prefill* faster. This chapter is about making **generation**
faster — the 104 tokens/second that determines how quickly words appear on your screen.

---

## 9.1 Why generation is slow, and why guessing helps

From §1.5: generation is **memory-bandwidth-bound**. To produce one token, the GPU reads all
the active weights and does a tiny amount of arithmetic. The compute units are mostly idle,
waiting on memory.

That idle compute is an opportunity. **Speculative decoding** exploits it:

```
1. Something cheap GUESSES the next k tokens.
2. The real model VERIFIES all k in a single forward pass
   -- which costs barely more than verifying one, because the pass was
      bandwidth-bound, not compute-bound.
3. Accept the longest correct prefix of the guess. Discard the rest.
```

If the guesser is right 70% of the time, you produce several tokens for the price of one.
**Output is mathematically identical to non-speculative decoding** — wrong guesses are
discarded, not accepted. This is a pure latency optimisation, not a quality trade.

Where it pays off most: **code**. Source code is highly predictable — closing brackets,
repeated identifiers, boilerplate. Guessers do very well.

---

## 9.2 What this build offers

```
--spec-type none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark,
             ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache
```

Two families:

**Model-based drafters** (`draft-*`) — a small neural network predicts the next tokens.
Accurate, but needs weights (VRAM) and compute.

**N-gram drafters** (`ngram-*`) — no model at all. They look for repeated token sequences in
the text so far and replay them. **Free in VRAM**, which on this machine matters enormously.
Excellent when output copies from input — refactoring, "rewrite this function", editing a
pasted file.

Key flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spec-type TYPE` | `none` | Which drafter (comma-separated list allowed) |
| `--spec-draft-n-max N` | 3 | Max tokens to guess per step |
| `--spec-draft-n-min N` | 0 | Min tokens to guess |
| `--spec-draft-p-min P` | 0.0 | Skip drafting when confidence is below this |
| `--spec-draft-model FNAME` | — | External draft model (for `draft-simple`) |
| `--spec-draft-ngl N` | auto | GPU layers for the draft model |
| `--spec-ngram-simple-size-n N` | 12 | Lookup n-gram length |
| `--spec-ngram-simple-size-m N` | 48 | Drafted m-gram length |

---

## 9.3 MTP — the head your model already has

**Your Qwopus model ships with a multi-token-prediction head.** That is what `MTP` in the
filename means, and it is why the model has 41 layers where the architecture declares 40:

```
print_info: n_layer     = 40
print_info: n_layer_all = 41          ← the extra one is the MTP head
qwen35moe.nextn_predict_layers = 1
```

MTP is a draft head **trained jointly with the base model**, so it shares its representations
and predicts far better than a generic small model. No separate download, no mismatched
tokeniser.

Right now, you are not using it. Every run so far logged:

```
W model has unused tensor blk.40.ffn_down_exps.weight (size = 220200960 bytes) -- ignoring
W model has unused tensor blk.40.nextn.eh_proj.weight -- ignoring
W model has unused tensor blk.40.nextn.enorm.weight   -- ignoring
```

`-- ignoring` means those weights are **not loaded**, so they cost no VRAM. They also do
nothing. To activate them:

```powershell
--spec-type draft-mtp --spec-draft-n-max 2
```

### The catch, and it is a real one

Activating MTP **loads blk.40**, and from the log above its expert tensors alone are
220 + 151 + 151 MB ≈ **~500 MiB**, plus attention and projection weights — call it **~530 MiB**.

Profile A (§7) has approximately **24 MiB spare**.

So on this machine MTP is not free — it must be paid for. The options, in order of
preference:

1. **Free ~530 MiB on the display GPU** by closing chat apps. Costs nothing else.
2. **Use the `q4_0` KV profile** (Profile B, §7) — its cache is ~700 MiB smaller than
   `q8_0`, which is more than MTP needs.
3. **Reduce context** — 130048 → 126976 frees KV *and* gains 26% prefill (§6.5).
4. **Use an n-gram drafter instead** — 0 MiB of VRAM. See §9.4.

> **Teaching point.** Speculative decoding trades **memory** for **latency**. On a machine
> with headroom it is close to a free win. On a machine at 99.9% VRAM utilisation it is a
> genuine trade, and the n-gram drafters — which cost nothing — become the smart first move.

### Measured result: it makes this model *slower*

Published figures for Qwen3.6 MTP are **1.17× on the 35B-A3B MoE** (1.73× on the 27B dense
model). Measured here, with `-c 65536`, `q4_0` KV, generating 300 tokens of fresh Python:

| Config | TG t/s | vs baseline |
| --- | --- | --- |
| **baseline (no drafting)** | **97–100** | — |
| `draft-mtp --spec-draft-n-max 1` | 90.1 | **−7%** |
| `draft-mtp --spec-draft-n-max 2` | 70.3 | **−29%** |
| `draft-mtp --spec-draft-n-max 3` | server failed to start (OOM) | — |
| `ngram-simple` n-max 2 / 4 / 8 | 96.6 / 97.6 / 97.4 | **0% (noise)** |

And the drafting itself was working *well*. From the server log:

```
draft acceptance = 0.67059 ( 171 accepted / 255 generated), mean len = 2.34
draft acceptance = 0.71837 ( 176 accepted / 245 generated), mean len = 2.43
```

**67–72% acceptance, 2.34 tokens per step — and throughput still fell by 29%.**

### Why — and it is instructive

The log says how the draft is produced:

```
common_speculative_init_result: creating MTP draft context against the target model
```

Each drafted token costs approximately a **full forward pass**. So drafting *k* tokens plus
verifying them costs about *k+1* passes to yield ~2.3 tokens. The arithmetic only works if a
model's per-token pass is expensive relative to the draft.

**This model's pass is extremely cheap** — ~3 B active parameters out of 35.5 B. There is
almost nothing to amortise. Watch the cost scale directly with `n-max`: −7% at 1, −29% at 2.
Every extra drafted token buys another pass and loses more ground.

That is also exactly why the vendor's own numbers show 1.73× on a *dense* 27 B model and only
1.17× on this MoE. On this hardware the same trend continues past 1.0× and into a loss.

> **Teaching point.** Speculative decoding trades *compute* for *latency*, and it needs idle
> compute to trade with. A sparse MoE at batch size 1 is bandwidth-bound but its total work
> per token is already tiny, so the drafting overhead is not hidden — it is the bill. **High
> acceptance does not mean a win.** Always compare end-to-end tokens/second against baseline;
> never conclude from the acceptance rate alone.

### Why the n-gram drafters were merely neutral

`ngram-simple` costs no forward pass, so it cannot lose much — and it didn't (0% within
noise). But it also gained nothing, because the test asked the model to **write a new module
from scratch**. There was nothing in the context to copy, so almost nothing was drafted.

**This is a limitation of the test, not a verdict on n-gram drafting.** The regime where it
should win is output that copies input:

- "refactor this function", "add type hints to this file"
- "translate this config to YAML"
- long code blocks reproduced with small edits

⚠️ **Untested and worth testing:** `ngram-simple` on an edit-a-pasted-file task. Since the
downside measured at 0%, it is close to a free bet in that regime.

### Recommendation for this machine

**Leave speculative decoding off.** `draft-mtp` costs 7–29% of your generation speed and
~530 MiB of VRAM you do not have; `ngram-simple` is free but did nothing for from-scratch
generation. Revisit `ngram-simple` if your workload is dominated by editing existing text.

---

## 9.4 N-gram drafters — the zero-VRAM option

Because they use no model weights, these cost **nothing** in VRAM. On a machine this tight,
that makes them the right thing to try *first*.

```powershell
--spec-type ngram-simple --spec-draft-n-max 4
```

They shine when generated text repeats something already in context:

- "Refactor this function" → most of the output is copied from the input
- "Add type hints to this file" → nearly all of it is copied
- Boilerplate, repeated imports, similar test cases

They do nothing for genuinely novel prose. Since the cost is zero and wrong guesses are
simply discarded, the downside is a small amount of wasted compute — which, per §9.1, you
were not using anyway.

Tuning knobs:

```powershell
--spec-ngram-simple-size-n 12       # how long a match must be to trigger
--spec-ngram-simple-size-m 48       # how many tokens to replay from it
--spec-ngram-simple-min-hits 1      # how many matches before trusting it
```

`ngram-mod`, `ngram-map-k` and `ngram-cache` are variants with different lookup strategies.
`ngram-cache` can be seeded from disk with `-lcs` / `-lcd`, so it learns your codebase's
idioms across sessions.

### `draft-dflash` and `draft-dspark`

Newer drafting methods, reported to be strong on Qwen3.6 for coding. They require
compatible draft weights. Worth investigating once the basics are measured — but measure
MTP and n-gram first.

---

## 9.5 How to measure it properly

Speculative decoding **cannot be measured with `llama-bench` or `llama-batched-bench`** —
neither supports `--spec-type`. You must measure over HTTP against `llama-server`.

It is also **workload-dependent in a way that raw throughput is not.** Acceptance rate
depends on what is being generated, so a synthetic prompt will mislead you. Test with *your*
prompts.

```powershell
# Launch with a drafter
cd S:\OneDrive\Tools\llamacpp
.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf" `
  --host 0.0.0.0 --port 9010 -c 98304 -np 1 `
  -ngl 999 -sm layer -ts 12,29 -fa on -b 2048 -ub 256 `
  -ctk q8_0 -ctv q8_0 -fit off `
  --spec-type draft-mtp --spec-draft-n-max 2
```

Then run a **realistic** generation-heavy request and read the server's own timing line:

```
eval time = 3850.1 ms / 400 tokens (9.63 ms per token, 103.9 tokens per second)
```

Compare that `tokens per second` with and without the drafter, on the *same* prompt. The
server log also reports draft acceptance statistics — that is what tells you whether to
raise or lower `--spec-draft-n-max`.

### Reading acceptance rate

| Acceptance | Meaning | Action |
| --- | --- | --- |
| > 70% | Excellent | Raise `--spec-draft-n-max` |
| 40–70% | Working | Near optimal |
| < 30% | Guessing badly | Lower `n-max`, or raise `--spec-draft-p-min`, or turn it off |

Low acceptance is worse than no speculation: you pay for drafting *and* for verifying
tokens you throw away.

---

## 9.6 The honest ranking

For this machine and an agentic coding workload, in order of value per unit of effort:

| # | Optimisation | Gain | VRAM cost | Status |
| --- | --- | --- | --- | --- |
| 1 | **Prompt caching** (`-np 1`, `-cram`) | **~34× on repeat prefill** | 0 (host RAM) | proven |
| 2 | Not overflowing VRAM (sysmem policy) | up to 5× | 0 | proven |
| 3 | Non-pipelined execution on mismatched GPUs | **+43% prefill** | frees memory | measured (§6.2) |
| 4 | `q4_0` KV enabling `-ub 1024` | +43% prefill | frees 700 MiB | measured; needle-tested (§6.4) |
| 5 | Right-sizing `-c` (130048 → 126976) | +26% prefill | frees memory | measured (§6.5) |
| 6 | `-ub 512` over 256 | +21% prefill | +819 MiB | measured |
| 7 | `ngram-simple` speculation | 0% on fresh generation | 0 MiB | measured; untested on edit-style work |
| — | `draft-mtp` speculation | **−7% to −29% TG** | ~530 MiB | measured — **do not use here** |

Speculative decoding is genuinely useful and it belongs at the bottom of this list. Get the
first five right before spending VRAM on the sixth — a 17% generation gain does not repay
breaking a configuration that only had 24 MiB of slack.

> **The closing lesson of this wiki:** the biggest wins were never exotic. They were *not
> overflowing memory*, *not asking for more context than needed*, and *reusing work you had
> already done*. Speculative decoding is the fun part. Prompt caching is the part that
> actually changed how the machine feels to use.

---

Back to the [index](README.md).

