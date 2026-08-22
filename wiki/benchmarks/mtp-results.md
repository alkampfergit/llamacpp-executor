# Speculative decoding results

> **Corrected 2026-08-22.** The `MTP loaded` column in the three `predict=400`
> tables below originally read `True` for every row, including baselines that ran
> with no `--spec-type` at all. That was an artefact of `mtp-test.ps1` grepping for
> a hardcoded `unused tensor blk.40`: the MTP head sits at `blk.<n_layer>`, which is
> `blk.40` on the 35B-A3B MoE but **`blk.64`** on Qwen3.8-27B. The pattern therefore
> never matched on the dense model, and "not ignored" was misread as "loaded".
>
> Corrected below: a baseline run cannot have the draft head active, and n-gram
> drafters use no model weights at all. The `draft-mtp` rows are genuinely `True`
> — independently confirmed by the draft-acceptance statistics those runs emitted.
> The detector now matches `unused tensor blk\.\d+\.nextn`, which is model-agnostic.
> **Throughput figures were never affected**; only this metadata column was.

## Qwopus3.6-35B-A3B (sparse MoE) — 3B active of 35.5B

MTP is a **net loss** here: each drafted token costs roughly a full forward pass,
and a model this sparse has nothing to amortise.

### ctx=65536 ub=512 KV=q4_0/q4_0 predict=300

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 99,61 | 0% | False |
| draft-mtp n2 | 70,32 | **-29%** | True |

### ctx=65536 ub=512 KV=q4_0/q4_0 predict=300

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 97,23 | 0% | False |
| draft-mtp n1 | 90,05 | **-7%** | True |

### ctx=65536 ub=512 KV=q4_0/q4_0 predict=300

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 97,64 | 0% | False |
| ngram-simple n2 | 96,64 | -1% | False |
| ngram-simple n4 | 97,62 | -0% | False |
| ngram-simple n8 | 97,43 | -0% | False |

## Qwen3.8-27B (dense) — every parameter read per token

Same flag, **opposite sign**: the expensive forward pass is exactly what makes the
draft pay for itself.

### ctx=65536 ub=512 KV=q8_0/q8_0 predict=400

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 22,55 | 0% | False |
| draft-mtp n1 | 35,17 | **+56%** | True |
| draft-mtp n2 | 41,07 | **+82%** | True |
| draft-mtp n3 | 44,99 | **+100%** | True |

### ctx=65536 ub=512 KV=q8_0/q8_0 predict=400

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 22,42 | 0% | False |
| draft-mtp n4 | 47,09 | **+110%** | True |
| draft-mtp n5 | 38,08 | +70% | True |
| draft-mtp n6 | 44,41 | +98% | True |
| draft-mtp n8 | 47,21 | **+111%** | True |

### ctx=65536 ub=512 KV=q8_0/q8_0 predict=400

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 22,42 | 0% | False |
| ngram-simple n4 | 22,33 | -0% | False |
| ngram-simple n8 | 22,35 | -0% | False |

## Reading these tables

**n5 at 38,08 is unexplained.** n4 and n8 both sit near 47 t/s, so a dip at n5 breaks
monotonicity by more than the ±8% noise floor. Treat it as unresolved rather than as
a real optimum, and note the default sweep in `mtp-test.ps1` originally skipped n5
entirely — it now includes it precisely because this is where the curve misbehaves.

**High draft acceptance is not a win.** The MoE runs reported 67–72% acceptance while
*losing* 29%. Only end-to-end tokens/second decides.

**These are shallow-context numbers.** At real 108k depth the dense model's MTP gain
falls from ~2.0× to ~1.4×, and MTP also costs ~27% of prefill plus ~1.1 GiB of VRAM.
See `deep-results.md`.
