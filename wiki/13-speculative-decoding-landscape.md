# Chapter 13 — The speculative-decoding landscape: DFlash2, DSpark, and PFlash

Chapters 11 and 12 audited a third-party optimization report for Qwen3.8-27B: chapter 11 took
the **file**, chapter 12 took the **build**. This chapter takes the part neither of them
touched, and the part our whole measurement campaign never went near — the **methods**.

The report's §12–§19 describe three things by name: **MTP**, which we measured at +110%
(§10.7); **DFlash2**, a diffusion drafter we have never run; and **PFlash**, a prefill
accelerator we have never run *and* which does not exist in llama.cpp at all. It then reasons
about how they compose.

That reasoning is where the interesting mistakes are. The report gets the right answer to the
composition question and gives the wrong reason for it, and it makes one claim — §16.2, about
prefix-cache reuse — that is sharper than anything else in the document and that our own data
can settle rather than speculate about.

The verdicts are chapter 11's, plus one the earlier chapters did not need:

| Verdict | Means |
| --- | --- |
| **Corroborated** | A measurement, a metadata dump, or the source on this box agrees |
| **Contradicted** | A measurement, a metadata dump, or the source on this box disagrees |
| **Right answer, wrong reason** | The conclusion holds; the stated mechanism is not the real one |
| **Untested** | Neither we nor (verifiably) the report has evidence. Say so and stop |

New evidence in this chapter comes from three places: the llama.cpp submodule at
`llama.cpp/` (read at `e85caa81e`, with every claim below re-checked against our baseline
commit `fe8156f78`), the Hugging Face model API, and the published README of the PFlash
project. **No new benchmark was run.** Where we have no number, the chapter says so instead of
siding with the report.

---

## 13.1 Already covered — do not re-read the report's first half

| Report sections | Where they were already audited |
| --- | --- |
| §1 quant ladder, §2 architecture, §3 KV arithmetic and the `q8_0`/`q4_0` pair, §4 (partly), §10–§11 launch command, §20–§22 recommendations | [Chapter 11](11-quant-selection-qwen38.md) |
| §5–§9 build flags, §18 which optimizations generalise, §21 priority ranking | [Chapter 12](12-build-flags-analysis.md) |
| §12 MTP, and every number behind it | [Chapter 10 §10.7](10-results-qwen38-27b.md), [Chapter 9](09-speculative-decoding.md) |

Two things from that half are load-bearing here and are worth restating in one line each,
because the rest of this chapter is arithmetic on top of them:

- **MTP on this model is worth ~2x generation and costs 1107 MiB and 27% of prefill** (§10.7).
- **At `-c 130048` with `q8_0` KV, MTP peaks at 23560 MiB against the 23215 MiB that was free
  when §10.5 was measured.** There is no headroom left at 130k. Remember that number; §13.4
  is entirely about it.

---

## 13.2 The DFlash2 drafters are real, and the report's sizes are exact

§13 claims small DFlash2 GGUF drafters exist for Qwen3.8-27B at roughly 1.1 GB (Q4_K_M),
2.0 GB (Q8_0) and 3.8 GB (BF16). This is the easiest claim in the document to check and the
report gets it dead right, to the byte:

`https://huggingface.co/api/models/z-lab/Qwen3.8-27B-DFlash2-GGUF?blobs=true`

| File | Bytes | MiB | Report says |
| --- | ---: | ---: | --- |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1,143,006,752 | **1090** | ~1.1 GB ✓ |
| `Qwen3.8-27B-DFlash2-Q8_0.gguf` | 2,056,414,752 | **1961** | ~2.0 GB ✓ |
| `Qwen3.8-27B-DFlash2-BF16.gguf` | 3,860,293,152 | **3681** | ~3.8 GB ✓ |

GGUF metadata: `general.architecture = dflash`, `context_length = 262144`. The same three
files are mirrored at `incoai/Qwen3.8-27B-DFlash2-GGUF`. **Verdict: corroborated.**

> **The report's decimal gigabytes are not our gibibytes.** 1.1 GB is 1090 MiB, and in a
> budget where 760 MiB was the entire headroom at 130k (§10.5), that 4% matters. Every VRAM
> figure in this wiki is MiB because that is what the allocator reports.

There is more on the shelf than the report knows about. **DSpark** — which the report never
mentions — also ships as GGUF for this target:

| Repo | File | Bytes | MiB |
| --- | --- | ---: | ---: |
| `magnitudedev/Qwen3.8-27B-DSpark-GGUF` | `Qwen3.8-27B-DSpark-Q8_0.gguf` | 1,455,376,576 | **1388** |
| `erlidev/Qwen3.8-27B-DSpark-GGUF` | — | — | — |

And our baseline binary already accepts it: `--spec-type` lists
`none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark, ngram-simple,
ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache`.

**None of these files is on this disk.** `S:\HuggingFace\lmstudio\` holds
`unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf` and nothing DFlash-shaped. So DFlash2 is
"testable with no rebuild" — but not testable with no download. That distinction runs through
§13.10.

---

## 13.3 What `draft-dflash` actually requires

The report says DFlash2 "uses a separate diffusion-based drafter to propose a block of future
tokens" and stops there. The source says considerably more, and most of it is operationally
relevant. Everything below is `common/speculative.cpp`, `src/models/dflash.cpp` and
`common/arg.cpp`, verified present at our baseline commit.

### One drafter file, and the type is inferred from it

`common_speculative_types_from_gguf()` opens the draft GGUF, reads
`general.architecture`, and if it is `"dflash"` returns `draft-dspark` when the tensor
`markov_w1.weight` is present and `draft-dflash` otherwise. So:

```
-md S:\...\Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

is sufficient on its own — **`--spec-type` is optional**, and the log will say
`auto-detected speculative type 'draft-dflash' from the draft model metadata`. Passing
`--spec-type draft-dflash` explicitly (as the z-lab README does) is belt and braces, not a
requirement.

`dspark` is not a different mechanism; it is `common_speculative_impl_draft_dflash` constructed
with a different `type` flag. The differences are a **Markov head** (`markov_w1/w2`) plus a
confidence head, and an **anchor-first block layout** that yields one extra draft token per
block. Note the source's own caveat: *"only Qwen3-style backbones are supported for now"* —
which is exactly our target family.

### The drafter is welded to one target model

This is the part the report's §18 gestures at ("a DFlash2 drafter must also be trained for a
compatible target model") without saying what enforces it. Three things do:

1. The drafter GGUF carries a `target_layers` array. `llama_model_dflash::load_arch_hparams`
   throws `"DFlash model requires 'target_layers' in GGUF metadata"` without it.
2. At construction the drafter reaches **into the target context** and switches on hidden-state
   extraction at exactly those layers:
   `llama_set_embeddings_layer_inp(ctx_tgt, target_layer_ids[k], true)`.
3. Every prefill and decode batch is mirrored: `process()` gathers the target's layer inputs,
   runs them through the drafter's encoder, and injects the result into the drafter's own KV
   cache. If the extraction did not happen it is a hard stop —
   `GGML_ABORT("DFlash: target layer %d input not extracted.")`.

The drafter's `fc` tensor is shaped `{target_layers.size() * n_embd_draft, n_embd_draft}` while
`process()` builds features of width `target_layers_n * n_embd_target`. A drafter trained
against a different target width therefore fails at a matmul, not at a plausibility check.
**Mismatch is loud, not silent** — which is a relief, and worth knowing before trying a
Qwen3.6 drafter against a Qwen3.8 target.

### `--spec-draft-n-max` defaults to 3, and DFlash wants more

The drafter's GGUF carries `dflash.block_size` (default 16 if absent) and
`dflash.sample_from_anchor`. The constructor clamps:

```
n_draft_max = (is_dspark && sample_from_anchor) ? block_size : block_size - 1;
```

and warns when you ask for more:
`requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d`.

Two practical consequences:

- **The default is wrong for DFlash.** `--spec-draft-n-max` defaults to **3** (confirmed in the
  baseline `--help`). Our MTP optimum was 4 (§10.7). The z-lab README uses **7**, which implies
  a trained `block_size` of 8. A DFlash run left at the default drafts less than half a block.
- **You can discover the block size without reading the file.** Pass
  `--spec-draft-n-max 99` once, read the clamp warning, then set it exactly. The warning
  prints both the trained block size and the clamped value.

### If the drafter is missing, drafting silently disappears

This is the trap. `common_speculative_init()` builds its implementation list with:

```cpp
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
```

No draft context means the config is never added. If it was the only one requested, `impls` is
empty, and:

```cpp
if (impls.empty()) {
    SPC_TRC("%s", "no implementations specified for speculative decoding\n");
    return nullptr;
}
```

`SPC_TRC` is `LOG_TRC`, verbosity level 4. At default verbosity **you get nothing**: the server
starts, `slot.can_speculate()` returns false, and every request decodes at baseline speed with
no warning anywhere. `--spec-type draft-dflash` without `-md` is a no-op that looks like a
success.

Contrast `draft-simple`, whose config is added unconditionally and whose constructor throws
`"draft-simple requires a draft context"`. Same mistake, opposite diagnostics. If you take one
operational habit from this chapter, take this one:

> **Never trust `--spec-type` from the command line. Confirm from the output.** The slot JSON
> carries `"speculative": true|false`, and the end-of-request line
> `draft acceptance = 0.xxxxx (N accepted / M generated), mean len = x.xx` only appears when a
> drafter really ran. §10.7 already caught a harness that reported "MTP loaded" for a baseline
> run; this is the same failure one level down.

### The draft context inherits the target's context size, and f16 KV

`common_base_params_to_speculative()` copies the entire `common_params` and overrides only the
model, `n_gpu_layers`, threads, devices, tensor overrides and KV types. **There is no
`--spec-draft-ctx-size`** — I checked the baseline `--help` and the parser. So a 1090 MiB
drafter serving a `-c 130048` target gets a **130048-token context of its own**, with
`cache_type_k = cache_type_v = GGML_TYPE_F16` by default.

The two flags that exist are `-ctkd` / `--cache-type-k-draft` and `-ctvd`. And the whole point
of chapter 11 applies to them:

> **The asymmetric-KV trap applies to the drafter too.** `CLAUDE.md` records why: in
> `ggml/src/ggml-cuda/fattn.cu`, `if (K->type != V->type) return BEST_FATTN_KERNEL_NONE;`.
> `-ctkd q8_0 -ctvd q4_0` on a drafter is the same 14x cliff as `--cache-type-k q8_0
> --cache-type-v q4_0` on the target, in a place nobody thinks to look. Keep the pair
> symmetric. Note also that DFlash sets `llama_set_causal_attn(ctx_dft, false)` — the draft
> attention is non-causal, which is one more reason not to assume the target's kernel coverage
> transfers.

Whether the drafter's KV is actually 130k-sized depends on whether its GGUF declares
`attention.sliding_window`; `load_arch_hparams` honours one if present, and the DSV4 DSpark
backbone forces one on every layer. **Untested** — the first thing to read out of a DFlash load
log is `n_swa` and the reported KV size.

---

## 13.4 What DFlash would cost here, and why §13's conclusion is right

§13 ends with: MTP is probably preferable for 128K coding, and "DFlash2 becomes more
interesting for shorter contexts where maximum interactive generation speed matters more." Our
VRAM data says that is correct — and gives a reason the report does not have.

Start from §10.7's measurement. **MTP costs 1107 MiB and brings no weights at all** — the head
is already inside `Qwen3.8-27B-Q4_K_M.gguf` as `blk.64`. All 1107 MiB is the second graph
reservation for the draft context and its compute buffers, and §10.7 records three failed
attempts to shrink it (`-ctkd q8_0`, `-ub 256`, `-ub 128`): "there is no knob that makes MTP
cheap."

DFlash pays that **and** brings weights:

| Component | MTP @130k | DFlash2 Q4_K_M @130k |
| --- | ---: | ---: |
| Target model + `q8_0` KV + buffers (§10.5) | 22453 MiB | 22453 MiB |
| Draft context + compute buffers | 1107 MiB (measured) | ~1100 MiB (assumed similar; **untested**) |
| Drafter weights | **0** | **1090 MiB** (measured from the GGUF) |
| Drafter KV @ the target's `-c`, f16 | included above | **unknown** — depends on `n_swa` |
| **Total** | **23560 MiB** | **≥ 24600 MiB** |

Free VRAM on this box when §10.5 was surveyed: **23215 MiB**. MTP at 130k already sits at
23560 MiB — at or past the line. DFlash2 needs at least another gigabyte on top of that,
before its own KV cache.

> **DFlash2 at 130k on this machine is not a tuning problem. It is arithmetic.** It does not
> fit, and the failure mode on Windows is not an error — it is the silent shared-memory
> overflow of chapter 8, at a fifth of the speed.

Where it *can* fit is exactly where §13 says. Dropping the target from 130k to 64k releases
**2142 MiB** of `q8_0` KV (4318 → 2176 MiB, §10.2), which is enough to pay for the drafter's
weights with change left over.

> **Verdict on §13: corroborated, with a better reason.** The report frames the trade-off as
> "extra 1–4 GB can reduce the maximum practical context length". On this box it is sharper
> than that: at 130k the budget is already spent, so DFlash2 is not a context-length *trade*,
> it is a **64k-or-below proposition**. And that reframes the comparison the report proposes in
> its Test D: DFlash2 must be benchmarked against **MTP at the same 64k**, not against MTP at
> 130k. Comparing a 64k DFlash run to a 130k MTP run would measure the context change, not the
> drafter.

---

## 13.5 §14 — can they be combined? The source says more than the report does

§14 says MTP and DFlash2 are "alternative model-based speculative decoding strategies", that
"you normally choose one", and that `MTP + n-gram` "may be supported". The flag documentation
appears to disagree outright: `--spec-type` is a **"comma-separated list of types of
speculative decoding to use"**.

Both are partly right, and the source explains why in four steps.

### Step 1 — the parser appends, it does not replace

```cpp
{"--spec-type"}, ...
    const auto types_str = string_split<std::string>(value, ',');
    auto types = common_speculative_types_from_names(types_str);
    params.speculative.types.insert(params.speculative.types.end(), types.begin(), types.end());
```

The default is `{ NONE }`, so `--spec-type draft-mtp` produces `{ NONE, DRAFT_MTP }`, and
repeating the flag accumulates. `NONE` in the list is inert (it is never passed to
`add_config_if_enabled` and hits an empty `case` in the factory switch), but it does make
`spec_types_is_default()` false — which is how `--spec-type none` explicitly *suppresses* the
GGUF and sidecar auto-detection that would otherwise pick a type for you.

### Step 2 — several implementations really are built, in a fixed order

`common_get_enabled_speculative_configs()` is a plain bitmask over the requested types. There
is **no mutual-exclusion check anywhere**. Every requested type that has what it needs becomes
its own `common_speculative_impl`, appended in this hard-coded order:

```
// this list here defines the priority of the speculators
// the one with highest priority are listed first
ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache,
draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark
```

So the report's "you normally choose one" is not enforced by the code. Multiple speculators
coexist.

### Step 3 — but they cascade, they do not add

From `common/speculative.h`, on the per-sequence draft flag:

```
// this flag is used to chain the drafts through all the available implementations
// after the first successful draft from an implementation, we set it
//   to false to prevent further drafts for that sequence
```

And `common_speculative_draft()` implements exactly that: it walks `spec->impls` in priority
order, calls `impl->draft(dparams)`, and for any sequence that came back with a non-empty
result sets `drafting = false`, records `impl_last[seq_id]`, and breaks out early once no
sequence is still looking. **Drafts are never concatenated.** The first speculator in priority
order that produces anything wins that token position outright.

> This is the sentence that resolves §14: the list is a **fallback chain**, not a sum. Adding a
> second speculator cannot make drafts longer. It can only change *which* speculator gets
> asked first.

### Step 4 — and that makes "MTP + n-gram" a demotion, not an addition

Look at the priority order again. **Every n-gram method outranks every model-based drafter.**
`--spec-type ngram-mod,draft-mtp` means: try n-gram first; ask MTP only when the n-gram lookup
finds nothing.

Now put §10.8 next to it. On this model, `ngram-simple` measured 22.33 t/s against a 22.42 t/s
baseline — **−0.4%, i.e. nothing**, because an n-gram drafter can only predict text that
literally repeats earlier in the window and freshly written code mostly does not. MTP measured
+110%.

So on the workload we actually care about, the cascade has no upside and a structural downside:
every time the n-gram *does* fire, it displaces an MTP draft that was probably better. The
report calls `MTP + n-gram` a thing that "may be supported". It is supported. Whether it is a
good idea is an empirical question that our two existing measurements suggest will come out
negative, and that is cheap to settle (§13.10, E1).

> **Verdict on the composition half of §14: contradicted as stated, and the correction matters.**
> Combining is supported. It is a priority cascade. And the specific combination the report
> floats as harmless puts the useless speculator in front of the useful one.

### And `MTP + DFlash2` is structurally impossible, for a reason the report never states

§14's conclusion here is right. Its reason — "competing strategies" — is not the mechanism.
The mechanism is that **there is exactly one draft model and exactly one draft context**:

```cpp
struct common_speculative_init_result::impl {
    llama_model_ptr   model;      // one
    llama_context_ptr context;    // one
};
```

and if `draft-mtp` appears anywhere in the type list, that single context is created with

```cpp
if (spec_mtp) { cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP; }
```

There is nowhere to put a second drafter. `--spec-type draft-mtp,draft-dflash` with
`-md <dflash.gguf>` would load the DFlash weights into a context typed as MTP and hand the same
context to both implementations. Do not do it; there is no configuration in which it means what
you want.

> **Verdict on §14's conclusion: right answer, wrong reason.** MTP and DFlash2 are not
> alternatives because of anything about drafting theory. They are alternatives because
> llama.cpp holds one `llama_context` for the draft side and lets the presence of `draft-mtp`
> decide its type.

One more detail for anyone scripting A/B tests: the HTTP API's per-request field is
`speculative.type`, **singular**, and its handler *replaces* the whole vector with one element.
The server schema describes it as "for debugging and research purposes". You cannot express a
cascade per request.

---

## 13.6 §15 and §17 — PFlash is not a llama.cpp feature at all

§17 says PFlash "is not currently a normal integrated llama.cpp switch" and is "still
experimental / external". That is right, and understated. A case-insensitive grep for
`pflash|p-flash` across every `.cpp`, `.h`, `.cu` and `.md` in the submodule returns **zero
matches**. It is not an unmerged PR or a hidden flag. It is a different program.

The program is real. It is [`Luce-Org/lucebox-hub`](https://github.com/Luce-Org/lucebox-hub)
`pflash`, and its published description matches §15's mechanism closely: a C++/CUDA
daemon-resident process holding drafter, scorer and target in one ggml allocator, running a
four-kernel `mean_K → score → select → sparse_fwd` pipeline so the target only attends to spans
a small drafter flagged as important. Its research lineage is the *Speculative Prefill* /
*Cross-Family Speculative Prefill* / *FlashPrefill* line of papers, not the DFlash line — the
report is right to file it under a different phase.

Its headline numbers, from its own write-up:

| | PFlash TTFT | its llama.cpp baseline | claimed |
| --- | ---: | ---: | ---: |
| 128K, RTX 3090, Qwen3.6-27B Q4_K_M | 24.8 s | ~257 s | **10.4x** |
| 64K, same | 13.5 s | ~135 s | **10.0x** |

**Recalibrate that against our box before getting excited.** A 257-second prefill of 131072
tokens is about **510 t/s**. Our measured *real* prefill on this hardware is **830 t/s** at
108k depth (§10.9), and §10.5 showed prefill is flat across the whole window, so 128k here is
roughly 158 s, not 257 s.

> **The honest headroom is ~6x on this machine, not 10x.** Their multiple is inflated by a
> llama.cpp baseline 1.6x slower than ours. That is still a very large number for a cold
> long-prompt TTFT — six times is not a rounding error — but it is the number to plan against.
> (Caveats: different target model generation, single 24 GB card versus our 8+16 GiB split, and
> no independent reproduction. **Untested** on this box.)

PFlash does not touch decode; its own write-up pairs it with "dflash spec decode" for
generation. So §15's framing of it as a prefill-only accelerator is **corroborated**, and §17's
"conceptually, yes, they compose" is corroborated *as a concept* — with the correction that
composition here means running two separate projects, not two flags.

---

## 13.7 §16.2 — the sharpest claim in the document, and our data says it is right

This is the one section of the report worth the whole document. It argues that PFlash may
select a different subset of the prompt per request, destroying prefix/KV-cache reuse, so that
after enough agent turns ordinary full prefill plus cache reuse becomes competitive or faster.

We have the measurement that settles it, and it is the single biggest win in this entire
project.

From §10.10, on **all seven** deep runs without exception:

```
slot get_availabl: selected slot by LCP similarity, f_sim_best = 1.000 (> 0.100 thold), f_keep = 0.998
```

The second identical request re-prefilled **4 tokens instead of 107,763**. A 130-second first
request became roughly a 14-second one — and the 14 seconds is almost entirely *generation*.
The prefill component of a warm turn on this box is four tokens: single-digit milliseconds.

Now hold PFlash's own published caveat next to that: the park/unpark memory dance costs
**~3 s per request**, before the drafter has scored a single token of the prompt.

> **So the report understates its own point.** On this machine, for turn 2 and after, PFlash is
> not "competitive" with full prefill plus cache reuse. Its fixed per-request overhead alone is
> about **three orders of magnitude** more prefill time than a `f_sim_best = 1.000` cache hit
> costs, and that is before the scoring pass over 128k tokens. Unless PFlash grows a prefix
> cache of its own, the crossover is not "after enough agent calls" — it is at **turn two**.

### And the mechanism is worse than "a different subset each time"

The report's example is two calls selecting `{4, 7, 20, 33...}` and `{2, 7, 14, 41...}`. That
undersells the problem. llama.cpp's cache is keyed on the **longest common prefix of the token
sequence the target actually processes**. A prefill-side selector computes its keep-set as a
function of the *entire* prompt — including the last turn's tool output. So appending one token
at the end can change which of the first 100,000 tokens survive.

The resulting sequences are not merely *different*. They are **not prefixes of each other**.
There is no LCP to find, so there is nothing for `f_sim_best` to score, and the failure is
total rather than partial.

### Contrast: llama.cpp's own decode-side speculators are cache-compatible by construction

This is the structural asymmetry that §19's tidy `PFlash → MTP/DFlash2` diagram hides. In
`tools/server/server-context.cpp`, the prompt cache saves and restores **both** contexts:

```cpp
const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, ...);
const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, ...) : 0;
```

and on a checkpoint hit it restores the target KV, the draft KV, and the speculator's internal
state together:

```cpp
it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
common_speculative_set_state(spec.get(), slot.id, it->data_spec);
```

MTP, DFlash and DSpark do not fight prefix reuse — they are **checkpointed alongside it**.
(DFlash even warns when they fall out of step: `begin()` logs
`ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. Drafts may
degrade.` That warning is the thing to watch for in a DFlash run.)

The cost is that every checkpoint now stores the drafter's state too, which comes out of
`-cram` / `--cache-ram` (default 8192 MiB). **Untested** — worth watching if you enable DFlash
with a large checkpoint budget.

> **Verdict on §16.2: corroborated and strengthened.** It is the best claim in the report. The
> reason is not that the subsets differ; it is that a prefill-side selector destroys the
> *prefix property* the cache is keyed on, while a decode-side speculator is explicitly
> serialised into the same checkpoint as the target. Decode-phase and prefill-phase
> acceleration are not symmetric partners on either side of a diagram. One of them is free to
> compose with prompt caching. The other is in direct competition with it.

---

## 13.8 §16.1 and §4 — the quality-gate critique lands, and it lands on PFlash first

§4 says NIAH is much easier than repository-level reasoning, that passing it does not prove a
model can relate files, classes, tool calls and previous agent steps, and that better tests
include multi-needle retrieval, sequential NIAH, long-context reasoning, repository-level
coding and repeated agent workflows. §16.1 applies the same argument to PFlash: a system can
pass NIAH while getting worse at real coding, because what got discarded was a class
definition or a tool description.

**Verdict: corroborated — and we said it first, in almost the same words.** §10.9 already
carried this disclaimer:

> "It does **not** test multi-step reasoning over long context, summarisation faithfulness, or
> code-edit fidelity. Cleared for retrieval-style work; not certified lossless."

So §4 is not new information about our campaign. It is a correct restatement of a limitation we
had already flagged. Chapter 11 hit the same pattern — a quality-shaped claim we could not
check.

**What *is* new is where the critique bites hardest.** PFlash's own published validation is
**single-needle NIAH at 32k, 64k and 128k, and nothing else** — a magic key plus a 7-digit
answer placed randomly in filler, all green ticks, with other benchmarks described as "verified
upstream" in the original research rather than re-measured. The report's §16.1 argument is
therefore, without saying so, an argument against adopting PFlash **on the evidence PFlash
itself offers**. That is a genuinely useful thing to have noticed, and it is the reason PFlash
belongs behind a stronger gate than the one we have.

### What a better gate looks like with what is already on this disk

The most valuable tool here is one we have never used. `llama-perplexity.exe` is in the
baseline root, and its `--help` confirms:

```
--kl-divergence                  computes KL-divergence to logits provided via --kl-divergence-base
--save-all-logits, --kl-divergence-base FNAME
--chunks N                       max number of chunks to process (default: -1, -1 = all)
```

That is a two-pass distributional gate:

1. Run the **reference** configuration with `--kl-divergence-base logits.bin` to dump all
   logits over a fixed corpus.
2. Run the **candidate** with `--kl-divergence --kl-divergence-base logits.bin`.

You get a per-token divergence distribution instead of a single pass/fail. Where it applies:

| Question | Right tool | Why |
| --- | --- | --- |
| Does `q4_0` KV degrade quality vs `q8_0`? | **KLD** | Continuous, sensitive, no prompt design needed. Chapter 11 wanted exactly this |
| Does a different quant degrade quality? | **KLD** | The axis chapter 11 could not measure |
| Does MTP / DFlash / DSpark degrade quality? | **Neither — it is provably exact** | The target verifies every drafted token; the output distribution is unchanged by construction (§10.7). A KLD of 0 here proves the harness works, not that drafting is safe |
| Does PFlash degrade quality? | **KLD, and it is the only honest gate** | PFlash changes *what the target sees*. It is not distribution-preserving. This is precisely the case single-needle NIAH cannot see |

Two cheaper additions, both pure scripting against `scripts/needle-test.ps1`:

- **Multi-needle**: plant four or five distinct passphrases at different depths and require all
  of them. Catches the "retained a small fraction" failure that one needle cannot.
- **Sequential needle**: plant facts that must be *combined* (`A is in B`, `B is in C`, ask
  where `A` is). This is the cheapest available proxy for the relational reasoning §4 is
  actually worried about.

And the one the report names that no NIAH variant measures: a **repeated-turn gate** — the same
conversation extended over five or six turns with tool output appended, checking both answer
quality and that `f_sim_best` stays at 1.000. That is the gate §16.2 demands, and we do not
have it.

> **One warning about the corpus.** `benchmarks/needle-prompt.txt` is 107,763 tokens of
> near-duplicate telemetry. It is an excellent haystack and a terrible perplexity corpus —
> near-duplicate text makes every model look good. A KLD run needs ordinary prose or code, and
> that is a (small) download.

---

## 13.9 §12, §19 and §21 — the mental model

| Claim | Verdict | Evidence |
| --- | --- | --- |
| §12: MTP is the safest speculative starting point for a 24 GB / 128K setup | **Corroborated, with a number** | +56% to +111% (§10.7), needle PASS on all seven variants, no extra weights, and the only method that fits at 130k (§13.4) |
| §12: MTP accelerates decode, not prefill | **Corroborated — and it *costs* prefill** | −27% at 108k depth (§10.7). The report never mentions a prefill cost |
| §19: MTP and DFlash2 are primarily alternatives | **Right answer, wrong reason** | One draft context; `ctx_type` forced by the presence of `draft-mtp` (§13.5) |
| §19: PFlash and MTP/DFlash2 are complementary in principle | **Corroborated in principle, misleading in practice** | Different phases, yes. But one composes with prompt caching and the other competes with it (§13.7). The diagram gives them equal standing |
| §21 rank 7: "use MTP or DFlash2 for decode" | **Corroborated** | Though on this box it is *MTP*, full stop, at ≥64k (§13.4) |
| §21 rank 9: "experiment with PFlash for cold long-context prefill" | **Corroborated, and the qualifier is the whole claim** | *Cold* is doing all the work. §13.7 shows it is a regression for turn ≥ 2 |
| §22: "optimize prefill, decode, memory usage and GPU placement separately" | **Corroborated** | This is the correct framing and the best line in the report |

> **The one place the report's own framing undercuts it.** §22 closes with "preserving context
> quality and KV-prefix reuse can be more important than obtaining the highest possible one-shot
> benchmark number" — which is exactly right, and is exactly why PFlash appears in the
> recommended configuration at all only as "experimental prefill". The report reasoned its way
> to the correct caution and then still put the feature on the list. Our data says the caution
> should win: on this machine, for agentic coding, PFlash is not a *later* item on the roadmap.
> It is an item for a **different workload** — one-shot analysis of a document you will never
> ask a second question about.

---

## 13.10 The experiment queue

Ordered by value per minute. Everything in the first group runs today on the baseline binaries
in `S:\OneDrive\Tools\llamacpp\` with nothing downloaded and nothing built.

### Today — no download, no rebuild

| # | Experiment | Command shape | Cost | Answers |
| --- | --- | --- | ---: | --- |
| **E1** | **Does the cascade demote MTP?** Three points at `-c 65536`, `-ts 22,43`, `q8_0` KV, 400-token code generation, one fresh server each — baseline / `draft-mtp` / `ngram-mod,draft-mtp`. Add `-lv 4` to get the per-implementation stats line | `--spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 -lv 4` | **~25 min** | §13.5 step 4. `common_speculative_print_stats` prints `#gen drafts` and `#acc tokens` **per implementation**, so you see directly how many token positions the n-gram stole from MTP and at what acceptance |
| **E2** | **Confirm the silent no-op.** Start with `--spec-type draft-dflash` and no `-md`. Check the slot JSON for `"speculative": false` and grep the default-verbosity log for any warning | `--spec-type draft-dflash` (no `-md`) | **~5 min** | §13.3. Records the exact (absent) log signature so a future DFlash run that quietly does nothing is caught in seconds |
| **E3** | **`--spec-type none` vs omitting it.** Both should give no drafting; `none` should additionally suppress auto-detection | — | **~5 min** | Confirms our §10.7 baselines were genuinely undrafted, and gives a safe way to A/B once a drafter file is on disk |
| **E4** | **Multi-needle and sequential-needle gates.** Extend `scripts/needle-test.ps1`: 4–5 passphrases at different depths; then a 3-hop chained variant | script change | **~30 min to write, ~10 min per config** | §13.8. Turns our weakest evidence into something that can actually fail. Prerequisite for trusting any prefill-side method later |

### Today, if a prose/code corpus is already local — otherwise a small download

| # | Experiment | Cost | Answers |
| --- | --- | ---: | --- |
| **E5** | **Stand up the KLD gate.** `llama-perplexity.exe --kl-divergence-base logits.bin` on the reference (`q8_0`/`q8_0`), then `--kl-divergence` on `q4_0`/`q4_0`. Start with `--chunks 20` to size the run before committing | **~45–90 min** | §13.8. Retires the "NIAH is too weak" objection permanently, and finally measures the `q4_0` KV question chapter 11 could only reason about. Not a substitute for E4 — different failure modes |

### Not today — needs a ~1.1 GB download

| # | Experiment | Cost | Notes |
| --- | --- | ---: | --- |
| **E6** | **DFlash2 at 64k.** `z-lab/Qwen3.8-27B-DFlash2-GGUF` → `Q4_K_M` (1090 MiB). `-md <path>` alone (type auto-detects), `-c 65536`, `-ts 22,43`, `q8_0` KV. First run with `--spec-draft-n-max 99` to read the clamp warning and learn the block size, then set it exactly | **~60 min** incl. download | Compare against **MTP at the same 64k**, never against MTP at 130k (§13.4). Read `n_swa` and the draft KV size out of the load log first. Then try `-c 130048` once, purely to record the predicted failure |
| **E7** | **DSpark at 64k.** `magnitudedev/Qwen3.8-27B-DSpark-GGUF` → `Q8_0` (1388 MiB). Same harness | **~45 min** | Anchor-first layout should allow one more draft token per block than DFlash. **Caution:** `src/models/dflash.cpp` changed three days after our baseline commit `fe8156f78` (the DSV4 rope path moved to `ggml_rope_set_offset`). If DSpark misbehaves on the baseline binary, that diff is the first suspect — and one of the few legitimate reasons in this project to build |
| **E8** | **Draft-KV precision.** `-ctkd q8_0 -ctvd q8_0` against the f16 default, on whichever of E6/E7 fits | **~20 min** | Keep the pair symmetric. §10.7 found quantising the *draft* KV bought nothing for MTP; a real 1090 MiB drafter with its own cache is a different case |

### Not today — needs an external project built from source

| # | Experiment | Notes |
| --- | --- | --- |
| **E9** | **PFlash.** Separate repo, separate daemon, sm_80+ (both our cards qualify) | Only worth it for a **cold single-shot** long-prompt workload. Two conditions before it touches anything agentic: measure against **our** 830 t/s baseline, not the 510 t/s one in its README (§13.6); and gate it on E4 + E5, not on single-needle NIAH, because single-needle NIAH is the only evidence it currently ships (§13.8) |

---

## 13.11 Scorecard

| | |
| --- | --- |
| **What the report adds** | DFlash2 exists, its GGUF sizes are exact to the byte, and our baseline binary already accepts it (§13.2). The prefill/decode phase split as an organising idea (§15, §22). And above all **§16.2** — the prefix-reuse objection to prefill-side selection, which is the most valuable claim in all three audited documents and which our own data upgrades from "may" to "does" (§13.7) |
| **What it gets wrong** | That MTP and DFlash2 cannot be listed together for mechanical reasons of drafting theory — they cannot be listed together because there is one `llama_context` on the draft side (§13.5). That `MTP + n-gram` is a harmless bonus — the n-gram *outranks* MTP in a fixed priority cascade, and §10.8 measured the n-gram at exactly 0% on this workload (§13.5). And an implicit framing throughout §19/§21 that PFlash is a *later* item on the same roadmap rather than an item for a different workload (§13.9) |
| **What was already here** | §4's NIAH critique, almost verbatim from §10.9. §12's MTP recommendation, with the number the report lacks. §13's "DFlash2 suits shorter contexts", which our VRAM arithmetic reaches independently and more sharply |
| **Unverifiable / untested** | Whether DFlash2 beats MTP on any workload on this box — no run, no download. The drafter's KV geometry (`n_swa`) and therefore its true VRAM cost. PFlash's 10x, on any hardware, by anyone but its author. Whether repository-level coding quality actually degrades under prefill selection — the exact question §16.1 raises and that nobody in this chain of documents has measured |
| **Our own gap, surfaced by the audit** | We have `llama-perplexity.exe --kl-divergence-base` in the baseline root and have never run it once. Every quality claim in this wiki rests on a single needle in a single haystack, and we knew it, and the tool to fix it has been sitting on the disk the whole time (§13.8, E5) |

> **The closing point.** Chapter 11's report reasoned about quality per byte. Chapter 12's
> reasoned about compiler flags. This one reasons about **algorithms** — and it is the first of
> the three to be mostly right, because algorithms are the one thing that does travel between
> machines. What does not travel is the budget. Every method in §12–§19 is real, and exactly
> one of them fits in 23215 MiB at 130k context.
>
> The lesson that generalises past this model is the one in §13.7. A speedup that composes with
> your cache and a speedup that competes with your cache look identical on a cold benchmark and
> opposite in production. **Measure the second request, not the first.** Ours went from 130
> seconds to 14, and no one-shot benchmark in any of these three reports would ever have
> noticed.

---

Previous: [Chapter 12 — Auditing a build report](12-build-flags-analysis.md) ·
Back to [README](README.md).
