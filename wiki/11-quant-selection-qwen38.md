# Chapter 11 — Choosing the *file*: auditing a third-party quant report

Chapters 6 to 10 tuned **flags**. This chapter is about the **file** — which of the two dozen
GGUF quantizations of Qwen3.8-27B on Hugging Face you should actually put on disk.

The occasion is a third-party research report that recommends specific quants for *this*
machine (RTX 3070 8 GiB + RTX 5060 Ti 16 GiB) and hands over a launch command. It is a
thoughtful document. Parts of it are better than anything in our own campaign, because it
looks at an axis we never measured at all — **quantization quality**. Parts of it would, if
pasted into a terminal, cost you an order of magnitude of prefill and then a failed load.

So this chapter is an audit, and it uses three verdicts:

| Verdict | Means |
| --- | --- |
| **Corroborated** | We have a measurement or a metadata dump on this box that agrees |
| **Contradicted** | We have a measurement or a metadata dump on this box that disagrees |
| **Untested** | Neither we nor (verifiably) the report has evidence. Say so and stop |

> **The rule this chapter is built on:** a recommendation is only as good as the machine it was
> measured on. Quality claims travel between machines; **fit and throughput claims do not.**
> Almost every error below is a quality-shaped claim wearing a performance-shaped costume.

Everything new here was produced with `llama-fit-params.exe -fitp on` (allocation projections,
one fresh process per point), a GGUF header parser reading tensor types directly out of the
files, and HTTP range requests against Hugging Face to read remote GGUF headers without
downloading 17 GB.

---

## 11.1 The report, in one paragraph

It recommends `unsloth/Qwen3.8-27B-GGUF` → **`UD-Q4_K_XL`** (17.6 GB) as the primary choice,
ranking `UD-Q5_K_M` (19.8 GB) second and `UD-Q4_K_M` (16.5 GB) third, on the grounds that
Unsloth's "Dynamic V3" pipeline is >10% better on KLD-style metrics. It raises
`lmcoleman/Qwen3.8-27B-MagicQuant-GGUF` (a custom Q4 hybrid, 15.7 GB, perplexity 6.7611 against
a BF16 baseline of 6.7443), Bartowski's imatrix quants, and `unsloth/Qwen3.8-27B-NVFP4`
(22.6 GB, ~2.5× faster on Blackwell). It describes the architecture as 64 layers of which 16 are
full attention with 4 KV heads of dim 256 and 48 are Gated DeltaNet with non-growing state. It
supplies KV arithmetic for a mixed `q8_0` key / `q4_0` value cache, a table of perplexities, and
this launch command:

```
-ngl all -c 131072 -fa on --split-mode layer --tensor-split 2,1 \
  --cache-type-k q8_0 --cache-type-v q4_0 -np 1 --jinja
```

Two of those eight flags are wrong on this machine in ways that cost more than every other
recommendation in the report is worth. The architecture section, by contrast, is essentially
perfect.

---

## 11.2 The claim-by-claim verdict

### Architecture

| Claim | Verdict | Evidence |
| --- | --- | --- |
| 64 layers | **Corroborated** | `print_info: n_layer = 64`. But `n_layer_all = 65` — the report misses `blk.64`, which is exactly where our +110% lives (§11.5) |
| Only 16 layers do full attention | **Corroborated, twice** | GGUF metadata key `qwen35.full_attention_interval = 4`; and 16 `blk.N.attn_k.weight` tensors, at indices 3, 7, 11 … 63 |
| 4 KV heads of dim 256 | **Corroborated** | `n_head_kv = 4`, `n_embd_head_k = n_embd_head_v = 256`, so `n_embd_k_gqa = 1024` |
| Other 48 layers are Gated DeltaNet with **non-growing** state | **Corroborated** | 48 `ssm_conv1d` layers; `ssm_d_state = 128`, `ssm_n_group = 16`, plus per-layer `ssm_alpha` / `ssm_beta` gates. The *non-growing* part is confirmed operationally: our measured KV is **exactly** 34 KiB/token at `q8_0`, which is `16 × 1024 × 2 × 1.0625` — the 48 recurrent layers contribute **zero** per-token growth (§10.2) |

This is the strongest part of the report. It gets the geometry right, and the geometry is what
makes this model's KV cache 3× the size of the 20 GiB MoE's despite the file being 4.5 GiB
smaller.

### The KV arithmetic

The report's numbers for a `q8_0` key + `q4_0` value cache:

| Context | Report | Computed here | |
| --- | --- | --- | --- |
| 32K | 0.81 GiB | 0.812 GiB | ✅ |
| 64K | 1.63 GiB | 1.625 GiB | ✅ |
| 128K | 3.25 GiB | 3.250 GiB | ✅ |
| 256K | 6.5 GiB | 6.500 GiB | ✅ |

**Corroborated to three decimal places.** `16 × 1024 × (34/32 + 18/32)` = 26 KiB/token, and
every row follows. It also cross-checks against measurement: our symmetric `q8_0` cache at
`-c 130048` is 4318 MiB (§10.2), the arithmetic says 4318 MiB, and `llama-fit-params` projects
3450 MiB for the mixed pair — right where 26 KiB/token puts it.

> **This is the most instructive single fact in the chapter.** The arithmetic is flawless and
> the configuration it justifies is catastrophic. Correct memory arithmetic does not tell you
> whether a CUDA kernel exists for the layout you just budgeted. §11.3.

### Performance and configuration

| Claim | Verdict | Evidence |
| --- | --- | --- |
| `--cache-type-k q8_0 --cache-type-v q4_0` | **Contradicted — dangerously** | Asymmetric KV has no compiled flash-attention kernel. Measured on **this** model: host compute buffer **14507 MiB** vs 529 MiB symmetric (§11.3) |
| `--tensor-split 2,1` | **Contradicted** | Device 0 here is the **3070**. `-ts 2,1` projects **13864 MiB onto an 8 GiB card**; our measured optimum is `-ts 22,43` (§11.4) |
| Try `q4_0/q4_0` — it may be *faster* (5090: F16 125.5 → q4_0 136.7 t/s @32K) | **Contradicted on this box** | Measured at `-c 130048`, `-ub 512`: `q8_0` 1208 PP / 22.14 TG, `q4_0` 1187 / 22.02. Inside the noise floor. `q4_0` buys 2032 MiB and **nothing else** (§10.4). The 5090 figure is a single-GPU Blackwell number and does not transfer |
| `-ngl all` | **Corroborated** | Valid syntax in this build: `-ngl N` accepts an exact number, `'auto'`, or `'all'` |
| `-c 131072` | **Corroborated** | `q8_0` KV reaches 130048 comfortably (peak 22453 MiB) and 139264 thinly. 131072 is fine |
| `-fa on` | **Corroborated** | `-fa off` costs ~9 GiB of compute buffers. Never optional |
| `-np 1` | **Corroborated** | `-np N` divides the context budget, and is the precondition for prompt caching — which is worth more than every flag (§10.10) |
| `--jinja` | **Untested** | Needed for tool calling. It was enabled for **no** measurement in chapters 6–10, so we have no throughput or VRAM number with it on |
| NVFP4 is ~2.5× faster on Blackwell but too big for one 16 GiB card | **Untestable — and moot** | `unsloth/Qwen3.8-27B-NVFP4` is tagged **safetensors**, not GGUF: `model.safetensors` 22.57 GB plus a separate `model_mtp.safetensors`. `llama-quantize --help` on this build lists `MXFP4_MOE` and **no NVFP4 type at all**. llama.cpp cannot load it. And even if it could, device 0 is Ampere with no FP4 units, so 22 of 65 layers would see none of the speedup |

### Quality

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Unsloth's "Dynamic" pipeline really does reallocate bits per tensor | **Corroborated, on disk** | Verified by parsing tensor types out of four files. It is not marketing (§11.5) |
| "Dynamic V3 is >10% better on KLD / Div-300" | **Untested** | The model card claims ">10% top-1% better accuracy at the same size compared to every other provider" and links to Unsloth's own docs. Vendor self-report; no independent replication found. **We have never measured quality across quants at all** |
| MagicQuant: 15.7 GB, PPL 6.7611 vs BF16 6.7443 | **Corroborated as quoted** | Read from the primary source. The card states 14.65 GiB, +0.25%, wikitext-2, **100 chunks, ctx 512** |
| PPL: Q8_0 6.9557, Q4_K_M 6.9576 (99.97%), IQ4_XS 7.0130, UD-Q3_K_XL 7.1113 | **Unverifiable, and internally inconsistent** | None of those four numbers appears on Bartowski's or Unsloth's model card. Worse: if BF16 is 6.7443, then Q8_0 at 6.9557 is **+3.13%** — a figure no Q8_0 has ever produced. The two series are on different corpora and must never be compared (§11.9) |
| `UD-Q4_K_XL` is the right primary choice for this box | **Contradicted on fit; untested on quality** | It cannot run our best profile. §11.6, §11.7 |
| `UD-Q5_K_M` is the second choice | **Contradicted** | It cannot hold 130k in **any** KV configuration on this box — 24172 MiB minimum against 23215 free (§11.6) |

### What the report never mentions

| Omission | Why it matters |
| --- | --- |
| `--spec-type draft-mtp --spec-draft-n-max 4` | **+110% generation** (22.4 → 47.1 t/s) — the largest single measured win on this model (§10.7). The report's launch command does not mention it |
| That the file must leave room for the MTP draft graph | It costs **1107 MiB** and no knob shrinks it (§10.7). This is what decides whether a larger quant is affordable |
| That the MTP head's own quantization differs by provider | Verified: `blk.64` is Q4_K in lmstudio-community's file, Q6_K/Q8_0 throughout in the UD files, and Bartowski's card states a deliberate Q4_0 policy "for its speed which massively benefits MTP performance". Three policies, three possible answers, **zero** measurements |
| That IQ- and Q3_K kernels can be *slower* than Q4_K | Chapter 6: Q3_K_XL at 15.69 GiB generated at **94 t/s** while Q4_K_M at 20.21 GiB did **117 t/s**. The UD files contain IQ4_XS, IQ4_NL, Q3_K and IQ3_S tensors (§11.5) |
| Prompt caching | A 130 s first request becomes ~14 s on the second (§10.10). Worth more than the file choice |
| `-fit off` | Without it llama.cpp may silently re-fit and serve something other than what you asked for |

---

## 11.3 The single most dangerous line in the report

```
--cache-type-k q8_0 --cache-type-v q4_0
```

**That is the whole trap.** A stock CUDA build compiles flash-attention kernels for exactly four
**symmetric** K/V pairs — `f16/f16`, `bf16/bf16`, `q8_0/q8_0`, `q4_0/q4_0` — and
`ggml/src/ggml-cuda/fattn.cu` enforces it with one line:

```c
if (K->type != V->type) {
    return BEST_FATTN_KERNEL_NONE;   // when GGML_CUDA_FA_ALL_QUANTS is undefined
}
```

When no kernel qualifies, **attention leaves the GPU**. The server starts. The output is
correct. There is no warning.

Chapter 10 §10.4 stated that this rule "still holds and was not re-tested" on the dense model.
It has now been re-tested, on this exact file and this exact binary, using the diagnostic from
`known-traps.md` — `llama-fit-params -fitp on` at `-c 130048 -fa on`, which prints projected
allocation as *(model, context, compute)* per device:

| `-ctk`/`-ctv` | CUDA0 compute | CUDA1 compute | **Host compute** |
| --- | --- | --- | --- |
| `q8_0`/`q8_0` | 1137 | 1177 | **529** |
| **`q8_0`/`q4_0`** — the report's line | 403 | 1114 | **14507** |
| `q4_0`/`q4_0` | 1137 | 1177 | **529** |

**14507 MiB of host compute buffer — 27× the symmetric figure.** And note the second tell: the
*GPU* compute buffer on device 0 collapses from 1137 to 403 MiB, because the attention op is no
longer being scheduled there at all. Both signatures fire together, exactly as the reference
predicts.

What that costs in throughput was measured on the MoE in chapter 6, at identical `-ub 512`:

```
-ctk q8_0 -ctv q8_0     1850 t/s prefill
-ctk q4_0 -ctv q8_0      160 t/s prefill     11.6x slower
best working config     2650 t/s prefill     16.6x slower
```

> **Be precise about what has been shown here.** We measured the **allocation signature** on the
> 27B, not its throughput — the 11.6× is a chapter-6 MoE number. But the mechanism is a CUDA
> kernel-*selection* rule, not a model property: it fires on the tensor types, and the 14507 MiB
> proves it fired. There is no plausible world in which attention runs on the host and prefill
> is unaffected.

**And now the fair part, because the report deserves it: `q8_0` keys with `q4_0` values is the
right idea.** Keys tolerate quantization worse than values, the layout saves 1016 MiB at 130k
against symmetric `q8_0` — almost exactly the 1107 MiB the MTP draft graph costs — and it is
listed as the **top open item** in §10.13. The report and this wiki want the same thing. The
difference is that the report presents it as a flag you can pass, and it is actually a **build**:

```powershell
.\.claude\skills\building-llamacpp-cuda\scripts\build-llamacpp.ps1 `
  -CMakeExtra '-DGGML_CUDA_FA_ALL_QUANTS=ON','-DGGML_SCHED_MAX_COPIES=1'
```

> **The lesson: check whether a configuration has a kernel before you check whether it has
> room.** Memory arithmetic is necessary and not sufficient, in exactly the way KV symmetry is
> necessary and not sufficient. And you can check it in two seconds with
> `llama-fit-params -fitp on`: **a Host figure in the thousands of MiB is the trap.**

---

## 11.4 The runner-up: `--tensor-split 2,1`

Same tool, `-c 131072 -ngl all -sm layer`, `q8_0` KV, projected MiB per device as
*(model, context, compute)*:

| `-ts` | CUDA0 = **RTX 3070, ~8018 MiB free** | CUDA1 = 5060 Ti, ~15023 MiB free |
| --- | --- | --- |
| **`2,1`** — the report's line | 9625 + 3094 + 1145 = **13864 MiB** ❌ | 5462 + 1406 + 1185 = 8053 |
| **`22,43`** — measured optimum | 5120 + 1416 + 1145 = **7681 MiB** ✅ | 9968 + 3085 + 1185 = 14238 |

**`-ts 2,1` asks 13.9 GiB of an 8 GiB card.** With *Prefer No Sysmem Fallback* set it fails to
load. Without it — the Windows default — the driver silently puts ~5.8 GiB across the PCIe bus
and you get chapter 5's 3–10× slowdown with no message.

The `22,43` row is not a projection to be taken on trust: it matches the measured peaks of
7421 / 15175 MiB from §10.6, and chapter 10 also showed that moving a *single* layer the wrong
way — `-ts 23,42` — fails outright.

Two things are worth separating, because the report is not simply wrong here:

- **The ratio is right.** 22:43 is 1:1.95. The report's 2:1 is the same ratio.
- **The order is wrong.** On this box `CUDA0` is the RTX 3070 and `CUDA1` is the 5060 Ti —
  `llama_prepare_model_devices` says so on every launch. The report assumed the big card is
  device 0, which is the more common arrangement and simply is not true here.

> **A `--tensor-split` is never portable advice.** It encodes an enumeration order, a per-device
> *free*-memory figure (not capacity), and how much of the display GPU your desktop is holding.
> Copy a ratio if you must; never copy the numbers. Read the two `using device CUDA{0,1}` lines
> in your own log first, then sweep *downward* from the proportional estimate (§6.8, §10.6).

---

## 11.5 What the report gets genuinely right — and we never measured

Here is the honest part. **Our campaign has no quality data across quants whatsoever.** We ran
one gate — needle retrieval of `CRIMSON-PELICAN-4417` from 108k tokens (§10.9) — and we ran it
only across *KV cache* precisions on a *single* file. We have never compared two quantizations
of the same weights on any quality axis. The report's central concern is a real gap in this
wiki, not a distraction from it.

More than that: **the report's core mechanism claim is true, and it is verifiable without
downloading anything.** Parse the tensor types straight out of the GGUF headers (the header sits
in the first ~28 MB, so a range request is enough for a remote file):

| File | Size | imatrix? | Tensor-type histogram |
| --- | --- | --- | --- |
| **lmstudio-community `Q4_K_M`** — *what we benchmarked* | 16032 MiB | **no** | Q4_K 439, F32 360, Q6_K 67 |
| unsloth `Q4_K_M` — *on disk, see §11.8* | 16314 MiB | yes | F32 456, Q4_K 294, Q6_K 67, Q5_K 48, Q8_0 1 |
| unsloth `UD-Q4_K_M` | 15702 MiB | yes | F32 360, Q5_K 131, **IQ4_XS 117**, Q8_0 106, Q4_K 104, Q6_K 30, **Q3_K 7**, IQ4_NL 7, **IQ3_S 4** |
| unsloth `UD-Q4_K_XL` | 16746 MiB | yes | F32 360, Q5_K 191, Q8_0 110, **IQ4_XS 70**, Q4_K 69, Q6_K 56, IQ4_NL 6, **Q3_K 3**, IQ3_S 1 |
| MagicQuant `Q4_K_M` | 15001 MiB | no | Q4_K 504, F32 360, Q6_K 2 |

All five contain **866 tensors**, which matters: the "noMTP" variant on disk has 851, and chapter
10 noted the `unused tensor blk.64...` warning appearing **fifteen times** — 866 − 851 = 15.
**Every candidate here keeps the MTP head**, so switching files does not cost you the +110%.

Now the interesting part. Diff the types tensor by tensor against what we actually run.

**lmstudio-community `Q4_K_M` → unsloth `Q4_K_M`** — 145 tensors differ, in only four patterns:

| Tensor | From | To | Count |
| --- | --- | --- | --- |
| `blk.N.ssm_alpha.weight` | Q4_K | **F32** | 48 |
| `blk.N.ssm_beta.weight` | Q4_K | **F32** | 48 |
| `blk.N.ssm_out.weight` | Q4_K | Q5_K | 48 |
| `blk.64.nextn.eh_proj.weight` | Q4_K | Q8_0 | 1 |

**lmstudio-community `Q4_K_M` → `UD-Q4_K_XL`** — 45 distinct patterns, and the top two are the
same two tensors:

| Tensor | From | To | Count |
| --- | --- | --- | --- |
| `blk.N.ssm_alpha.weight` | Q4_K | **Q8_0** | 48 |
| `blk.N.ssm_beta.weight` | Q4_K | **Q8_0** | 48 |
| `blk.N.ssm_out.weight` | Q4_K | Q5_K / Q6_K / Q8_0 | 47 |
| …then 42 further patterns, in **both** directions — `ffn_gate` down to IQ4_XS (25), `ffn_up` down to Q3_K (2), `ffn_down` down to IQ3_S (1), against `attn_v` up to Q8_0 (11), `attn_output` up to Q6_K (9)… | | | |

`ssm_alpha` and `ssm_beta` are the **gates of the Gated DeltaNet recurrence** — the tensors that
decide how much of the recurrent state to keep and how much to overwrite, in the 48 layers that
carry this model's long-range memory. Both Unsloth files treat them as sensitivity-critical and
refuse to put them at 4 bits. **The file we have been benchmarking quantizes them to Q4_K**, and
so does MagicQuant.

> **This is the most valuable thing in the report, and it arrives by a route the report does not
> take.** Quantizing a *gate* is not like quantizing a weight matrix: a gate's error does not
> average out over a matmul, it **compounds through a recurrence**. If Qwen3.8-27B has a
> quantization-sensitive spot, the DeltaNet gates are the obvious candidate, they are exactly
> where our current file is weakest, and fixing them costs **282 MiB**.
>
> It is a hypothesis, not a finding. We have measured nothing. But it is the first
> quant-quality hypothesis in this wiki with a mechanism behind it, and it is cheap to test.

Two further things the diffs show that neither we nor the report anticipated.

**MagicQuant is not "dynamic" at all — it is the *opposite* of UD.** Its only differences from
lmstudio-community's file are 66 tensors moved **down**: `ffn_down` Q6_K → Q4_K on 33 layers,
`attn_qkv` Q6_K → Q4_K on 24, `attn_v` Q6_K → Q4_K on 9, plus `eh_proj` up to Q6_K. Its 1031 MiB
saving comes from *removing* the precision llama.cpp's own Q4_K_M recipe deliberately adds, and
it leaves the DeltaNet gates at Q4_K. Its quality claim — verified as quoted, +0.25% PPL — rests
on wikitext-2, 100 chunks, **at context 512**, measured by the party that made the file. A
512-token window cannot see recurrent-state degradation at all. That does not make the number
wrong; it makes it silent on the one failure mode this architecture is most exposed to.

**But MagicQuant is still the most interesting candidate on the board, for a reason the report
never gives.** At 15001 MiB it is **1031 MiB smaller** than what we run — almost exactly the
1107 MiB the MTP draft graph costs. Chapter 10's headline discomfort was that profile A peaks at
23560 MiB against 23215 free, leaving no room for a browser. A file 1031 MiB smaller makes
profile A comfortable. That is §10.13's top open item solved by a *download* instead of a
*rebuild*.

**And a warning the report should have carried.** The UD files contain IQ4_XS, IQ4_NL, Q3_K and
IQ3_S tensors. Chapter 6 measured Q3_K_XL (15.69 GiB) generating at **94 t/s** against Q4_K_M
(20.21 GiB) at **117 t/s** — the *smaller* file was 24% slower, because Q3_K CUDA kernels are
slower than Q4_K. The report evaluates quants purely on quality-per-byte and never once on
kernel throughput. **On this box that axis has already cost 24% once.**

---

## 11.6 The arithmetic the report never does

The report budgets the KV cache beautifully and then never adds the weights, the compute buffers,
or the draft graph. Do it properly. Take each measured peak from chapter 10 and shift it by the
difference in file size — legitimate, because weights are the only term that changes when you
swap files.

Free VRAM at campaign start: **23215 MiB** (8018 + 15023). All rows at `-c 130048`, `-ub 512`,
`-ts 22,43`.

| Candidate | MiB | Δ | q8 KV, no MTP | q8 KV, **MTP n4** | q4 KV, **MTP n4** | q4 KV, no MTP |
| --- | --- | --- | --- | --- | --- | --- |
| *measured anchor* | 16032 | — | *22453* | *23560* | *22893* | *21348* |
| **lmstudio-community Q4_K_M** *(current)* | 16032 | +0 | 22453 ✅ | 23560 ⚠️ *ran 5/5* | 22893 ✅ | 21348 ✅ |
| unsloth `Q4_K_M` *(on disk)* | 16314 | +282 | 22735 ✅ | 23842 ❌ | 23175 ⚠️ | 21630 ✅ |
| unsloth `UD-Q4_K_S` | 14647 | −1385 | 21068 ✅ | 22175 ✅ | 21508 ✅ | 19963 ✅ |
| unsloth `UD-Q4_K_M` | 15702 | −330 | 22123 ✅ | 23230 ⚠️ | 22563 ✅ | 21018 ✅ |
| **unsloth `UD-Q4_K_XL`** *(the report's pick)* | 16746 | **+714** | 23167 ⚠️ | **24274 ❌** | **23607 ❌** | 22062 ✅ |
| **unsloth `UD-Q5_K_M`** *(the report's #2)* | 18856 | **+2824** | **25277 ❌** | **26384 ❌** | **25717 ❌** | **24172 ❌** |
| unsloth `UD-IQ4_XS` | 13593 | −2439 | 20014 ✅ | 21121 ✅ | 20454 ✅ | 18909 ✅ |
| **MagicQuant `Q4_K_M`** | 15001 | −1031 | 21422 ✅ | **22529 ✅** | 21862 ✅ | 20317 ✅ |

✅ = 400+ MiB clear · ⚠️ = under 23215 but thin · ❌ = over the free-VRAM line

Read three conclusions off that table:

- **`UD-Q5_K_M` cannot serve 130k on this box in any configuration.** Its cheapest option —
  `q4_0` KV, no drafting — projects 24172 MiB, roughly 957 MiB over. To fit it you would have to
  give up about 55k tokens of context. The report ranks it **second**. That is a plain
  contradiction, and the kind that only shows up if you do the fit arithmetic on the actual
  machine.
- **`UD-Q4_K_XL` cannot run profile A or profile B.** With MTP on it is 1059 MiB (q8 KV) or
  392 MiB (q4 KV) over the line. It fits only by turning drafting **off** — which costs the
  +110% generation win — or by cutting context. Adopting it means trading a measured 2× for an
  unmeasured quality gain.
- **`MagicQuant` and `UD-Q4_K_S` are the only candidates that make profile A *comfortable*.**

> **The general lesson: a quant recommendation is meaningless until you name the KV type, the
> context, and whether drafting is on.** "17.6 GB fits in 24 GiB" is arithmetic that forgets
> 4.3 GiB of KV cache, 2.3 GiB of compute buffers and 1.1 GiB of draft graph. On this box the
> elastic budget for *weights* is about 16.5 GiB, not 24.

---

## 11.7 Should we switch from `lmstudio-community/Q4_K_M` to `UD-Q4_K_XL`?

**No — not on this evidence, and probably not to that file at all.** But the report has
identified a real question, and the answer to *that* is "measure it".

The case against `UD-Q4_K_XL` specifically:

1. **It costs the largest measured win on this model.** +714 MiB puts profile A at 24274 MiB and
   profile B at 23607 MiB, both over the line. You would be giving up a **measured +110%
   generation** for an **unmeasured** quality gain (§11.6).
2. **Its quality advantage is a vendor self-report.** ">10% better" appears on Unsloth's own card,
   pointing at Unsloth's own docs. That is a reason to *test*, not a reason to switch.
3. **It carries a throughput risk the report never mentions** — 70 IQ4_XS, 6 IQ4_NL, 3 Q3_K and
   1 IQ3_S tensor, on a box where non-Q4_K kernels have already cost 24% once (§11.5).
4. **The four PPL numbers used to rank the ladder are unverifiable and internally inconsistent**
   (§11.9).

The case *for* changing something:

1. **The DeltaNet-gate hypothesis is good and cheap.** Both Unsloth files protect `ssm_alpha` /
   `ssm_beta`; ours does not. If quantized gates hurt, that is where it happens.
2. **`UD-Q4_K_M` (15702 MiB, −330) and `UD-Q4_K_S` (14647 MiB, −1385) get the same gate protection
   while being *smaller* than what we run.** The report ranks `UD-Q4_K_M` third and never mentions
   `UD-Q4_K_S` — yet on this machine those are the two UD files that keep every profile in
   chapter 10 intact, and `UD-Q4_K_S` makes profile A comfortable. **If the UD thesis is right,
   the correct pick here is close to the file the report ranked last, not the one it ranked
   first.**
3. **`MagicQuant` (−1031 MiB) buys back the MTP headroom for free** — but does *not* protect the
   gates, so it is a fit win and a quality unknown.

### What to measure, in order

We have `llama-perplexity.exe` locally, so none of this needs new tooling.

| # | Experiment | Command shape | Answers |
| --- | --- | --- | --- |
| 1 | **PPL, same corpus, same context, one fresh process per file** — and **not at ctx 512** | `llama-perplexity -m <file> -f <corpus> -c 4096 --chunks 200 -fa on -ngl all -sm layer -ts 22,43` | Is there *any* resolvable quality difference? Runnable **today** on the two files already on disk |
| 2 | **KL divergence against a common reference** — the metric Unsloth's claim is actually about | `llama-perplexity -m Q8_0.gguf -f <corpus> --kl-divergence-base logits.bin`, then `-m <cand> --kl-divergence --kl-divergence-base logits.bin` | Replicates the vendor's own metric instead of trusting it. Needs the 29.05 GB `Q8_0` (CPU / partial offload; 64 GiB RAM makes it slow but possible) |
| 3 | **A long-context gate that is not verbatim recall** | a multi-hop / aggregation question over ~100k tokens | The needle test is too easy — the answer is literally in context, so it cannot detect a degraded DeltaNet gate. **This is the gap in our own campaign, not the report's** |
| 4 | **Re-run the chapter-10 grid on whatever file wins** | `bench-harness.ps1`, synthetic PP/TG + peak VRAM, then one deep HTTP request | File size changes the fit; §11.6 is a projection, not a measurement |
| 5 | **MTP head precision vs draft acceptance** | same config, different files; compare `draft acceptance` and end-to-end TG | Three providers, three MTP policies (Q4_K / Q6_K–Q8_0 / Q4_0 per Bartowski). Nobody has measured which wins |

> **Do experiment 1 first and be prepared for it to find nothing.** Chapter 6's `q4_0`-KV needle
> test and chapter 10's `-ub` sweep both came back "no measurable difference", and both were
> worth running: a null result at 200 chunks tells you the ladder in the report is below the
> resolution of the measurement, which is itself a decision. **Do not switch files on a
> difference you cannot resolve.**

---

## 11.8 What is actually on disk right now

`Get-ChildItem -Recurse -Filter *Qwen3.8*` under `S:\HuggingFace\lmstudio\`:

| Path | Size | Notes |
| --- | --- | --- |
| `lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf` | 16032 MiB (16.81 GB) | **The file every number in chapter 10 was measured on.** No imatrix. DeltaNet gates at Q4_K |
| `unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf` | 16314 MiB (17.11 GB) | imatrix (`imatrix_unsloth.gguf`, 496 entries / 45 chunks). Gates at **F32**. ⚠️ **This filename now 404s in the repo** and its size matches nothing currently published — treat it as a superseded revision and record the fact next to any result you get from it |
| `JonathanColetti\...\Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` | 15781 MiB (16.55 GB) | 851 tensors — **no MTP head**, so no `draft-mtp`. Also a different fine-tune, so not a clean control |
| `lmstudio-community\...\mmproj-Qwen3.8-27B-BF16.gguf.renamed` | 0.87 GiB | Vision projector; unused by any measurement |

**So a comparison can be run today**, without downloading anything: lmstudio-community's
uncalibrated Q4_K_M against Unsloth's imatrix Q4_K_M. The two differ in exactly 145 tensors and
the difference is precisely the DeltaNet-gate hypothesis (§11.5). It is not the report's
recommended file, but it is a **cleaner** experiment than the report's recommendation would be,
because 282 MiB does not perturb the fit while 714 MiB does.

Anything else needs a download: `UD-Q4_K_XL` 17.56 GB, `UD-Q4_K_M` 16.46 GB, `UD-Q4_K_S`
15.36 GB, MagicQuant 15.73 GB, and for experiment 2 a reference — `Q8_0` 29.05 GB or BF16
54.66 GB in two shards.

---

## 11.9 Citation quality

The report's citations are opaque internal tokens — `turn194141news0`, `citeturn496774search6` —
which cannot be resolved, followed, or dated. That is not a stylistic complaint: for several
claims it is the *only* support offered, and two of those do not survive a check against primary
sources.

| Claim | Support | Status |
| --- | --- | --- |
| Repo names and file sizes (`UD-Q4_K_XL` 17.6 GB, `UD-Q5_K_M` 19.8 GB, `UD-Q4_K_M` 16.5 GB, MagicQuant 15.7 GB, NVFP4 22.6 GB) | opaque token | **Confirmed independently.** Every size matches the Hugging Face API to two decimals |
| MagicQuant PPL 6.7611 vs BF16 6.7443 | opaque token | **Confirmed from the primary source** — it is on the model card, along with the method (wikitext-2, 100 chunks, ctx 512) the report omits |
| Unsloth ">10% better" | opaque token | **Traced to the vendor's own card and docs.** Faithfully paraphrased; independently unverified |
| **PPL: Q8_0 6.9557, Q4_K_M 6.9576, IQ4_XS 7.0130, UD-Q3_K_XL 7.1113** | opaque token **only** | **Not found on Bartowski's card, not found on Unsloth's card.** Provenance unknown |
| **The 5090 result: F16 125.5 → q4_0 136.7 t/s @32K** | opaque token **only** | Unverifiable, and contradicted as guidance for this box (§11.2) |
| NVFP4 "~2.5× faster on Blackwell" | opaque token | Unverifiable **and moot** — the repo is safetensors and this build has no NVFP4 type |

The fourth row deserves emphasis, because the report's whole quant ladder rests on it. Those four
numbers are internally inconsistent with the MagicQuant series the report quotes on the same
page: **if BF16 is 6.7443, a Q8_0 at 6.9557 is +3.13% perplexity**, and Q8_0 does not lose 3% of
anything — it is normally within ~0.05% of BF16. The only consistent reading is that the two
series come from different corpora, chunk counts or context lengths, in which case they cannot be
placed on the same ladder. And "Q4_K_M is 99.97% of Q8_0" compounds the problem twice: a
perplexity *ratio* is not a percentage of quality, and a 0.03% gap is far below what a 100-chunk
measurement can resolve.

> **The lesson is the one `GGML_SCHED_MAX_COPIES` taught us in §6.2, pointed at someone else's
> document instead of our own: a number you cannot trace is not evidence, and a difference below
> your measurement's resolution is not a finding.** Chapter 6 credited +13% to an environment
> variable that does not exist, on a machine with an ±8% noise floor. This is that mistake
> wearing a decimal point.

---

## 11.10 Scorecard, and what changes today

| | |
| --- | --- |
| **Corroborated** | The entire architecture section, verified twice over (`full_attention_interval = 4`, the 16 `attn_k` tensors, 34 KiB/token measured). All four KV-arithmetic rows, exact. Every repo name and file size. `-ngl all`, `-c 131072`, `-fa on`, `-np 1`. The MagicQuant perplexity figure as quoted. **And the central mechanism claim: Unsloth's dynamic reallocation is real and visible on disk** |
| **Contradicted** | `--cache-type-v q4_0` against `--cache-type-k q8_0` — 14507 MiB of host compute buffer on this exact model. `--tensor-split 2,1` — 13864 MiB asked of an 8 GiB card. `UD-Q5_K_M` as a 130k option — impossible in every KV configuration. `UD-Q4_K_XL` as the primary pick — it cannot run the profile that carries our +110%. `q4_0` KV being faster — 1187 vs 1208 t/s, noise. NVFP4 as a llama.cpp option — it is not a GGUF |
| **Untested — by us and, verifiably, by the report** | Whether any of these quants is measurably better than any other **on this machine**. Whether protecting the DeltaNet gates changes long-context behaviour. Whether IQ4_XS / Q3_K content slows generation here. Whether MTP-head precision changes draft acceptance. `--jinja`'s cost |

**Nothing in chapter 10 changes.** Profiles A–D stand exactly as written, on the file already on
disk, and §10.13's top open item — rebuild with `-DGGML_CUDA_FA_ALL_QUANTS=ON` — is now *more*
attractive, because the report independently reasoned its way to the same asymmetric-KV layout
from the quality side.

What the report adds to the queue:

| Priority | Action | Why |
| --- | --- | --- |
| **1** | PPL at ctx 4096, 200 chunks, on the **two files already on disk** | The first quality measurement in this wiki. Costs a download of nothing |
| **2** | Build a long-context gate that is not verbatim recall | The needle test cannot see a degraded recurrence. Our gap, not the report's |
| **3** | Rebuild with `-DGGML_CUDA_FA_ALL_QUANTS=ON` | Unchanged from §10.13, and now double-motivated |
| 4 | If (1) resolves a difference: download `UD-Q4_K_S` and `UD-Q4_K_M`, **not** `UD-Q4_K_XL` | They protect the gates *and* leave the fit intact — or improve it |
| 5 | Download MagicQuant and re-run the chapter-10 grid | −1031 MiB makes profile A comfortable. Fit win, quality unknown |
| 6 | KL divergence against a `Q8_0` reference | Replicate the vendor's metric rather than trusting or dismissing it |
| — | *Do not* pursue NVFP4 | Not a GGUF, no kernel in this build, and device 0 has no FP4 units |

> **The closing point, and the reason this chapter is not a takedown.** The report is wrong about
> this machine in two places and right about something we had not thought about at all. Its
> failures are all of one kind: it reasons about *quality per byte* correctly and then treats
> fit, kernel availability and device order as details. Ours are the mirror image — ten chapters
> of throughput and VRAM, and not one number about whether the weights are any good. **The two
> documents are each other's blind spot, which is exactly why it was worth reading carefully
> instead of dismissing.**

---

Previous: [Chapter 10 — Qwen3.8-27B results](10-results-qwen38-27b.md) ·
Back to [README](README.md).

