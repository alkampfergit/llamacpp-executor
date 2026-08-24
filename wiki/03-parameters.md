# Chapter 3 — The Parameters That Matter

llama-server accepts well over two hundred flags. About fifteen of them decide whether your
setup is fast or useless. This chapter covers those fifteen, plus the traps.

For each flag: **what it does**, **what it costs**, and **how to choose**.

---

## 3.1 Placement — getting the weights onto the GPUs

### `-ngl N` / `--n-gpu-layers N`

How many layers to keep in VRAM. Layers that don't fit run on the CPU, which is roughly
20–50× slower per layer.

```powershell
-ngl 999      # "all of them" — the idiomatic way to say max
-ngl auto     # let llama.cpp decide
```

**Choose:** `-ngl 999` if the model fits, and if it doesn't, *fix that a different way*
(smaller quant, smaller context, quantised KV) rather than by leaving layers on the CPU.

> **Important side effect.** Setting `-ngl` explicitly **disables the auto-fitter**. You
> will see this in the log:
> ```
> failed to fit params to free device memory: n_gpu_layers already set by user to 999, abort
> ```
> This is *not* an error — it is llama.cpp saying "you took the wheel, so I won't steer."
> Loading continues normally. Confirm with the `model loaded` line a few rows later.

### `-sm MODE` / `--split-mode`

`layer` (default), `row`, `tensor`, or `none`. See §1.4. **Use `layer`.**

### `-ts A,B` / `--tensor-split`

The ratio of layers given to each GPU. `-ts 12,29` means 12/41 to GPU 0.

**Choose:** start from *free* VRAM, not total VRAM, then measure. On this machine `-ts 12,29`
works and `-ts 13,28` runs out of memory — a one-layer difference. There is no formula.
Chapter 4 shows how to find the boundary in seconds rather than minutes.

> **Note the separator inconsistency:** `llama-server` wants commas (`-ts 12,29`),
> `llama-bench` wants slashes (`-ts 12/29`). Both accept both in practice, but the help text
> differs and it will confuse you eventually.

### `-mg N` / `--main-gpu`

Which GPU holds intermediate results. **Only meaningful with `-sm none` or `-sm row`.**
With the default `-sm layer` it does essentially nothing — if you have `-mg` in a
`-sm layer` command line, delete it.

### `-ncmoe N` / `--n-cpu-moe N` and `-cmoe` / `--cpu-moe`

Keep the Mixture-of-Experts weights of the first *N* layers in CPU RAM. This is the
standard trick for running an MoE model that doesn't fit — the attention stays on the GPU
and only the bulky expert matrices move.

**It is also a trap on this hardware.** Measured, at 130k context:

| Offload | Prefill t/s | Generation t/s |
| --- | --- | --- |
| none | 1466 | 95 |
| `-ncmoe 2` | 826 | 88 |
| `-ncmoe 4` | 541 | 82 |
| `-ncmoe 6` | 421 | 76 |
| `-ncmoe 8` | 344 | 60 |

Moving **2 layers out of 40** — 5% of the model — cut prefill by 44%.

Why so brutal? Because of the prefill/generation distinction from §1.5. Generating one token
activates 8 of 256 experts, so the CPU reads ~3% of that layer's weights: cheap. But
prefilling a 512-token batch activates *almost every expert*, so the CPU must stream the
entire expert matrix across the memory bus, and it becomes the bottleneck for the whole
pipeline while both GPUs idle.

**Choose:** `-ncmoe 0`. Treat any non-zero value as evidence that something else in your
configuration needs to shrink. If you must use it, note the log hint:
`tensor overrides to CPU are used with mmap enabled — consider using --load-mode none`.

### `-ot PATTERN=BUFFER` / `--override-tensor`

Surgical placement by tensor-name regex. This is what `-ncmoe` expands into internally, and
what `llama-fit-params` emits:

```
-ot "blk\.38\.ffn_(gate|up|gate_up|down).*=CPU,blk\.39\.ffn_.*_exps=CPU"
```

Useful when you need to move *half* a layer to squeeze under a limit. Powerful, fiddly, and
rarely necessary once you've got the other knobs right.

---

## 3.2 Context — how much conversation you can hold

### `-c N` / `--ctx-size N`

The total token window. llama.cpp rounds it up (ask for 130000, get 130048).

**What it costs:** KV cache, linearly. On the Qwopus model, ~11 MiB per 1000 tokens at
`q8_0`, ~20 MiB at `f16`.

**What it also costs — and nobody tells you this:** *prefill throughput, even for short
prompts.* Measured with an identical 8192-token prompt, changing only `-c`:

| `-c` | Prefill t/s | Loss vs 8k |
| --- | --- | --- |
| 8,448 | 2122 | — |
| 32,768 | 2017 | −5% |
| 65,536 | 1764 | −17% |
| 98,304 | 1664 | −22% |
| 130,048 | 1466 | **−31%** |

This table was initially misread as an inherent context tax. Later fit-controlled results show
that the 31% decline was primarily the approach to the VRAM cliff: a 130k configuration that
fits cleanly prefills as fast as the 8k point. A larger window still consumes KV memory and
reduces compute-buffer headroom, so ask for what you need, but do not predict a 31% compute tax
from `-c` alone. See §6.4 and chapter 18.

### `-np N` / `--parallel N`

Number of slots. **With statically divided caches, per-request context is `-c / -np`.** See
§2.4 — this is the single most common misconfiguration. For one user, `-np 1`.

### `-ctk TYPE` / `-ctv TYPE` — KV cache quantisation

Store the KV cache in a smaller format. `-ctk` is keys, `-ctv` is values.
Allowed: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`.

At 130k context on this model:

| Setting | KV size | Prefill t/s | Generation t/s |
| --- | --- | --- | --- |
| `f16` / `f16` | 2622 MiB | *doesn't fit at `-ub 512`* | — |
| `q8_0` / `q8_0` | 1422 MiB | 1466 | 95 |
| `q4_0` / `q4_0` | ~720 MiB | *untested — on the fast path* | — |
| `q5_1` / `q5_1` | ~1100 MiB | **107** | **19** |
| `q4_0` / `q8_0` | ~1050 MiB | **160** | **25** |

**Read that table carefully.** The bottom two rows are *smaller*, they *load fine*, they
produce *correct output* — and they are **14× slower**. But the reason is not "4-bit KV is
slow", and getting the reason right changes what you should do.

A stock CUDA build compiles flash-attention kernels for only **four K/V type pairs**:

```
f16 / f16        bf16 / bf16        q8_0 / q8_0        q4_0 / q4_0
```

and `ggml-cuda/fattn.cu` enforces the restriction directly:

```c
if (K->type != V->type) { return BEST_FATTN_KERNEL_NONE; }
```

with supported types limited to **F32, F16, Q4_0, Q8_0, BF16**. So there are two distinct
ways to fall off the fast path:

- **`q4_0`/`q8_0`** — asymmetric. `K->type != V->type`, so no kernel qualifies.
- **`q5_1`/`q5_1`** — symmetric, but `q5_1` is not a supported type at all.

**Symmetry is necessary but not sufficient.** When no CUDA kernel qualifies, attention runs
on another backend — which is why `-ctv q4_0` also pushes ~5 GiB into host RAM. The
operation left the GPU.

> **The rule: use a symmetric pair from the compiled set — `f16/f16`, `bf16/bf16`,
> `q8_0/q8_0`, or `q4_0/q4_0`. Never mix types; never use a type outside that list.**
>
> `q8_0/q8_0` is the practical sweet spot: half the KV cache for a few percent of speed and
> no measurable quality loss. **`q4_0/q4_0` is also on the fast path** and saves a further
> ~700 MiB at 130k — untested here, and worth trying, but check output quality because keys
> tolerate aggressive quantisation less well than values.
>
> If a model produces gibberish at long context, the model authors suggest
> `-ctk bf16 -ctv bf16`. Try `q8_0/q8_0` first and escalate only on visible corruption.

**Want asymmetric KV?** You must compile it. `GGML_CUDA_FA_ALL_QUANTS` is a CMake option,
**OFF by default**:

```
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
```

That instantiates the full quantised-FA matrix, making `-ctk q8_0 -ctv q4_0` fast — a very
attractive layout for a 130k window, since it protects the keys and shrinks the values. Cost:
a long compile and a much larger CUDA binary.

**This has now been built and measured** — [chapter 14](14-build-experiment.md). Asymmetric
`q8_0`/`q4_0` went from 38.9 to 1187 t/s prefill (**×30.5**) and became indistinguishable from
the symmetric pairs, so the rule above ("use a symmetric pair from the compiled set") is a
statement about the *default build*, not about the formats. On that build `q5_1` and `q4_1`
also stop falling off the fast path, because all 49 kernels are compiled instead of 4. The
real cost turned out to be 10.4 min of compile and `ggml-cuda.dll` growing 84.2 -> 116.2 MB.

---

## 3.3 Batching — the compute/memory dial

### `-b N` / `--batch-size N` (logical batch)

The most tokens llama.cpp will accept into one `decode` call. Default 2048. This is a
scheduling parameter; it has little memory cost. Leave at 2048.

### `-ub N` / `--ubatch-size N` (physical / micro batch)

**The most important performance knob you have.** The number of tokens actually pushed
through the GPU at once. Compute buffers scale with it *linearly*; prefill speed scales with
it until it saturates.

Measured on this machine (small context, so memory is not the constraint):

| `-ub` | Prefill t/s | Generation t/s | Compute buffers |
| --- | --- | --- | --- |
| 128 | 1368 | 106 | ~0.4 GiB |
| 256 | 2017 | 102 | ~0.8 GiB |
| **512** | **2651** | 105 | ~1.6 GiB |
| 1024 | 2615 | 106 | ~3.2 GiB |
| 2048 | OOM | — | ~6.5 GiB |

Three lessons:

1. **512 is the sweet spot.** 1024 costs twice the memory for nothing.
2. **Generation ignores `-ub` completely** (105 t/s at every setting). Of course it does —
   generation has no batch. So if you only care about chat responsiveness, spend that VRAM
   on context instead.
3. **Below 256 you fall off a cliff.** `-ub 128` is worse than half of `-ub 256`.

**Choose:** `-ub 512` if it fits. Drop to `-ub 256` to buy ~0.8 GiB. Never go to 128 unless
desperate.

### `-fa on|off|auto` / `--flash-attn`

Fused attention kernels. `auto` enables it when supported.

| Setting | Compute buffers at 130k |
| --- | --- |
| `-fa on` | 1.6 GiB |
| `-fa off` | **10.4 GiB** |

**Choose: always `-fa on`.** Be explicit rather than relying on `auto`, so that a silent
fallback shows up as an error instead of as a mysterious 9 GiB.

---

## 3.4 Pipeline parallelism — a compile-time knob, not a flag

With multiple GPUs, llama.cpp enables **pipeline parallelism**: while GPU 1 works on
micro-batch *n*, GPU 0 starts micro-batch *n+1*. To do that it keeps several copies of the
intermediate activations — by default 4 — which multiplies your compute buffers.

On this pair, the extra copies are often exactly what pushes a tight configuration over the
VRAM cliff. Whether fewer copies improve throughput is model- and ubatch-dependent; it must be
measured rather than inferred from the mismatch.

> ### ⚠️ `GGML_SCHED_MAX_COPIES` is **not** an environment variable
> It is a compile-time `#define` in `ggml/src/ggml-backend.cpp`:
> ```c
> #ifndef GGML_SCHED_MAX_COPIES
> #define GGML_SCHED_MAX_COPIES 4
> #endif
> ```
> Setting `$env:GGML_SCHED_MAX_COPIES = "1"` **does nothing.** An earlier version of this
> wiki said otherwise; see §6.2 for the correction and the evidence.

There are two real ways to get the lean path.

**1. Rebuild (deterministic, recommended):**

```
cmake -B build -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1
```

**2. Let memory pressure trigger the automatic fallback (what actually happens today):**

```
graph_reserve: failed to allocate compute buffers
sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
```

When the pipelined buffers do not fit, llama.cpp retries with a smaller scheduler. The retry is
a valid recovered run, but the old 2650-versus-1850 comparison changed KV type and ubatch too.
At identical Qwopus `c64000 / ub512 / ts13,28` settings, runtime fallback and a one-copy build
are only **1.6% apart**. Changing only the split from `12,29` to `13,28` is the repeatable large
effect: **+12.8% prefill**. See chapter 18.

> **Teaching point.** A warning followed by a result row means recovery succeeded. Record the
> execution mode, but tune the split and memory headroom instead of deliberately provoking it.

---

## 3.5 The auto-fitter

Recent llama.cpp can size the configuration for you.

| Flag | Meaning |
| --- | --- |
| `-fit on\|off` | Enable/disable auto-fitting (default `on`) |
| `-fitt MiB0,MiB1` | Free VRAM to leave spare per device (default 1024 each) |
| `-fitc N` | Smallest context the fitter may choose (default 4096) |
| `-fitp on` | **Print the memory estimate and continue** — extremely useful |

The fitter only adjusts arguments you did *not* set. Set `-ngl` and it steps aside entirely.

**How much to trust it:** use it to *plan*, not to *decide*. Measured against reality on this
machine it under-predicted the CUDA0 compute buffer by ~235 MiB (596 predicted, 831 actual),
which is the difference between fitting and not. So: read its numbers, then add a margin,
then verify by loading.

**When to use `-fit off`:** whenever you are benchmarking. You need the configuration you
asked for, not a helpfully adjusted one, or your comparisons are meaningless.

---

## 3.6 Loading

### `--load-mode auto|none|mmap|mlock|mmap+mlock|dio`

Replaces the old `--no-mmap` / `--mlock`. Memory-mapping is the default and is right almost
always. Two exceptions:

- **You are using `-ot` or `-ncmoe`** — llama.cpp itself suggests `--load-mode none` for
  better CPU-side throughput, because mapped pages defeat the CPU backend's weight
  repacking.
- **You have RAM to spare and want to avoid page-outs** — `mlock`.

### `-t N` / `--threads N`, `-tb N` / `--threads-batch N`

CPU threads for generation and for batch work. On a 12-core/24-thread 5900X, llama.cpp picks
12 and that is correct — **do not use 24.** Hyper-threads contend for the same
memory bandwidth, which is the actual bottleneck. Only relevant at all if something is
running on the CPU.

---

## 3.7 Quick reference

| Flag | Set it to | Why |
| --- | --- | --- |
| `-ngl` | `999` | All layers on GPU |
| `-sm` | `layer` | Correct for two GPUs |
| `-ts` | measured | `12,29` here; must be found empirically |
| `-fa` | `on` | Saves 9 GiB |
| `-b` | `2048` | Default is fine |
| `-ub` | `512`, or `256` if tight | Biggest prefill knob |
| `-ctk`/`-ctv` | `q8_0`/`q8_0` | Must be a **symmetric pair from the compiled set**; mixing types costs 14× |
| `-c` | exactly what you need | Costs memory *and* prefill speed |
| `-np` | `1` for single user | Otherwise divides your context |
| `-cram` | `24576` | Free prompt-cache hits |
| `-ncmoe` | `0` | Halves prefill per 2 layers |
| `-mg` | omit | No effect with `-sm layer` |
| `-fit` | `off` when benchmarking | Reproducibility |
| `GGML_SCHED_MAX_COPIES` | **not settable at runtime** | Compile-time only: `-DGGML_SCHED_MAX_COPIES=1` |

---

Next: [Chapter 4 — Budgeting your VRAM](04-vram-budget.md), where we stop guessing and start
predicting.

