# Chapter 1 — The Mental Model

Before you type a single flag, you need a picture in your head of what llama.cpp is doing.
Most tuning mistakes come from a wrong mental model, not from a wrong flag.

---

## 1.1 What llama.cpp actually is

llama.cpp is not an application. It is a **C++ inference library** (`llama.dll`) sitting on
top of a tensor library (`ggml.dll`), plus a set of small command-line programs that each
expose the library in a different way.

The tensor library has *backends* — one per kind of hardware. In this folder you can see
them as separate DLLs:

```
ggml-cuda.dll             ← your two NVIDIA GPUs
ggml-cpu-zen4.dll         ← AMD Zen 4 CPU kernels
ggml-cpu-haswell.dll      ← generic x86-64 CPU kernels
ggml-rpc.dll              ← remote GPUs over the network
```

At startup llama.cpp loads the backends it can find and picks the best CPU variant for your
processor. You will see this in the first lines of every run:

```
ggml_cuda_init: found 2 CUDA devices (Total VRAM: 24502 MiB):
  Device 0: NVIDIA GeForce RTX 3070,    compute capability 8.6, VRAM: 8191 MiB
  Device 1: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0, VRAM: 16310 MiB
load_backend: loaded CUDA backend from ggml-cuda.dll
load_backend: loaded CPU backend from ggml-cpu-haswell.dll
```

**Read those lines every time.** If a GPU is missing, nothing else you do matters.

---

## 1.2 Which binary to use when

There are many `.exe` files here. You only need five, and knowing which is which will save
you hours.

| Binary | Use it for | Key property |
| --- | --- | --- |
| **`llama-server.exe`** | Actually serving the model over HTTP | The production tool. Has slots, prompt caching, an OpenAI-compatible API. |
| **`llama-bench.exe`** | Comparing kernel speed between configurations | Fast, repeatable. **Does not let you set a real context size.** |
| **`llama-batched-bench.exe`** | Measuring speed *under realistic memory conditions* | Accepts `-c`, so it allocates the same KV cache your server will. |
| **`llama-fit-params.exe`** | Asking "will this fit, and how should I split it?" | Answers in **under one second** without loading the weights. Your most under-rated tool. |
| **`llama-cli.exe`** | A quick sanity chat in the terminal | Good for "is the model coherent?", useless for benchmarking. |

The others are specialised: `llama-quantize` makes GGUFs, `llama-perplexity` measures
quality, `llama-tokenize` counts tokens, `llama-mtmd-cli` handles vision models.

> **Teaching point.** A very common mistake is tuning with `llama-bench` and then deploying
> with `llama-server`, and being confused when the server is half as fast. `llama-bench`
> defaults to a tiny context, so it never allocates the multi-gigabyte KV cache your server
> does. You were measuring a different problem. Chapter 5 covers this properly.

---

## 1.3 The single most important idea: the three consumers of VRAM

When you load a model, your VRAM is spent on **three separate things**. Almost every
tuning decision is a trade between them. Internalise this table and the rest of the wiki
becomes obvious.

```
┌──────────────────────────────────────────────────────────────────┐
│  1. MODEL WEIGHTS          fixed by your quantisation choice     │
│     "the model itself"     20.21 GiB for Qwopus Q4_K_M           │
│     Knobs: quant level, -ngl, -ncmoe, -ot                        │
├──────────────────────────────────────────────────────────────────┤
│  2. KV CACHE               grows linearly with -c                │
│     "the conversation"     1.4 GiB for 130k tokens (q8_0)        │
│     Knobs: -c, -ctk, -ctv                                        │
├──────────────────────────────────────────────────────────────────┤
│  3. COMPUTE BUFFERS        scratch space for the math            │
│     "the workbench"        0.8–6.5 GiB depending on -ub!          │
│     Knobs: -ub, -fa, pipeline parallelism (build-time)        │
└──────────────────────────────────────────────────────────────────┘
```

Newcomers obsess over #1 and forget #3 exists. That is a mistake, because **#3 is the most
elastic of the three**. On this machine, with the Qwopus model at 130k context:

| `-ub` (micro-batch) | Compute buffers | Prefill speed |
| --- | --- | --- |
| 256 | ~0.8 GiB | 2017 t/s |
| 512 | ~1.6 GiB | 2651 t/s |
| 1024 | ~3.2 GiB | 2615 t/s |
| 2048 | ~6.5 GiB | out of memory |

Notice two things. First, the workbench can grow larger than many entire models. Second,
**buying more workbench stops helping after 512** — but it never stops costing VRAM. That
is the shape of nearly every knob in llama.cpp: diminishing returns on speed, linear cost
in memory.

And one more, which is the reason flash attention is non-negotiable:

| Setting | Compute buffers at 130k |
| --- | --- |
| `-fa on` | 1.6 GiB |
| `-fa off` | **10.4 GiB** |

Turning flash attention off costs you nine gigabytes of VRAM for no benefit. Always `-fa on`.

---

## 1.4 How layers get spread across two GPUs

Your model has 40 transformer layers (plus one extra MTP layer — see chapter 9). With two
GPUs, llama.cpp must decide where each layer lives. That decision is `--split-mode`:

- **`-sm layer`** (default) — Give the first *N* layers to GPU 0 and the rest to GPU 1.
  A token flows through GPU 0's layers, crosses the PCIe bus once, then finishes on GPU 1.
  Each layer's KV cache lives on the same GPU as the layer. **This is what you want.**

- **`-sm row`** — Split each individual weight matrix across both GPUs by rows, so both
  GPUs work on the same layer simultaneously. Sounds better; usually isn't, because it
  requires a synchronisation after every layer.

- **`-sm none`** — Use one GPU only (`--main-gpu` picks it).

- **`-sm tensor`** — Newer tensor-parallel mode.

With `-sm layer`, **`--tensor-split` controls the boundary.** `-ts 12,29` does not mean
"12 GiB and 29 GiB" and it does not mean percentages. It means *ratio*: give GPU 0
`12/(12+29) = 29%` of the layers, i.e. 12 of the 41. Those numbers are just convenient
because they happen to equal the layer counts.

> **Teaching point — why the obvious split is wrong.** Your GPUs are 8 GiB and 16 GiB, so
> the natural instinct is `-ts 1,2`. But GPU 1 *also drives your monitor*, and Windows plus
> Slack plus a browser take ~1.6 GiB off the top. So GPU 1 does not have 16 GiB to give —
> it has about 14.6 GiB. Meanwhile GPU 0 has nearly all of its 8 GiB free. The correct
> split is therefore skewed *towards the small card* compared to what capacity alone
> suggests. On this machine the empirical answer is **`-ts 12,29`**, and `-ts 13,28` fails
> to load. There is no formula that would have told you that. You have to measure.

---

## 1.5 Prefill vs generation — two completely different workloads

Every performance number in this wiki comes in a pair, and they behave nothing alike.

**Prefill / prompt processing (PP)** — reading your prompt. All tokens are processed *in
parallel*, so the GPU does enormous matrix-times-matrix work. This is **compute-bound**.
Big batches help. Measured in the thousands of tokens/second.

**Generation / token generation (TG)** — writing the answer. Tokens must come out one at a
time, because token *n+1* depends on token *n*. Each step reads the whole model's active
weights to produce a single token, so this is **memory-bandwidth-bound**. Batch size is
irrelevant. Measured in the tens of tokens/second.

This distinction explains results that otherwise look absurd:

- Raising `-ub` from 256 to 512 improved prefill by 31% and generation by 0%.
  *Of course* — generation never uses a batch.
- Moving 2 of 40 layers' experts to the CPU cut prefill in half but generation by only 12%.
  *Of course* — with 8 of 256 experts active, generation touches ~3% of an expert layer's
  weights, but prefill with a 512-token batch touches nearly all of them.

> **Teaching point.** Decide which one you care about *before* you tune. If you paste
> 50,000-token files into a coding assistant, prefill dominates your wait and you should
> spend VRAM on `-ub`. If you chat in short turns, generation dominates and you should
> spend it on context or speculative decoding instead.

---

## 1.6 Why this MoE model is unusual (and why 130k context is possible)

The Qwen3.6-35B-A3B family is a **Mixture of Experts** with a **hybrid attention** design.
Both facts have large practical consequences.

**Mixture of Experts.** The model has 35.5 B parameters but activates only ~3 B per token:
256 experts per layer, 8 chosen per token. All 256 must be *in memory*, but only 8 are
*read* per token. This is why generation is so fast (117 t/s from a "35 B" model) and why
CPU offload behaves so strangely — see chapter 6.

**Hybrid attention.** Reading the model metadata:

```
n_layer          = 40        n_head        = 16
n_head_kv        = 2         n_embd_head_k = 256
ssm_d_conv       = 4         ssm_n_group   = 16
n_ctx_train      = 262144
```

Those `ssm_*` keys are the giveaway: some layers are not attention layers at all, they are
state-space / linear-attention layers whose memory does **not** grow with context length.

You can verify this arithmetically. A full-attention layer needs
`2 KV heads × 256 dims × 2 (K and V) × 2 bytes = 2 KiB` per token per layer. If all 40
layers were full attention, 131,072 tokens would need **10.2 GiB** of f16 KV cache. The
measured figure is **2.56 GiB**, which corresponds to roughly ten such layers. The other
thirty carry a fixed-size state instead.

> **This is the whole reason a 130k context is affordable here.** On a conventional
> 40-layer model, the KV cache alone would eat 10 GiB and the plan would be dead on
> arrival. Do not carry this optimism over to non-hybrid models.

---

## 1.7 Where to go next

- Want it running right now? → [Chapter 2: Serving a model over HTTP](02-serving-http.md)
- Want to know what a flag does? → [Chapter 3: The parameters that matter](03-parameters.md)
- Want to predict whether something fits? → [Chapter 4: Budgeting your VRAM](04-vram-budget.md)

