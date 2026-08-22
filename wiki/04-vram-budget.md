# Chapter 4 — Budgeting Your VRAM

The difference between someone who fights llama.cpp and someone who directs it is this: the
second person can predict, before launching, whether a configuration will fit. This chapter
teaches you to do that arithmetic — and then shows you the tool that does it for you in
under a second.

---

## 4.1 Start with what you actually have, not what you bought

You own 24 GiB of VRAM. You do not have 24 GiB to spend.

```powershell
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
```

On this machine, at rest:

```
0, NVIDIA GeForce RTX 3070,     8192 MiB,   407 MiB
1, NVIDIA GeForce RTX 5060 Ti, 16311 MiB,  1658 MiB
```

So the real budget is:

```
GPU 0:   8192 −  407 (Camtasia)                    =  7785 MiB
GPU 1:  16311 − 1658 (Windows desktop + apps)      = 14653 MiB
                                                    ─────────────
                                          usable   = 22438 MiB
```

**You lost 2.0 GiB before you started**, and almost all of it on the *display* GPU. On this
box the culprits are Slack, Signal, WhatsApp, Camtasia, Snagit, the NVIDIA overlay, PowerToys
and a browser — each taking 50–300 MiB of the 5060 Ti.

> **Teaching point.** With a model this tight, closing chat apps is a *performance tuning
> action*. Freeing 800 MiB on the display GPU is what lets you move from `-ub 256` to
> `-ub 512`, which is worth roughly +30% prefill. That is a bigger win than most flags.

And you must leave a margin. The desktop's usage is not constant — opening a browser tab or
letting a video play will claim more VRAM *while your model is loaded*. Leave ~300 MiB on the
compute-only card and ~1 GiB on the display card.

> **On the display GPU choice:** your monitor is on the *faster, larger* card. Moving the
> cable to the 3070 would free ~1.6 GiB on the 5060 Ti at the cost of ~1.6 GiB on the 3070 —
> roughly neutral in total, but it shifts headroom onto the card that does most of the work,
> letting it hold more layers. Worth trying if you are squeezed.

---

## 4.2 The three consumers, quantified

From §1.3, VRAM goes to model weights, KV cache, and compute buffers. Here is how to
estimate each.

### Model weights

Take the file size and add ~2%. A 20.21 GiB GGUF needs about 20.2 GiB — llama.cpp reports
20174 MiB for this one. Simple, and the one term you cannot tune without changing quant.

### KV cache

The textbook formula is:

```
bytes/token = n_layer × n_head_kv × head_dim × 2 (K and V) × bytes_per_element
```

For this model that would be `40 × 2 × 256 × 2 × 2 = 81920` bytes/token at f16 — meaning
131,072 tokens would need **10.2 GiB**.

The measured value is **2622 MiB**. The formula is wrong by 4×.

Why? Because of the hybrid architecture from §1.6 — only about a quarter of the 40 layers
are full-attention layers; the rest keep a fixed-size state. So for *this family*:

| Context | `f16` | `q8_0` |
| --- | --- | --- |
| 8,192 | 222 MiB | ~120 MiB |
| 32,768 | 702 MiB | ~380 MiB |
| 65,536 | 1342 MiB | ~730 MiB |
| 131,072 | 2622 MiB | 1422 MiB |

Roughly **20 MiB per 1000 tokens at f16, 11 MiB at q8_0**.

> **Lesson: don't trust a formula you found online — measure your model.** §4.4 shows how to
> get these numbers for any GGUF in one second.

### Compute buffers

The term everyone forgets, and the most elastic. Driven by `-ub`, `-fa`, and
`GGML_SCHED_MAX_COPIES`. At 130k context on this model:

| Configuration | Compute buffers |
| --- | --- |
| `-fa on -ub 256` | 818 MiB |
| `-fa on -ub 512` | 1637 MiB |
| `-fa on -ub 1024` | ~3200 MiB |
| `-fa on -ub 2048` | 6548 MiB |
| **`-fa off -ub 512`** | **10376 MiB** |

---

## 4.3 Doing the sum

For the Qwopus Q4_K_M at 130k context, `-fa on`, `-ub 512`, `q8_0` KV:

```
  model weights                    20174 MiB
  KV cache (q8_0, 130k)             1422 MiB
  compute buffers (ub 512)          1637 MiB
                                   ─────────
  required                          23233 MiB
  available (§4.1)                  22438 MiB
                                   ─────────
  SHORT BY                            795 MiB      ✗ will not fit
```

Now try `-ub 256`:

```
  20174 + 1422 + 818            =  22414 MiB
  available                        22438 MiB
                                   ─────────
  spare                                24 MiB      ✓ fits (barely)
```

And that is exactly what happens in practice — measured peak usage 23606 MiB across both
cards, and it runs. (The measured figure is higher than the arithmetic because
`nvidia-smi` includes the desktop's own usage in the same number.)

> **This is the central result for this model.** Q4_K_M at 130k fits with **zero** CPU
> offload, but only at `-ub 256`, and only with `q8_0` KV. Every gigabyte matters, and the
> arithmetic above tells you *which* gigabyte to go after.

### The four ways to close a gap, best first

| Move | Saves at 130k | Speed cost |
| --- | --- | --- |
| **Close GPU-using desktop apps** | up to ~1.5 GiB | **none** |
| `-ctk q8_0 -ctv q8_0` instead of f16 | 1200 MiB | ~2% |
| `-ctk q4_0 -ctv q4_0` instead of q8_0 | ~700 MiB | **negative — it's faster** (but see quality, §6.4) |
| Rebuild `-DLLAMA_SCHED_MAX_COPIES=1` | several hundred MiB | **negative — it's faster** |
| Reduce `-c` (e.g. 130048 → 126976) | 11 MiB per 1000 tokens | *gains* prefill speed (+26%) |
| `-ub 512` → `256` | 819 MiB | −21% prefill |
| `-ncmoe 2` | ~1000 MiB | **−69% prefill** ✗ |
| Mixed or unsupported KV types | ~350 MiB | **−89 to −93% prefill** ✗✗ |

> `GGML_SCHED_MAX_COPIES` is **not** a runtime environment variable — it is a compile-time
> define. See §6.2.

Work down that list in order. The first three are free or better than free. The last two are
traps that look attractive on paper.

---

## 4.4 The one-second answer: `llama-fit-params`

You do not have to do that arithmetic by hand, and you do not have to wait for a model load
to find out you were wrong. `llama-fit-params.exe` reads only the GGUF *header* and reports
the memory plan — in well under a second.

### Mode 1: print the memory estimate

```powershell
.\llama-fit-params.exe `
  -m "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf" `
  -c 131072 -fa on -ub 512 -fitp on
```

```
        model  context  compute
CUDA0    6899      791      596
CUDA1   13003     1831     1041
Host      272        0      520
```

Three columns — **exactly the three consumers from §1.3**, per device, in MiB. Add a row up
to get that GPU's total requirement. This single command replaces most of §4.3.

Sweeping a parameter is now trivial:

```powershell
$m = "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf"
foreach ($c in 8192, 32768, 65536, 131072, 262144) {
  Write-Output "=== c=$c ==="
  .\llama-fit-params.exe -m $m -c $c -fa on -ub 512 -fitp on 2>$null
}
```

That is the whole KV-cache table from §4.2, measured on your actual model, in about five
seconds. **This is how you should explore, not by launching servers.**

### Mode 2: emit a ready-to-run command line

Without `-fitp`, the tool prints the flags it would use:

```powershell
.\llama-fit-params.exe -m $m -c 131072 -fa on -ub 512 -ctk q8_0 -ctv q8_0 -fitt 300,1024
```

```
-c 131072 -ngl 41 -ts 12,29 -ot "blk\.38\.ffn_(gate|up|gate_up|down).*=CPU,..."
```

Read that as a diagnosis: *"at these settings I must push about two layers' experts to the
CPU."* If you see `-ot ... =CPU` in the output, your configuration does not fit — and now you
know by roughly how much, before waiting for a load.

`-fitt A,B` is the spare VRAM to leave per device. Use your §4.1 numbers.

### Its two limitations

1. **It under-predicts compute buffers.** It said 596 MiB for CUDA0; reality was 831 MiB. Add
   ~250 MiB per device to whatever it tells you, or set `-fitt` generously.
2. **It cannot know what your desktop will do next.** It measures free VRAM at the instant it
   runs.

So: plan with `llama-fit-params`, then verify with a real load. Which brings us to chapter 5.

---

## 4.5 Verifying a real load

Two things to check after launching.

**Did anything land on the CPU that shouldn't have?** Look for `CPU` buffer entries and for:

```
llama_model_loader: tensor overrides to CPU are used with mmap enabled
```

**Are you actually inside physical VRAM?** Watch it while the model runs:

```powershell
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 2
```

Two failure signatures to recognise:

- **`memory.used` pinned at the card's maximum** and throughput far below expectation →
  you have overflowed into shared system memory. See chapter 8.
- **Both GPUs near 0% utilisation while the CPU is busy** → layers are running on the CPU.
  This is the signature of the LM Studio problem that started this whole investigation:
  ~300 t/s prefill, CPU at 47%, both GPUs idle.

> **Teaching point.** "It loaded" is not "it fits". On Windows, a configuration that
> overflows still loads and still answers — just slowly. You must look at the numbers.

---

Next: [Chapter 5 — Benchmarking honestly](05-benchmarking.md).
