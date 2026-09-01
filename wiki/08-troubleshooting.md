# Chapter 8 — Troubleshooting

Organised by **symptom**, because that is what you actually have when something goes wrong.

---

## 8.1 "It works, but it's five times slower than it should be"

**By far the most common problem on Windows, and the hardest to spot, because nothing errors.**

### Diagnosis

Run the model and watch:

```powershell
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv -l 2
```

Open Task Manager → Performance → your GPU, and look at **Shared GPU Memory**.

> ⚠️ **`Shared GPU Memory > 0` on its own is NOT overflow.** llama.cpp always keeps several
> hundred MiB of host-visible buffers, and Windows counts those here: healthy runs on this box
> read **480–940 MiB**, and the *fastest* configuration measured had the *highest* figure.
> Overflow needs the **card also being full** — see the signature table below, and
> [chapter 19](19-vram-residency.md) for the per-process counter and the calibration against
> `llama-fit-params`' `Host` budget. `scripts/check-vram-residency.ps1` applies the rule for you.

| Signature | Cause |
| --- | --- |
| `memory.used` pinned at the card's maximum (**no headroom left**) **and** Shared GPU Memory far above llama.cpp's predicted `Host` budget, low GPU utilisation | **VRAM overflow into system RAM** → §8.2 |
| Shared GPU Memory non-zero but the card still has headroom | **Normal.** llama.cpp's own host buffers → not a fault |
| Prefill down ~25%, no error, everything "fits", and the **display GPU** has < ~500 MiB free | **WDDM demoting ~130 MiB off the display adapter** → close GPU-using desktop apps, or reduce `-c`. See [chapter 19](19-vram-residency.md) |
| Both GPUs at ~0–6% utilisation, CPU at ~50% | **Layers running on the CPU** → §8.3 |
| GPUs busy, but throughput still low | Wrong `-ub`, or a slow KV type → §8.4 |

The real case that started this wiki:

```
LM Studio:   ~300 t/s prefill,  CPU ~47%,  3070 ~0%,  5060 Ti 2–6%,  Shared GPU Memory high
llama.cpp:  ~2900 t/s prefill
```

Identical hardware, identical model. Nothing was broken — the allocation had overflowed and
the driver hid it.

---

## 8.2 Fix: stop the driver from hiding overflow

Since NVIDIA driver 536.40 (June 2023), a CUDA allocation that exceeds VRAM on Windows does
**not** fail. The driver silently places the overflow in system RAM. Result: a 5–10×
slowdown, no error message.

> **NVIDIA Control Panel → Manage 3D Settings → CUDA - Sysmem Fallback Policy →
> "Prefer No Sysmem Fallback"**

Notes:

- Set it globally, or per-program for `llama-server.exe`, `llama-bench.exe`,
  `llama-batched-bench.exe`.
- If you set it per-program, add the executable that **actually creates the CUDA context** —
  not a `.bat` or `.ps1` wrapper.
- The policy is read when the CUDA context is created, so **restart** any running server.

After this change, an over-large configuration fails immediately and clearly:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 831.13 MiB on device 0:
  cudaMalloc failed: out of memory
```

**That is a good outcome.** You want the loud failure. Ten seconds of crash beats a week of
measuring the wrong thing.

Linux users: this never applied to you; CUDA has always OOM'd properly there.

---

## 8.3 "Both GPUs are idle and the CPU is pegged"

Layers or tensors are on the CPU. Three causes:

**1. `-ngl` too low.** Use `-ngl 999`.

**2. `-ncmoe` / `-cmoe` / `-ot ...=CPU` in your command line.** Even a small value is very
expensive — 2 of 40 layers costs 44% of prefill (§6.5). Remove it and find the memory
elsewhere (§4.3).

**3. The auto-fitter decided for you.** If you did *not* pass `-ngl`, llama.cpp may have moved
tensors to CPU to make things fit. Check with:

```powershell
.\llama-fit-params.exe -m <model> -c <ctx> -fa on -ub 512 -ctk q8_0 -ctv q8_0 -fitt 300,1024
```

If the output contains `-ot "...=CPU"`, that is what happened. Either shrink the
configuration yourself or accept the cost knowingly.

If you must keep CPU tensors, note llama.cpp's own hint:

```
llama_model_loader: tensor overrides to CPU are used with mmap enabled
                    -- consider using --load-mode none for better performance
```

---

## 8.4 "Prefill is slow but memory looks fine"

Check, in order:

1. **`-fa on` present?** Without flash attention, compute buffers balloon to 10.4 GiB and you
   are probably overflowing (§3.3).
2. **KV cache types mixed, or outside the compiled set?** A stock CUDA build only has
   flash-attention kernels for `f16/f16`, `bf16/bf16`, `q8_0/q8_0` and `q4_0/q4_0`. Mixing
   types (`-ctk q8_0 -ctv q4_0`) or using an unsupported type (`q5_1`, `q5_0`, `q4_1`,
   `iq4_nl`) pushes attention **off the GPU entirely** — a 14× regression with no error
   message. Confirm by checking the *host* memory figure:
   ```powershell
   .\llama-fit-params.exe -m <model> -c 130048 -fa on -ctk q8_0 -ctv q4_0 -fitp on
   ```
   A `Host` row in the thousands of MiB instead of ~800 is the signature. This is the
   nastiest trap in the wiki (§6.6).
3. **`-ub` too small?** 128 gives 1368 t/s where 512 gives 2651 (§6.3).
4. **`-c` much larger than you need?** The window consumes KV memory and may push compute
   buffers against the VRAM cliff; the old "31% empty-context tax" explanation was withdrawn.
5. **Bad tensor split or scheduler-copy pressure?** Re-derive `-ts` before blaming pipeline
   mode. `GGML_SCHED_MAX_COPIES` is compile-time only, and deliberately raising `-ub` until an
   allocation fails is not a sound speed-tuning method (§6.2 and chapter 18).

---

## 8.5 "It won't load at all"

### `cudaMalloc failed: out of memory`

Honest OOM. Reduce in this order (cheapest first):

1. Close GPU-using desktop apps — check `nvidia-smi` first; you may be losing 1.6 GiB.
2. `-ctk q8_0 -ctv q8_0` if you were on f16 (or `q4_0/q4_0` if quality allows — §6.4).
3. Reduce `-c` a little. 130048 → 126976 costs 2.4% of the window and *gains* 26% prefill.
4. `-ub 512` → `256`.
5. Rebuild with `-DGGML_SCHED_MAX_COPIES=1` for a permanently smaller footprint.
6. Only then, `-ncmoe`.

### `CUDA error: the resource allocation failed` … `cublasCreate_v2`

```
E CUDA error: the resource allocation failed
E   current device: 0, in function cublas_handle at ggml-cuda/common.cuh:1501
```

**This is also out-of-memory** — cuBLAS could not get its workspace. Treat it exactly as
above. Seen on this machine at `-ub 384`, where `-ub 256` worked fine.

### `failed to allocate compute pp buffers`

Compute buffers didn't fit. Lower `-ub`, reduce `-c`, or rebuild with
`-DGGML_SCHED_MAX_COPIES=1`.

### `failed to fit params to free device memory: n_gpu_layers already set by user to 999, abort`

**Not an error.** You set `-ngl`, so the auto-fitter stood down. Look a few lines further for
`model loaded`. Nothing to fix.

### `sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism`

**A recoverable warning.** llama.cpp is retrying with a smaller scheduler. If a result row or
`listening on` follows, record the run as successful fallback; if the retry also fails, it is a
real OOM. Controlled `ub512` tests put fallback and a one-copy build within 1.6%, so do not call
the warning a speedup. See chapter 18. The environment variable still does **not** work.

### `model has unused tensor blk.40.… -- ignoring`

**Expected for MTP models.** `blk.40` is the multi-token-prediction head, unused unless you
pass `--spec-type draft-mtp`. It costs no VRAM while ignored. See [chapter 9](09-speculative-decoding.md).

---

## 8.6 "Every request reprocesses my whole prompt"

You are losing prompt caching, which is worth more than every other optimisation combined
(§5.5).

1. **Check `-np`.** With `-np 4 -c 130048`, each slot only holds 32512 tokens. A 50k prompt
   cannot be cached because it does not fit in a slot. Use `-np 1` for a single user.
2. **Check the log** for `selected slot by LCP similarity, f_sim_best = …`.
   - `f_sim_best > 0.95` → working well.
   - Low, or absent → the prefix genuinely changed, or the cache was evicted.
3. **Raise `-cram`.** Default is 8192 MiB of host RAM for cached states. With 64 GiB of RAM,
   `-cram 24576` is cheap.
4. **Check your client.** Some clients reorder or re-summarise history between turns, which
   destroys the common prefix. Nothing server-side can fix that.

---

## 8.7 "The output is gibberish / repeats / never stops"

**At long context specifically:** try `-ctk bf16 -ctv bf16` (the model authors' suggestion).
If f16 or bf16 fixes it, the KV quantisation was the cause. Note that this costs VRAM you may
not have — see §4.3.

**At all context lengths:** you are probably fighting the chat template or the sampler.

- Confirm the template: `curl.exe http://localhost:9010/props`
- Use the authors' sampling settings (§2.5). For precise coding:
  `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0`
- `--temp 0` on a reasoning model often produces loops. Don't.

**Thinking tags leaking into your application:** this is a reasoning model. Either disable
thinking with `--chat-template-kwargs "{\"enable_thinking\":false}"` or let llama.cpp strip
the tags with `--reasoning-format`.

**Never stops generating:** set `max_tokens` in the request, and check
`--reasoning-budget` — an unbudgeted thinking phase can run a long way.

---

## 8.8 "It was fast yesterday"

Almost always **VRAM pressure from something else.**

```powershell
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
```

Profile A has roughly 24 MiB of spare VRAM. One browser tab, one video call, one screen
recorder can push it over — and if you have not set "Prefer No Sysmem Fallback", it will
*not* crash. It will just get slow.

This is the strongest practical argument for setting that policy: it converts "mysteriously
slow since Tuesday" into "failed to start, with a message".

---

## 8.9 Diagnostic one-liners

```powershell
# What GPUs does llama.cpp see?
.\llama-bench.exe --version

# What is on my GPUs right now?
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv

# Will this configuration fit? (under one second, no model load)
.\llama-fit-params.exe -m <model> -c 130048 -fa on -ub 512 -ctk q8_0 -ctv q8_0 -fitp on

# What would llama.cpp choose by itself?
.\llama-fit-params.exe -m <model> -c 130048 -fa on -ub 512 -fitt 300,1024

# Model architecture, layer count, training context
.\llama-fit-params.exe -m <model> -v 2>&1 | Select-String "n_layer|n_head|n_ctx_train|n_expert"

# Live watch while serving
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 2

# Is the server healthy?
curl.exe http://localhost:9010/health

# What is each slot holding?
curl.exe http://localhost:9010/slots

# How many tokens is my prompt really?
curl.exe http://localhost:9010/tokenize -H "Content-Type: application/json" -d '{\"content\":\"...\"}'
```

---

Next: [Chapter 9 — Speculative decoding & MTP](09-speculative-decoding.md).

