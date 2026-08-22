# Chapter 7 — Recommended Configurations

Copy-paste commands for this machine. Each states **what it optimises**, **what it gives
up**, and whether the numbers are **measured** or a **starting point**.

Ready-to-run scripts: [`scripts/`](scripts/).

---

## Choosing between them

The real decision is **KV cache precision vs prefill speed**, and the honest answer depends
on what you do:

```
Do you need the full 130k window?
├── NO (prompts under ~120k)
│     → Profile C.  q8_0 quality AND 2323 t/s. The best all-round config.
└── YES
      ├── Quality matters more than latency (code edits, careful reasoning)
      │     → Profile A.  q8_0 KV, 1850 t/s.
      └── Latency matters more (search, retrieval, bulk reading)
            → Profile B.  q4_0 KV, 2650 t/s -- 43% faster, and it PASSED a
                          108k needle-retrieval test (§6.4).
```

**Before any of them, do the two free things:**

1. Set **NVIDIA Control Panel → Manage 3D Settings → CUDA - Sysmem Fallback Policy →
   "Prefer No Sysmem Fallback"**. Without it, an over-large config silently runs at a fifth
   speed. Two configurations in chapter 6 were caught doing exactly this (§6.5).
2. Close GPU-using desktop apps. Every profile here sits within a few hundred MiB of the
   ceiling.

> ### ⚠️ Do not set `GGML_SCHED_MAX_COPIES` as an environment variable
> Earlier versions of this wiki told you to. **It has no effect** — it is a compile-time
> `#define`, not a runtime variable (§6.2). The commands below no longer set it. The speed
> these profiles get from non-pipelined execution comes from llama.cpp's automatic fallback
> under memory pressure, which is reproducible (±3% over three runs) but accidental. To make
> it deliberate, see [§7.5](#75-the-rebuild-that-would-fix-all-of-this).

---

## Profile A — full 130k, `q8_0` KV (quality-first)

**Measured:** 1850 t/s prefill synthetic; **1414 t/s and 47.5 t/s generation on a real
107,743-token request**. Needle test: **PASS**.
**Gives up:** 30% of prefill versus Profile B.

```powershell
cd S:\OneDrive\Tools\llamacpp

.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf" `
  --alias qwopus-coder `
  --host 0.0.0.0 --port 9010 `
  -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 12,29 -fa on `
  -b 2048 -ub 512 `
  -ctk q8_0 -ctv q8_0 `
  -fit off `
  -cram 24576 `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

| Flag | Why |
| --- | --- |
| `-ctk/-ctv q8_0` | Symmetric pair on the compiled FA path; the quality choice (§6.3) |
| `-ub 512` | 1850 t/s. `ub 1024` does **not** fit with q8_0 at 130k |
| `-ts 12,29` | The only ratio that loads. 11,30 and 13,28 both fail (§6.8) |
| `-np 1` | Otherwise each request gets `130048/N` tokens |
| `-fit off` | Don't let the fitter alter a measured config |
| `-cram 24576` | Prompt cache — worth more than everything above (§5.5) |

---

## Profile B — full 130k, `q4_0` KV (speed-first)

**Measured:** 2650 t/s prefill synthetic (2757 / 2599 / 2594 over three runs); **1758 t/s and
46.6 t/s on the same real 107,743-token request** — 15 seconds faster end-to-end than
Profile A. Needle test: **PASS**.
**Gives up:** KV precision. Validated for *retrieval*; not validated for long-context
reasoning or code-edit fidelity (§6.4).

```powershell
cd S:\OneDrive\Tools\llamacpp

.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf" `
  --alias qwopus-coder `
  --host 0.0.0.0 --port 9010 `
  -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 12,29 -fa on `
  -b 2048 -ub 1024 `
  -ctk q4_0 -ctv q4_0 `
  -fit off `
  -cram 24576 `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

The `q4_0` cache is 700 MiB smaller than `q8_0`, and that is exactly what makes `-ub 1024`
reachable — which in turn triggers the fast non-pipelined path. The two changes compound.

> **Expect an alarming log line.** You will see `failed to allocate compute buffers` followed
> by `retrying without pipeline parallelism`. **That is this profile working as intended**
> (§6.2), not an error. Confirm `model loaded` and `listening on` follow it.

---

## Profile C — 127k, `q8_0` KV (best all-round)

**Measured:** 2323 t/s prefill, 109 t/s generation — the **fastest `q8_0` configuration
found**, and faster than Profile B's `q4_0` at `ub 512`.
**Gives up:** 3072 tokens of context (2.4%).

```powershell
cd S:\OneDrive\Tools\llamacpp

.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf" `
  --alias qwopus-coder `
  --host 0.0.0.0 --port 9010 `
  -c 126976 -np 1 `
  -ngl 999 -sm layer -ts 12,29 -fa on `
  -b 2048 -ub 512 `
  -ctk q8_0 -ctv q8_0 `
  -fit off `
  -cram 24576 `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

**If you can live with 127k instead of 130k, this is the profile to use.** You keep the
higher-precision KV cache *and* get 26% more prefill than Profile A. Giving up 2.4% of the
window for 26% of the speed is the best trade in this entire wiki.

---

## Profile D — Qwen3.6 Q3_K_XL, 130k with headroom

**For:** long context without living at the ceiling. At 15.69 GiB the model leaves ~2.5 GiB
spare, so `f16` KV and `-ub 512` both fit.
**Status:** the 130k load is **proven**; `llama-bench` measured 3107–3206 t/s prefill and
~94 t/s generation. `-ts` below is a **starting point** — verify.
**Gives up:** Q3 quality, and ~20% generation speed versus Q4_K_M (117 → 94 t/s).

```powershell
cd S:\OneDrive\Tools\llamacpp

.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf" `
  --alias qwen3.6-35b `
  --host 0.0.0.0 --port 9010 `
  -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 1,2 -fa on `
  -b 2048 -ub 512 `
  -fit off `
  -cram 24576 `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

No `-ctk`/`-ctv`: **f16 KV fits here, and f16 is the fastest and most accurate KV there is.**
Only quantise when you must. Your proven `-ts 1,3` also works; `1,2` relieves the display
GPU, which is the binding constraint.

---

## What *not* to do

| Anti-pattern | Why | Ref |
| --- | --- | --- |
| `$env:GGML_SCHED_MAX_COPIES = "1"` | **No effect.** Compile-time define | §6.2 |
| Mixing KV types (`-ctk q8_0 -ctv q4_0`) | Asymmetric → attention leaves the GPU, −89% | §6.3 |
| `-ctk q5_1` / `q5_0` / `q4_1` / `iq4_nl` | Not a supported FA type at all, −93% | §6.3 |
| `-ncmoe 2` to make something fit | −69% prefill for 5% of the model | §6.7 |
| `-fa off`, or omitting `-fa` | 9 GiB of extra compute buffers | §3.3 |
| `-ub 1024` with `q8_0` at 115k–123k | Saturates the display GPU → spills to system RAM, 1036 t/s | §6.5 |
| `-np 4` with `-c 130048` for one user | Each request capped at 32512 tokens | §2.4 |
| `-mg 1` with `-sm layer` | No effect | §3.1 |
| Assuming a smaller quant is faster | Q3_K_XL generates *slower* than Q4_K_M | §6.1 |
| Tuning before the sysmem policy is set | You will measure overflow, not speed | §5.3 |

---

## 7.5 The rebuild that would fix all of this

Both remaining awkwardnesses — relying on an allocation failure for speed, and being unable
to use asymmetric KV — are build-time limitations:

```
cmake -B build -DGGML_CUDA=ON -DLLAMA_SCHED_MAX_COPIES=1 -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build --config Release -j
```

- `LLAMA_SCHED_MAX_COPIES=1` makes the fast non-pipelined path **deterministic** rather than
  a side effect of running out of memory. It should also free enough to run `q8_0` KV at
  `-ub 1024` at full 130k — Profile A's quality with Profile B's speed.
- `GGML_CUDA_FA_ALL_QUANTS=ON` enables `-ctk q8_0 -ctv q4_0`: precise keys, compact values.

Cost: a long compile and a much larger CUDA binary. **Untested — this is the top open item
in §6.9.**

---

## Verifying a profile after launch

```powershell
curl.exe http://localhost:9010/health          # alive?
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv   # inside physical VRAM?
```

Check the server log for `n_ctx_slot` — it should be ~130048, not 32512 (that would mean
`-np` divided it). Then send a real request and read the timings llama.cpp prints:

```
prompt eval time = ... tokens per second      <- real prefill
       eval time = ... tokens per second      <- real generation
```

**Those two lines are the only numbers that matter.** Everything in chapter 6 exists to make
them good.

On a second, similar request, look for:

```
selected slot by LCP similarity, f_sim_best = 0.987, f_keep = 0.995
```

Above 0.95 means prompt caching is working and your effective prefill wait has collapsed by
an order of magnitude. If the line never appears, check `-np` first.

Finally, to confirm long-context quality on your own settings:

```powershell
.\wiki\scripts\needle-test.ps1 -Label mine -Ctk q8_0 -Ctv q8_0 -Ub 512
```

---

Next: [Chapter 8 — Troubleshooting](08-troubleshooting.md).
