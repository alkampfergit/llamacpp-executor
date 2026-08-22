# NVIDIA "Prefer No Sysmem Fallback" — the exact executables to add

The per-program setting in NVIDIA Control Panel is keyed on the **full executable path**.
Adding one binary does not cover its siblings, and it does not cover a rebuild at a different
path. This page lists every path that matters.

---

## Why this page exists — a measured failure

The policy was set for at least one executable at **10:19 UTC**. Every Qwen3.8-27B benchmark
ran **12:22–13:14 UTC**, two hours later. They still spilled silently, because the
measurements came from `llama-batched-bench.exe`, which was not in the profile list.

Re-running the same configuration afterwards proved it:

| Run | GPU1 free | Prefill | Allocation failure logged? |
| --- | --- | --- | --- |
| `-ts 21,44` (healthy) | 527 MiB | **1219 t/s** | — |
| `-ts 20,45` | 324 MiB | 943 t/s | **none** |
| `-ts 20,45` re-run, busier desktop | 337 MiB | **790 t/s** | **none** |

Prefill degrades as free VRAM shrinks and **the driver never refuses anything**. That is
textbook shared-memory spill, and it is exactly what this setting is supposed to convert into a
clean, visible out-of-memory error.

> **Until every binary below is covered, any slow result is ambiguous:** you cannot tell a badly
> configured run from a silently spilling one.

---

## 1. Critical — these allocate model weights and KV cache

Add all six. `llama-batched-bench.exe` is the one that produced all 73 recorded throughput
measurements, and it was the one missing.

```
S:\OneDrive\Tools\llamacpp\llama-server.exe
S:\OneDrive\Tools\llamacpp\llama-batched-bench.exe
S:\OneDrive\Tools\llamacpp\llama-bench.exe
S:\OneDrive\Tools\llamacpp\llama-cli.exe
S:\OneDrive\Tools\llamacpp\llama-perplexity.exe
S:\OneDrive\Tools\llamacpp\llama-completion.exe
```

| Executable | Why it matters |
| --- | --- |
| `llama-server.exe` | The production server. Every HTTP/deep/needle/MTP number came from here. |
| `llama-batched-bench.exe` | **The main benchmark tool** — accepts `-c`, so it allocates the real KV cache. All of `results.tsv`. |
| `llama-bench.exe` | Secondary benchmark tool. |
| `llama-cli.exe` | Interactive/one-shot runs. |
| `llama-perplexity.exe` | Quality gating (`--kl-divergence-base`). Long runs at full context. |
| `llama-completion.exe` | Non-interactive completion. |

## 2. Worth adding — allocate VRAM in some modes

```
S:\OneDrive\Tools\llamacpp\llama-imatrix.exe
S:\OneDrive\Tools\llamacpp\llama-mtmd-cli.exe
S:\OneDrive\Tools\llamacpp\llama-tts.exe
S:\OneDrive\Tools\llamacpp\llama.exe
S:\OneDrive\Tools\llamacpp\ggml-rpc-server.exe
```

## 3. Not needed — no meaningful GPU allocation

`llama-fit-params.exe` (reads GGUF headers only — it *predicts* memory, never allocates it),
`llama-tokenize.exe`, `llama-quantize.exe`, `llama-gguf-split.exe`, `llama-results.exe`,
`llama-template-analysis.exe`, and the `*-cli.exe` vision wrappers
(`gemma3`, `llava`, `minicpmv`, `qwen2vl`), `llama-mtmd-debug.exe`.

Adding them anyway is harmless.

---

## 4. ⚠️ Future builds are DIFFERENT paths and need adding again

A rebuild does not inherit the setting. Per `CLAUDE.md` the baseline binaries are never
overwritten, so a new build lands somewhere new — and starts out unprotected:

```
S:\OneDrive\Tools\llamacpp\llama.cpp\build\bin\llama-server.exe
S:\OneDrive\Tools\llamacpp\llama.cpp\build\bin\llama-batched-bench.exe
S:\OneDrive\Tools\llamacpp\llama.cpp\build-control\bin\...
S:\OneDrive\Tools\llamacpp\llama.cpp\build-fa\bin\...
```

**Add the new paths before benchmarking a fresh build**, or the first comparison against the
baseline is measuring the policy difference rather than the flags.

---

## 5. How to set it

1. Right-click the desktop → **NVIDIA Control Panel**
2. **Manage 3D settings** → **Program Settings** tab
3. **Add** → **Browse…** → pick the executable by full path (the dropdown only lists programs
   the driver has already seen, so use *Browse* for these)
4. Find **CUDA - Sysmem Fallback Policy** → set to **Prefer No Sysmem Fallback**
5. **Apply**
6. Repeat per executable — there is no wildcard or folder-level rule

Alternatively set it once under the **Global Settings** tab, which covers everything including
future builds. That is simpler and, on a machine used mainly for local inference, usually the
right choice. The cost is that unrelated CUDA applications also lose the fallback and will now
fail instead of running slowly — which for anything you actually care about is an improvement.

> The policy is read **when the CUDA context is created**, so restart any running `llama-server`
> after changing it.

---

## 6. Verifying it took effect

Deliberately ask for slightly too much and check that it now **fails** instead of getting slow:

```powershell
cd S:\OneDrive\Tools\llamacpp
. .\.claude\skills\tuning-llamacpp-configs\scripts\bench-harness.ps1 -OutDir wiki\benchmarks
Probe "policycheck" @('-ub','512','-ts','20,45','-ctk','q8_0','-ctv','q8_0',
                      '-fa','on','-ngl','999','-fit','off') `
  -Ctx 130048 -Npp '8192' `
  -Model "S:\HuggingFace\lmstudio\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf"
```

| Outcome | Meaning |
| --- | --- |
| `OOM`, and the log contains `cudaMalloc failed: out of memory` | ✅ Policy is active for this executable |
| `OK` at ~790–950 t/s with no allocation error in the log | ❌ Still spilling — this path is not covered |

`-ts 20,45` is chosen deliberately: it is a *known* over-subscription of GPU1 that a healthy
`-ts 21,44` avoids, so it is the smallest reliable trigger.

---

## 7. What this unblocks

Two recorded caveats collapse once the policy covers the benchmark binaries:

- The "degraded rows" become provable rather than merely *consistent with* spill.
- Display-GPU drift (idle usage moved 703 → 1288 → 1844 MiB during one session) stops silently
  invalidating tight profiles — a config that no longer fits will say so.

Both are the same win: **the edge and the cliff stop looking alike.**
