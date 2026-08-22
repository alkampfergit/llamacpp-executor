# Chapter 5 — Benchmarking Honestly

Tuning is only as good as your measurements, and llama.cpp benchmarks will lie to you in at
least four distinct ways. This chapter shows you each lie, how it was caught on this machine,
and the methodology that avoids it.

---

## 5.1 The three benchmark tools, and when each is wrong

### `llama-bench` — fast comparison, unrealistic memory

```powershell
.\llama-bench.exe -m $model -ngl 999 -sm layer -ts 10/18 -fa on -b 2048 -ub 512 `
  -p 4096 -n 128 -r 2 --progress
```

```
| model            | size      | ngl | fa | ts          | test   |          t/s |
| qwen35moe 35B.A3B|  20.21GiB | 999 |  1 | 10.00/18.00 | pp4096 | 2883.27 ±7.88 |
| qwen35moe 35B.A3B|  20.21GiB | 999 |  1 | 10.00/18.00 | tg128  |  116.68 ±0.39 |
```

Fast (27 s including load), repeatable, gives error bars, and sweeps parameters natively:
`-ub 128,256,512,1024` runs four configurations in one invocation.

**Its fatal flaw for our purpose: there is no `-c` flag.** The context is inferred from
`-p + -n + -d`, so a `-p 4096` run allocates a ~4k KV cache. Your server allocates 130k.
**You are measuring a different memory regime**, and for a model this tight, memory regime
*is* the problem.

Use it for: relative kernel speed, quick parameter shape-finding.
Never use it for: "will this configuration work on my server?"

### `llama-batched-bench` — realistic memory, the one to trust

```powershell
.\llama-batched-bench.exe -m $model -c 130048 -npp 8192 -ntg 128 -npl 1 `
  -ngl 999 -ts 12,29 -fa on -ub 256 -ctk q8_0 -ctv q8_0 -fit off
```

```
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|  8192 |    128 |    1 |   8320 |    5.588 |  1466.12 |    1.344 |    95.20 |
```

It accepts `-c`, so it allocates **exactly the KV cache your server will**. Every number in
chapter 6 comes from this tool.

`-npp` is prompt tokens, `-ntg` generated tokens, `-npl` parallel sequences. No `-r` flag, so
run it twice yourself when you need a variance estimate.

### `llama-fit-params` — prediction, not measurement

Covered in §4.4. Use it to decide *what is worth measuring*, so you spend your GPU time on
plausible configurations.

---

## 5.2 Lie #1: the contaminated sweep

The first real sweep on this machine produced this:

```
-ts  6/18   pp4096  2244 t/s     tg128  95.5
-ts  8/18   pp4096  2541 t/s     tg128  98.6
-ts 10/18   pp4096  1703 t/s     tg128  76.8      ← ???
-ts 12/18   FAILED TO LOAD
```

But `-ts 10/18`, measured minutes earlier **in its own process**, gave **2883 t/s and
116.7 t/s** — 69% and 52% higher.

Nothing about the configuration changed. What changed is that it was the third model load in
one process. VRAM fragmentation and residue from the earlier configurations left less
contiguous memory, and the run degraded.

> **Rule 1: one process per configuration.** Multi-value sweeps inside a single
> `llama-bench` invocation are fine when memory is abundant and actively misleading when it
> is tight. Since "tight" is exactly the regime we care about, always relaunch.

This is why the harness in [`scripts/bench-harness.ps1`](scripts/bench-harness.ps1) spawns a
fresh process for every single data point.

---

## 5.3 Lie #2: the silent overflow (the big one)

On Windows, since NVIDIA driver 536.40, **a CUDA program that exceeds VRAM does not get an
out-of-memory error.** The driver moves the overflow into system RAM and lets it run — at a
fraction of the speed. Windows calls this "Shared GPU Memory".

This is the answer to the question that started the whole investigation:

```
LM Studio:      ~300 t/s prefill,   CPU 47% busy,   both GPUs ~0–6% utilised
Raw llama.cpp: ~2900 t/s prefill
```

Same model, same hardware, 10× difference. LM Studio was not "worse at inference" — it had
overflowed, and the driver hid it.

**Turn the silent failure back into a loud one.** This is the single highest-value change you
can make before tuning anything:

> **NVIDIA Control Panel → Manage 3D Settings → CUDA - Sysmem Fallback Policy →
> "Prefer No Sysmem Fallback"**

Set it globally, or per-program for `llama-server.exe`, `llama-bench.exe` and
`llama-batched-bench.exe`. The policy is read when the CUDA context starts, so restart any
already-running process afterwards.

With this set, an over-large configuration **fails immediately with a clear error** instead of
quietly running at a fifth of the speed. You want that. A crash costs you ten seconds; a
silent 5× slowdown costs you a week of wrong conclusions.

On Linux this is not an issue — CUDA allocations past VRAM have always failed properly.

> **Rule 2: always record peak VRAM alongside throughput.** A throughput number without a
> memory number is uninterpretable, because you cannot tell a fast configuration from an
> overflowing one.

---

## 5.4 Lie #3: the helpful fallback you didn't ask for

Look at this pair of results, both `-ub 512`:

```
-c  98304    PP = 1630 t/s
-c 122880    PP = 2145 t/s      ← larger context, 32% FASTER?
```

That is impossible — unless something changed. The log for the second run:

```
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 789.13 MiB on device 0:
  cudaMalloc failed: out of memory
E graph_reserve: failed to allocate compute buffers
W sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
```

llama.cpp could not fit the pipeline-parallel buffers, silently retried **without pipeline
parallelism**, and that mode is *faster on this hardware* (see §3.4).

So the second run was not a better configuration — it was a **different** one. The benchmark
was honest; the configuration silently changed underneath it.

> **Rule 3: grep the log of every run, not just the results table.** Specifically for
> `retrying`, `out of memory`, `failed to`, and `ignoring`. The harness saves the full log of
> every run for exactly this reason.

This lie is also how the best finding in this whole campaign was discovered. When your
fallback path outruns your intended path, that is not noise — that is the answer.

---

## 5.5 Lie #4: the benchmark that doesn't measure your workload

`llama-bench` reports `pp4096` — prefill of a 4096-token prompt into an empty cache. Your
actual workload is a coding assistant re-sending a 57,000-token context where 56,000 tokens
are unchanged.

Measured on this machine, the gap between synthetic and real:

| Scenario | Prefill throughput |
| --- | --- |
| `llama-bench` synthetic | 2900–3200 t/s |
| `llama-server`, cold HTTP request | 1750–2100 t/s |
| `llama-server`, cached prefix | **effectively ~60,000 t/s** |

That last row is not a typo. A 57,559-token request took 32.5 s cold. The next request with
the same prefix evaluated **790 tokens in 0.96 s** — because prompt caching meant it only had
to process what changed.

> **Teaching point.** Once prompt caching is working, the prefill throughput you spent all
> that time optimising **stops being your bottleneck.** Optimise raw prefill for the
> *first* request of a session; optimise cache-hit rate for every request after it. The
> second is worth far more, and its knobs are `-np`, `-cram` and `--cache-reuse` — not `-ub`.

Always finish a tuning campaign by measuring the real thing over HTTP.

---

## 5.6 The methodology, assembled

1. **Set "Prefer No Sysmem Fallback"** so overflow is loud. (§5.3)
2. **Predict with `llama-fit-params`.** Reject configurations that need `-ot ...=CPU`. (§4.4)
3. **Measure with `llama-batched-bench` at the real `-c`.** (§5.1)
4. **One fresh process per data point.** (§5.2)
5. **Record peak VRAM with every throughput number.** (§5.3)
6. **Pass `-fit off`** so you measure what you asked for. (§3.5)
7. **Save every log; grep for `retrying`, `out of memory`, `ignoring`.** (§5.4)
8. **Append each result to disk immediately** — a driver crash must not cost you the campaign.
9. **Change one variable at a time.**
10. **Validate over HTTP at the end.** (§5.5)

Steps 4, 5, 7 and 8 are automated by [`scripts/bench-harness.ps1`](scripts/bench-harness.ps1):

```powershell
. "S:\OneDrive\Tools\llamacpp\wiki\scripts\bench-harness.ps1"

$common = @("-fa","on","-ngl","999","-fit","off","-ts","12,29","-ctk","q8_0","-ctv","q8_0")
foreach ($u in 256,512,1024) {
  Probe "ub$u" (@("-ub","$u") + $common) -Suite "ubatch" -Ctx 130048 -Npp "8192"
}
```

Each call: fresh process → sampled peak VRAM → parsed result → **appended to
`benchmarks/results.tsv` before the next run starts** → full log written to
`benchmarks/logs/<label>.txt`.

Output looks like:

```
ub256      OK   PP= 1466.12 TG=  95.20  VRAM  7660/15957=23617 MiB   19.2s
ub512      OOM  PP=         TG=         VRAM  7889/15534=23423 MiB   12.5s
```

The `OOM` row is as valuable as the `OK` rows — it is where the boundary is, and boundaries
are what you are actually mapping.

---

## 5.7 On variance

Repeated identical runs on this machine varied by about **±5%** (`-c 8448 -ub 256` measured
2017 and 2122 t/s on two occasions). The cause is the desktop competing for the display GPU.

So: **differences under ~8% are noise.** Do not conclude anything from a 3% improvement
without repeating it. Do confidently conclude from a 44% regression.

This is also why you must check whether a knob is even *real* before crediting it with a
gain. A 13% improvement was once attributed here to setting `GGML_SCHED_MAX_COPIES` as an
environment variable — which turned out to be a compile-time define that ignores the
environment entirely (§6.2). The measurement was variance; the attribution was wishful.

---

## 5.8 The quality gate: benchmarks cannot see a regression

Everything so far measures **speed**. Nothing so far would notice if a setting made the model
*worse*.

That matters most for KV cache quantisation. Dropping from `q8_0` to `q4_0` buys 24% more
prefill — and a throughput benchmark will report that as an unambiguous win even if the model
has started losing the middle of your document. You need a test that fails on **recall**, not
on speed.

### Needle in a haystack

The technique: build a very long, deliberately monotonous document, hide one unique
unguessable fact near the **start**, and ask for it at the **end**.

```
RECORD 0001: The commissioning passphrase for reactor bay 7 is CRIMSON-PELICAN-4417.
RECORD 0002..3000: routine telemetry; coolant nominal; no anomalies ...
=== END ARCHIVE ===
Question: what is the commissioning passphrase for reactor bay 7?
```

Every design choice here is deliberate:

| Choice | Reason |
| --- | --- |
| `CRIMSON-PELICAN-4417` | Arbitrary nonsense. Cannot be guessed, inferred, or recalled from training data — so a correct answer proves *retrieval*, not plausibility. |
| Placed in RECORD 0001 | The start is furthest from the question and the first thing a degraded cache loses. Hardest position. |
| 2999 near-identical filler records | Nothing distinguishes the needle by novelty of *topic*; the model must actually attend to it. |
| ~107,700 tokens total | Exercises most of a 130k window. A 4k test proves nothing about 130k. |
| `temperature 0`, `max_tokens 32` | Deterministic, and the answer is the first thing said. |

Run it with [`scripts/needle-test.ps1`](scripts/needle-test.ps1):

```powershell
.\wiki\scripts\needle-test.ps1 -Label q8-130k -Ctk q8_0 -Ctv q8_0 -Ub 512
.\wiki\scripts\needle-test.ps1 -Label q4-130k -Ctk q4_0 -Ctv q4_0 -Ub 1024
```

It builds the haystack once and reuses it, so every configuration sees a byte-identical
prompt. Results append to `benchmarks/needle-results.md`. As a bonus it gives you **real**
deep-context prefill and generation numbers over HTTP, which no synthetic benchmark does.

### Two warnings

**Use the chat endpoint.** On the raw `/completion` endpoint an instruction-tuned model emits
EOS immediately and returns an empty string. On this machine that produced an empty answer for
*both* KV types — which looks exactly like catastrophic quality failure and is actually a
request-format bug. **A "quality failure" that affects every configuration equally is a bug
in your harness, not a finding.**

**Know what the test does and does not prove.** Passing shows the setting preserves long-range
*exact recall*. It does not test multi-step reasoning across long context, summarisation
faithfulness, or code edits where one wrong token breaks a build. Treat a pass as "cleared for
retrieval-style work", not as "lossless".

---

Next: [Chapter 6 — Measured results](06-results.md).
