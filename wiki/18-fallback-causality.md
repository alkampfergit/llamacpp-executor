# Chapter 18 - The retry worked; the split made it fast

The observation was real: changing Qwopus from `-ts 12,29` to `-ts 13,28` produced an alarming
allocation error, llama.cpp retried without pipeline parallelism, and performance improved.
The old wiki treated those three events as one causal chain. They are two effects:

1. `13,28` moves one layer from device 1 to device 0 and materially improves this workload.
2. That distribution leaves too little room on device 0 for the four-copy reservation, so the
   stock build retries with one copy. The retry is successful and saves memory; it is not the
   source of the large throughput gain.

## 18.1 Why the old comparison could not answer the question

The quoted **2650 versus 1850 t/s** compared different KV types, ubatches, memory states and
execution modes. Chapter 17's later **2371 t/s** result proved that `13,28` plus fallback repeats,
but its `-ngl 41` control also left a layer on CPU. Neither experiment held the split and all
runtime arguments fixed while changing only scheduler behavior.

## 18.2 Three controlled experiments

All new campaigns use Qwopus, symmetric `q8_0/q8_0` KV, `npl 1`, interleaved arms and matched
rebuilds from the same source. `build-control` has `GGML_SCHED_MAX_COPIES=4`; `build-fa` has one
copy. Its additional `FA_ALL_QUANTS` flag is inert for symmetric KV.

| question | controlled comparison | result |
| --- | --- | ---: |
| Does one scheduler copy improve `ub512`? | same `c8448 / ub512 / ts13,28`, max 4 vs max 1 | **+1.8% PP, +0.1% TG** |
| Is runtime fallback faster than the one-copy build? | same `c64000 / ub512 / ts13,28`, fallback vs max 1 | **-1.6% PP, +0.1% TG** for fallback |
| Does the split matter when scheduler mode is fixed? | same one-copy build, `12,29` vs `13,28` | **+12.8% PP, +7.4% TG** for `13,28` |

The first two are well below the repo's 8% practical noise threshold. The split result is large,
repeated and survives with scheduler copy count held fixed. A three-repetition `ub256` pilot did
show +8.8% for one copy, exactly at the noise boundary; scheduler effects may be batch-dependent,
but that does not explain the `ub512` observation.

Evidence:

- `wiki/benchmarks/scheduler-copies-ab-ub512/`
- `wiki/benchmarks/runtime-fallback-vs-max1/`
- `wiki/benchmarks/tensor-split-ab/`
- `wiki/benchmarks/scheduler-copies-ab-ub256/` (pilot)

## 18.3 The source-level nuance

The runtime fallback and `GGML_SCHED_MAX_COPIES=1` are close, but not literally the same state.
`ggml_backend_sched_new` turns both into one scheduler copy. The runtime fallback additionally
sets `cparams.pipeline_parallel = false`; the compile-time one-copy build leaves it true, so
`llama_context::graph_compute` still performs the pipeline-mode synchronization before reusing a
graph. That distinction was missing from chapters 14 and 17.

The same-source `c64000` experiment above measures the practical result: despite the source
difference, the two states are only 1.6% apart in prefill and identical in generation here.

## 18.4 What the warning means operationally

This sequence is a recovered run, not an OOM result:

```text
cudaMalloc failed: out of memory
sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
... benchmark or server continues successfully ...
```

Record it as `OK` with execution mode `fallback-single-copy`. If the retry also fails and no
throughput/result row appears, record `OOM` with mode `fallback-failed`. The benchmark harness
now writes both `execution_mode` and `notes` in new result files, while preserving the legacy TSV
schema when appending to old evidence.

For serving, `-ts 13,28` remains the best evidenced split at 64k on this exact GPU pair. The
warning is expected for the stock binary under the measured memory state. It should not be used
as a tuning target: a browser, display load or driver change can move the allocation boundary.
The robust lesson is to benchmark the split and keep enough headroom, not to manufacture an OOM.

## 18.5 Corrected conclusion

> The user's performance observation was correct. The causal label was not. On Qwopus, changing
> `12,29` to `13,28` is the repeatable large win (+12.8% prefill with scheduler mode fixed).
> Retrying without pipeline parallelism is a useful recovery path and memory saving, but at
> `ub512` it changes throughput by only about 2% in controlled comparisons.

Previous: [Chapter 17 - Pipeline parallelism and `-ngl`](17-pipeline-parallelism-ngl.md) ·
Back to [README](README.md).

