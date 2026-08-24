# `scheduler-copies-ab-ub512/` - scheduler copies at the relevant ubatch

Qwopus, matched rebuilds, identical `-c 8448 -ub 512 -ts 13,28 -ctk q8_0 -ctv q8_0` arguments.
The arms were interleaved for five repetitions. The two build commits differ only in
`PROVENANCE.md`; their compiled source is the same. Symmetric KV makes
`GGML_CUDA_FA_ALL_QUANTS` inert, leaving scheduler copy count as the operative build difference.

| arm | prefill, five reps | mean | generation mean |
| --- | --- | ---: | ---: |
| `GGML_SCHED_MAX_COPIES=4` | 2686.62 / 2638.30 / 2710.66 / 2694.37 / 2738.06 | **2693.6** | 103.85 |
| `GGML_SCHED_MAX_COPIES=1` | 2656.37 / 2860.79 / 2647.42 / 2866.93 / 2681.12 | **2742.5** | 103.91 |

Delta: **+1.8% prefill, +0.1% generation** for one copy. Both are below the noise floor.

