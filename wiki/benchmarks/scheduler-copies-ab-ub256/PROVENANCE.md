# `scheduler-copies-ab-ub256/` - scheduler-copy pilot

Qwopus, matched rebuilds, identical `-c 32768 -ub 256 -ts 13,28 -ctk q8_0 -ctv q8_0` arguments.
The arms were interleaved for three repetitions.

| arm | prefill | mean | generation mean |
| --- | --- | ---: | ---: |
| `build-control`, `GGML_SCHED_MAX_COPIES=4` | 1802.27 / 2041.50 / 1873.72 | **1905.8** | 101.6 |
| `build-fa`, `GGML_SCHED_MAX_COPIES=1` | 2141.26 / 1944.98 / 2131.59 | **2072.6** | 100.5 |

The single-copy arm is +8.8% prefill, exactly at this repo's practical display-GPU noise
boundary, and generation is unchanged. This is a batch-size-specific lead, not evidence for a
general fallback speedup; the five-repetition `ub512` campaign is the primary result.

