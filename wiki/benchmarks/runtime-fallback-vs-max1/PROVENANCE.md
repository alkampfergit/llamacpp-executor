# `runtime-fallback-vs-max1/` - is the retry itself faster?

This is the decisive same-source test. Both arms use Qwopus with identical
`-c 64000 -ub 512 -ts 13,28 -ctk q8_0 -ctv q8_0` arguments and five interleaved repetitions.

- `build-control` (`MAX_COPIES=4`) fails its larger reservation and succeeds through the runtime
  `retrying without pipeline parallelism` path on every repetition.
- `build-fa` (`MAX_COPIES=1`) fits directly and does not retry.
- The build commits differ only in their provenance manifest; symmetric KV keeps the other
  treatment flag inert.

| arm | prefill, five reps | mean | generation mean |
| --- | --- | ---: | ---: |
| runtime fallback | 2643.65 / 2653.87 / 2654.53 / 2709.91 / 2603.71 | **2653.1** | 104.55 |
| one-copy build | 2721.56 / 2882.45 / 2718.71 / 2633.57 / 2524.21 | **2696.1** | 104.45 |

Delta: **+1.6% prefill, -0.1% generation** for the one-copy build. The retry itself is not a
material throughput advantage. Its important effect is recovery with a smaller reservation.

