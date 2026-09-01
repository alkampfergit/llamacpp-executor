# `tensor-split-ab/` - the user's layer-distribution change

Qwopus on `build-fa`, so scheduler copy count is fixed at one in both arms. All other arguments
are identical: `-c 8448 -ub 512 -ctk q8_0 -ctv q8_0`. Five repetitions were interleaved.

| split | prefill, five reps | mean | generation mean |
| --- | --- | ---: | ---: |
| `12,29` | 2371.47 / 2529.72 / 2491.55 / 2508.45 / 2442.50 | **2468.7** | 97.79 |
| `13,28` | 2877.72 / 2714.49 / 2869.70 / 2716.03 / 2740.27 | **2783.6** | 105.07 |

Changing only the split gives **+12.8% prefill** and +7.4% generation. It moves one layer from
the display GPU to the 8 GiB GPU; the logs and sampler show that memory moved between devices.
The experiment establishes the split as the large effect. It does not by itself identify
whether the cause is compute balance, display contention, or extra headroom on device 1.

