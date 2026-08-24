# `scheduler-copies-ab/` - aborted preflight

This was the first attempt to compare the matched `build-control` and `build-fa` binaries on
Qwopus. It is retained because it exposed and fixed a harness error classification.

- Config: `-c 32768 -ub 512 -ts 14,27`, symmetric `q8_0` KV.
- Result: all three attempted runs failed after both the pipelined reservation and its retry.
- Cause: `14,27` put too much on the 8 GiB device. No throughput row was produced.
- Harness fix: a retry warning without a result is now `fallback-failed`, while a retry followed
  by a benchmark row is `fallback-single-copy` and status `OK`. The campaign runner now stops on
  the first arm that does not produce a result.

These rows are negative evidence only. Do not include them in performance averages.

