# CLAUDE.md — llamacpp-executor

Prebuilt Windows CUDA llama.cpp binaries in the root (gitignored), a tuning wiki in `wiki/`,
and the upstream source as a submodule in `llama.cpp/`.

Hardware: RTX 3070 (8 GiB, device 0) + RTX 5060 Ti (16 GiB, device 1, **drives the display**).
Compute capabilities `8.6` and `12.0` → `-DCMAKE_CUDA_ARCHITECTURES="86;120"`.

## When a benchmark result looks strange, read the source

**A surprising measurement is a signal to go read llama.cpp's source in `llama.cpp/`, not to
theorise about it.** Guessing at causes has produced wrong conclusions in this repo twice, and
both were settled in minutes by reading the code:

| Symptom | Wrong guess | What the source said |
| --- | --- | --- |
| KV cache `q8_0`/`q4_0` was 14× slower than `q8_0`/`q8_0` | "4-bit KV is slow" | `ggml/src/ggml-cuda/fattn.cu`: `if (K->type != V->type) return BEST_FATTN_KERNEL_NONE;` — only four *symmetric* pairs are compiled. Attention was leaving the GPU. |
| Setting `$env:GGML_SCHED_MAX_COPIES=1` appeared to gain 13% | "it disables pipeline parallelism" | `ggml/src/ggml-backend.cpp`: it is a compile-time `#define`, never read from the environment. The gain was noise. |

Applies to: a result that contradicts a documented claim, a flag that "works" but is
inexplicably slow, non-monotonic behaviour, a silent fallback, or any gain near the ±8% noise
floor. Prefer `Grep` over the submodule to a web search — the checked-out code is the code
that produced the measurement.

## Keep the submodule current before reading it

Stale source explains nothing about a binary built from a newer tree. Before investigating,
fetch and rebase:

```powershell
cd S:\OneDrive\Tools\llamacpp\llama.cpp
git fetch upstream
git rebase upstream/master          # NOTE: master, not main
```

- `upstream` = `https://github.com/ggml-org/llama.cpp.git` (push URL deliberately disabled).
  `origin` = the user's fork, `alkampfergit/llama.cpp`.
- **The default branch is `master`.** `upstream/main` does not exist.
- If the rebase would not fast-forward, stop and report rather than resolving conflicts
  unasked — local commits on the fork may be deliberate.
- Moving the submodule's checkout changes the gitlink in the parent repo and needs its own
  commit there. Mention it; don't commit it silently.
- **Check the source you read matches the binary you measured.** `llama-server.exe --version`
  reports `version: 0.1.2-dev (build 10509, commit fe8156f78)` — compare that commit against
  the submodule's `git log -1`. If they diverge, say so before attributing observed behaviour
  to the code you just read. (`llama-bench` has no `--version`; use `llama-server` or
  `llama-cli`, or read the `build:` line at the end of normal `llama-bench` output.)

## Skills

- `tuning-llamacpp-configs` — find optimal runtime flags (context, ubatch, split, KV type).
- `building-llamacpp-cuda` — build from source. The plain `cmake -B build -DGGML_CUDA=ON …`
  from llama.cpp's docs **fails on this machine**; use the skill.

## Conventions

- **Commit unsigned** here (`commit.gpgsign false` is set repo-locally); the user signs at PR
  close.
- **Never commit** `*.gguf`, or the root `*.dll` / `*.exe` (668 MB of CUDA runtime).
- **Benchmark evidence is committed on purpose** — `wiki/benchmarks/results.tsv`, the per-run
  logs, and `needle-prompt.txt` (marked `-text`, since line-ending normalisation would change
  its token count). Flush each run to disk as it completes; GPU-heavy runs here have crashed.
- Differences under **~8%** are noise when a GPU also drives the display.
