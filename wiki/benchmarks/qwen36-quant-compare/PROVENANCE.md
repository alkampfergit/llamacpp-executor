# `qwen36-quant-compare/` — read this before averaging anything

Evidence for [chapter 16 §16.3](../../16-best-commandlines.md): is "Q3_K CUDA kernels are slower
than Q4_K" a quant effect, or was chapter 6 comparing two different fine-tunes?

- Binaries: **the stock baseline** in `S:\OsDevelop\llamacpp\` (`build 10509, commit fe8156f78`),
  chosen because chapter 6's numbers came from these.
- Config, identical for every arm: `-c 32768 -ctk q8_0 -ctv q8_0 -ngl 999 -sm layer -ts 12,29
  -fa on -b 2048 -ub 512 -fit off`, `npp 8192 / ntg 128 / npl 1`.
- Arms: three quants of the **same base model**, `Qwen3.6-35B-A3B`.
  A vs B isolates the quant (same publisher, same imatrix method); B vs C isolates publisher.

## ⚠️ Two of the eleven rows are from a BROKEN run. Do not include them.

`results.tsv` holds two suites. **Filter on the `suite` column, never average the file.**

| `suite` | rows | use it? |
| --- | ---: | --- |
| `qwen36-quant` | 2 | ❌ **discard** |
| `qwen36-quant-v2` | 9 | ✅ the result |

The `qwen36-quant` rows are labelled `q36 C-lmstudio-Q4_K_M r1/r2` — **arm C only, and that is
the bug.** The first driver script validated model paths with `foreach ($m in $M) {...}`.
PowerShell variables are **case-insensitive**, so `$m` and `$M` are the same variable: the
validation loop left `$M` holding the *last hashtable* instead of the array, and the main loop
then iterated one element. It ran one arm out of three, exited 0, and wrote a plausible-looking
results file with no error anywhere.

The rewrite asserts `$MODELS.Count -eq 3` before starting, and uses `$mdl` as the loop variable.

## The v2 result

| arm | file | prefill (r1·r2·r3) | mean | generation | mean | peak VRAM |
| --- | --- | --- | ---: | --- | ---: | ---: |
| A | `unsloth/…UD-Q3_K_XL` | 2748·2802·2783 | **2778** | 91.2·88.3·92.0 | **90.5** | **18273** |
| B | `unsloth/…UD-Q4_K_S` | 2525·2457·2305 | 2429 | 92.5·89.2·92.6 | 91.5 | 22035 |
| C | `lmstudio-community/…Q4_K_M` | 2079·1361·1297 | 1579 | 100.0·79.5·94.5 | 91.3 | 22414 |

**Generation is quant-independent to within 1.0%.** Prefill is not: a 76% spread, with the
*smallest* file fastest.

**Arm C is unstable** — across all five of its runs (both suites) prefill went 1304 / 1952 /
2079 / 1361 / 1297 and generation dropped to 79.5. It is the largest file and sits nearest the
VRAM wall. A and B held within ~4%.

## Why three reps, interleaved

The first run of any model reads 17–21 GB from disk; later runs hit the OS file cache. The
discarded v1 rows measure exactly that — the *same* config at prefill 1304 then 1952, **+50%**,
with nothing changed. Arms are interleaved (A,B,C / A,B,C / A,B,C) so rep 1 is cold for every
arm rather than for one.

Logs are complete: 11 rows, 11 files in `logs/`, one per row, named by label.
