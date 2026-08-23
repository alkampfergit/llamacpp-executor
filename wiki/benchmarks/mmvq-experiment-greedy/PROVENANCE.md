# `mmvq-experiment-greedy/` — what these files are

Evidence for [chapter 15](../../15-mmvq-ampere-experiment.md) §15.5, the
**`temperature 0`** half of the MMVQ Ampere cap experiment. This is the half that
settled it.

- Binary: `llama.cpp/build-mmvq/bin`, `build 10585, commit 95aab10a9` (patch
  `exp/mmvq-ampere-cap` on chapter 14's control base `c80c4c8a8`).
- Config: `-c 65536 -ub 512 -ctk/-ctv q8_0 -ts 22,43 -np 1`, 400-token code
  generation, **`temperature 0`**, one fresh server per point.
- Arms interleaved (`unset, 5, unset, 5`) so thermal drift could not masquerade
  as an arm effect. `n_max` 6/7/8, two repetitions each.

Greedy decoding is the point of this directory: at `temperature 0.6` throughput
tracks resampled draft acceptance at r² = 0.872 with a spread up to 33%, which is
four times the effect being measured (§15.4). Greedy makes acceptance repeatable —
it came out bit-identical across repetitions in every cell — and drops the
rep-to-rep spread to 0.8–4.1%.

## The verdict lives in the `out sha` column

`mtp-results.md` records a SHA of the generated text per run. MMVQ and MMQ are not
bit-identical, so the cap can change the greedy token sequence, and then the two
arms are no longer computing the same thing:

| `n_max` | output text | reading |
| ---: | --- | --- |
| 6, 7 | **differs** between arms | Acceptance changed (0.700→0.720, 0.643→0.676). The +4.6% / +8.5% is a different token path, not cheaper kernels |
| 8 | **identical** between arms | Same acceptance 0.6044, same mean accepted 5.78. **The cap is worth +0.1%.** This row is the result |

## ⚠️ `logs/` holds rep 2 only

`mtp-results.md` is complete — four sections, one per (rep, arm). The logs are
**rep 2 only**: both repetitions of an arm share one env-derived filename tag, so
rep 1 was overwritten. Per-env tagging fixed the arm collision (§15.7) but not the
repetition collision; a `-RunTag` would be the next fix if this matters.
