# `mmvq-experiment/` — what these files are, and what is missing

Evidence for [chapter 15](../../15-mmvq-ampere-experiment.md) §15.3–§15.4, the
**`temperature 0.6`** half of the MMVQ Ampere cap experiment.

- Binary: `llama.cpp/build-mmvq/bin`, `build 10585, commit 95aab10a9` (patch
  `exp/mmvq-ampere-cap` on chapter 14's control base `c80c4c8a8`).
- Config: `-c 65536 -ub 512 -ctk/-ctv q8_0 -ts 22,43 -np 1`, 400-token code
  generation, `temperature 0.6`, one fresh server per point.
- Arms: `GGML_CUDA_MMVQ_MAX_BATCH` **unset** (upstream: Ampere k-quant MMVQ cap 8)
  versus **5** (what Blackwell already does for Q4_K).

## ⚠️ `mtp-results.md` is complete. `logs/` is NOT.

Every measured number is in `mtp-results.md`, which appends one section per sweep.
The logs are a **subset**, because per-env log filenames were only added partway
through — the fix is described in chapter 15 §15.7 and this directory is the
evidence of why it was needed.

| Files | Which run | What was lost |
| --- | --- | --- |
| `mtp_baseline.log`, `mtp_draft-mtp_n3..n8.log` (untagged) | First sweep, §15.3 | Both arms wrote these same names, arm A first. **These are arm B (cap 5) only** — arm A's logs were overwritten before anyone read them |
| `mtp_mmvqdef_*`, `mtp_mmvq5_*` | Confirmation run, §15.4 | Two repetitions per arm shared one tag, so **these are rep 2 only**; rep 1's logs were overwritten |

So: the untagged logs are **not** a control arm, despite looking like the default
case, and no log here corresponds to a first repetition. Do not pair a log with a
`mtp-results.md` row by name alone — only the tagged `n6/n7/n8` logs match their
row, and only for the *last* repetition of that cell.

Acceptance data survives for the confirmation run only; §15.3's first sweep ran
before the harness parsed the `draft acceptance = ... mean len = ...` line at all,
which is why its table has no acceptance column.

For the clean, fully-labelled half of the experiment see
[`../mmvq-experiment-greedy/`](../mmvq-experiment-greedy/PROVENANCE.md).
