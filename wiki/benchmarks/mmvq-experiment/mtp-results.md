# Speculative decoding results


## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400 ts=22,43

- bin: `S:\OsDevelop\llamacpp\llama.cpp\build-mmvq\bin`
- version: 0.2.0-dev (build 10585, commit 95aab10a9)
- GGML_CUDA_MMVQ_MAX_BATCH unset

| config | TG t/s | vs baseline | acceptance | mean accepted | MTP loaded |
|---|---|---|---|---|---|
| baseline | 22,63 | 0% | - | - | False |
| draft-mtp n3 | 47,35 | 109% | - | - | True |
| draft-mtp n4 | 48,91 | 116% | - | - | True |
| draft-mtp n5 | 51,82 | 129% | - | - | True |
| draft-mtp n6 | 54,05 | 139% | - | - | True |
| draft-mtp n7 | 43,44 | 92% | - | - | True |
| draft-mtp n8 | 42,75 | 89% | - | - | True |

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400 ts=22,43

- bin: `S:\OsDevelop\llamacpp\llama.cpp\build-mmvq\bin`
- version: 0.2.0-dev (build 10585, commit 95aab10a9)
- GGML_CUDA_MMVQ_MAX_BATCH=5

| config | TG t/s | vs baseline | acceptance | mean accepted | MTP loaded |
|---|---|---|---|---|---|
| baseline | 22,64 | 0% | - | - | False |
| draft-mtp n3 | 45,21 | 100% | - | - | True |
| draft-mtp n4 | 49,5 | 119% | - | - | True |
| draft-mtp n5 | 49,17 | 117% | - | - | True |
| draft-mtp n6 | 50,72 | 124% | - | - | True |
| draft-mtp n7 | 50,08 | 121% | - | - | True |
| draft-mtp n8 | 51,97 | 130% | - | - | True |

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400 ts=22,43

- bin: `S:\OsDevelop\llamacpp\llama.cpp\build-mmvq\bin`
- version: 0.2.0-dev (build 10585, commit 95aab10a9)
- GGML_CUDA_MMVQ_MAX_BATCH unset

| config | TG t/s | vs baseline | acceptance | mean accepted | MTP loaded |
|---|---|---|---|---|---|
| baseline | 22,64 | 0% | - | - | False |
| draft-mtp n6 | 44,94 | 98% | 0,53065 | 4,16 | True |
| draft-mtp n7 | 43,75 | 93% | 0,50325 | 4,52 | True |
| draft-mtp n8 | 51,58 | 128% | 0,55272 | 5,39 | True |

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400 ts=22,43

- bin: `S:\OsDevelop\llamacpp\llama.cpp\build-mmvq\bin`
- version: 0.2.0-dev (build 10585, commit 95aab10a9)
- GGML_CUDA_MMVQ_MAX_BATCH=5

| config | TG t/s | vs baseline | acceptance | mean accepted | MTP loaded |
|---|---|---|---|---|---|
| baseline | 22,68 | 0% | - | - | False |
| draft-mtp n6 | 53,5 | 136% | 0,66597 | 4,99 | True |
| draft-mtp n7 | 59,68 | 163% | 0,70426 | 5,87 | True |
| draft-mtp n8 | 42,18 | 86% | 0,44492 | 4,53 | True |

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400 ts=22,43

- bin: `S:\OsDevelop\llamacpp\llama.cpp\build-mmvq\bin`
- version: 0.2.0-dev (build 10585, commit 95aab10a9)
- GGML_CUDA_MMVQ_MAX_BATCH unset

| config | TG t/s | vs baseline | acceptance | mean accepted | MTP loaded |
|---|---|---|---|---|---|
| baseline | 21,95 | 0% | - | - | False |
| draft-mtp n6 | 50,34 | 129% | 0,64562 | 4,87 | True |
| draft-mtp n7 | 52,27 | 138% | 0,6323 | 5,39 | True |
| draft-mtp n8 | 46,27 | 111% | 0,49687 | 4,93 | True |

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400 ts=22,43

- bin: `S:\OsDevelop\llamacpp\llama.cpp\build-mmvq\bin`
- version: 0.2.0-dev (build 10585, commit 95aab10a9)
- GGML_CUDA_MMVQ_MAX_BATCH=5

| config | TG t/s | vs baseline | acceptance | mean accepted | MTP loaded |
|---|---|---|---|---|---|
| baseline | 22,56 | 0% | - | - | False |
| draft-mtp n6 | 49,75 | 121% | 0,58052 | 4,48 | True |
| draft-mtp n7 | 44,84 | 99% | 0,4944 | 4,43 | True |
| draft-mtp n8 | 42,66 | 89% | 0,46222 | 4,63 | True |
