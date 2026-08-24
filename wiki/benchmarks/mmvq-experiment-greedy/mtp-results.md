# Speculative decoding results


## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400 ts=22,43 temp=0

- bin: `S:\OsDevelop\llamacpp\llama.cpp\build-mmvq\bin`
- version: 0.2.0-dev (build 10585, commit 95aab10a9)
- GGML_CUDA_MMVQ_MAX_BATCH unset

| config | TG t/s | vs baseline | acceptance | mean accepted | out sha | MTP loaded |
|---|---|---|---|---|---|---|
| baseline | 22,42 | 0% | - | - | 68FAE608 | False |
| draft-mtp n6 | 54,55 | 143% | 0,7 | 5,18 | 68FAE608 | True |
| draft-mtp n7 | 52,91 | 136% | 0,643 | 5,47 | 68FAE608 | True |
| draft-mtp n8 | 55,44 | 147% | 0,6044 | 5,78 | 68FAE608 | True |

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400 ts=22,43 temp=0

- bin: `S:\OsDevelop\llamacpp\llama.cpp\build-mmvq\bin`
- version: 0.2.0-dev (build 10585, commit 95aab10a9)
- GGML_CUDA_MMVQ_MAX_BATCH=5

| config | TG t/s | vs baseline | acceptance | mean accepted | out sha | MTP loaded |
|---|---|---|---|---|---|---|
| baseline | 22,7 | 0% | - | - | 68FAE608 | False |
| draft-mtp n6 | 58,02 | 156% | 0,72 | 5,32 | 71D22399 | True |
| draft-mtp n7 | 58,2 | 156% | 0,67556 | 5,7 | DEB557C2 | True |
| draft-mtp n8 | 55,3 | 144% | 0,6044 | 5,78 | 68FAE608 | True |

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400 ts=22,43 temp=0

- bin: `S:\OsDevelop\llamacpp\llama.cpp\build-mmvq\bin`
- version: 0.2.0-dev (build 10585, commit 95aab10a9)
- GGML_CUDA_MMVQ_MAX_BATCH unset

| config | TG t/s | vs baseline | acceptance | mean accepted | out sha | MTP loaded |
|---|---|---|---|---|---|---|
| baseline | 22,71 | 0% | - | - | 68FAE608 | False |
| draft-mtp n6 | 56,81 | 150% | 0,7 | 5,18 | 68FAE608 | True |
| draft-mtp n7 | 53,39 | 135% | 0,643 | 5,47 | 68FAE608 | True |
| draft-mtp n8 | 54,41 | 140% | 0,6044 | 5,78 | 68FAE608 | True |

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400 ts=22,43 temp=0

- bin: `S:\OsDevelop\llamacpp\llama.cpp\build-mmvq\bin`
- version: 0.2.0-dev (build 10585, commit 95aab10a9)
- GGML_CUDA_MMVQ_MAX_BATCH=5

| config | TG t/s | vs baseline | acceptance | mean accepted | out sha | MTP loaded |
|---|---|---|---|---|---|---|
| baseline | 22,67 | 0% | - | - | 68FAE608 | False |
| draft-mtp n6 | 58,48 | 158% | 0,72 | 5,32 | 71D22399 | True |
| draft-mtp n7 | 57,17 | 152% | 0,67556 | 5,7 | DEB557C2 | True |
| draft-mtp n8 | 54,63 | 141% | 0,6044 | 5,78 | 68FAE608 | True |
