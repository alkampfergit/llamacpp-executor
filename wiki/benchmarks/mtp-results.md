# Speculative decoding results


## ctx=65536 ub=512 KV=q4_0/q4_0 predict=300

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 99,61 | 0% | False |
| draft-mtp n2 | 70,32 | -29% | True |

## ctx=65536 ub=512 KV=q4_0/q4_0 predict=300

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 97,23 | 0% | False |
| draft-mtp n1 | 90,05 | -7% | True |

## ctx=65536 ub=512 KV=q4_0/q4_0 predict=300

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 97,64 | 0% | False |
| ngram-simple n2 | 96,64 | -1% | False |
| ngram-simple n4 | 97,62 | -0% | False |
| ngram-simple n8 | 97,43 | -0% | False |
