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

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 22,55 | 0% | True |
| draft-mtp n1 | 35,17 | 56% | True |
| draft-mtp n2 | 41,07 | 82% | True |
| draft-mtp n3 | 44,99 | 100% | True |

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 22,42 | 0% | True |
| draft-mtp n4 | 47,09 | 110% | True |
| draft-mtp n5 | 38,08 | 70% | True |
| draft-mtp n6 | 44,41 | 98% | True |
| draft-mtp n8 | 47,21 | 111% | True |

## ctx=65536 ub=512 KV=q8_0/q8_0 predict=400

| config | TG t/s | vs baseline | MTP loaded |
|---|---|---|---|
| baseline | 22,42 | 0% | True |
| ngram-simple n4 | 22,33 | -0% | True |
| ngram-simple n8 | 22,35 | -0% | True |
