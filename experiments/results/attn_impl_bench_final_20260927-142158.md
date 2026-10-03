
## 汇总（以 std 为基准，>1 表示 sdpa 更快/更省）

| stage | mask | branch | seq | compile | 对比 | ms_step 比 | ktok/s 比 | 设备显存比 |
|---|---|---|---|---|---|---|---|---|
| capacity | pad | gate | 256 | False | sdpa/std | 1.39x | 1.39x | 0.71 |
| capacity | pad | gate | 256 | False | 最大 batch sdpa/std | - | - | 1.00x |
| model | none | prod | 256 | False | sdpa/std | 1.14x | 1.14x | 0.79 |
| model | none | prod | 256 | True | sdpa/std | 1.02x | 1.01x | 0.94 |
| model | pad | prod | 256 | False | sdpa/std | 1.12x | 1.10x | 0.80 |
| model | pad | prod | 256 | True | sdpa/std | 0.99x | 0.99x | 0.95 |
| module | none | gate | 256 | False | sdpa/std | 1.48x | 1.49x | 0.64 |
| module | none | gate | 256 | True | sdpa/std | 0.74x | 0.75x | 1.00 |
| module | pad | gate | 256 | False | sdpa/std | 1.37x | 1.36x | 0.75 |
| module | pad | gate | 256 | True | sdpa/std | 0.97x | 0.95x | 0.96 |