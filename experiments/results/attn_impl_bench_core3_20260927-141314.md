
## 汇总（以 std 为基准，>1 表示 sdpa 更快/更省）

| stage | mask | branch | seq | compile | 对比 | ms_step 比 | ktok/s 比 | 设备显存比 |
|---|---|---|---|---|---|---|---|---|
| core | none |  | 256 | False | sdpa/std | 2.11x | 2.10x | 0.45 |
| core | none |  | 256 | False | sdpa_cudnn/std | 2.42x | 2.39x | 0.47 |
| core | none |  | 256 | False | sdpa_memeff/std | 2.09x | 2.06x | 0.45 |
| core | none |  | 256 | False | sdpa_math/std | 0.99x | 0.98x | 0.64 |
| core | none |  | 256 | True | sdpa/std | 1.11x | 1.13x | 1.02 |
| core | none |  | 512 | False | sdpa/std | 3.10x | 3.02x | 0.42 |
| core | none |  | 512 | False | sdpa_cudnn/std | 3.87x | 3.82x | 0.43 |
| core | none |  | 512 | False | sdpa_memeff/std | 3.08x | 2.99x | 0.42 |
| core | none |  | 512 | False | sdpa_math/std | 1.34x | 1.32x | 0.89 |
| core | none |  | 512 | True | sdpa/std | 1.22x | 1.23x | 1.08 |
| core | none |  | 1024 | False | sdpa/std | 3.68x | 3.57x | 0.25 |
| core | none |  | 1024 | False | sdpa_cudnn/std | 5.64x | 5.54x | 0.26 |
| core | none |  | 1024 | False | sdpa_memeff/std | 3.66x | 3.57x | 0.25 |
| core | none |  | 1024 | False | sdpa_math/std | 1.31x | 1.32x | 0.87 |
| core | none |  | 1024 | True | sdpa/std | 1.09x | 1.10x | 0.83 |
| core | none |  | 2048 | False | sdpa/std | 4.41x | 4.27x | 0.10 |
| core | none |  | 2048 | False | sdpa_cudnn/std | 8.07x | 7.83x | 0.10 |
| core | none |  | 2048 | False | sdpa_memeff/std | 4.39x | 4.24x | 0.10 |
| core | none |  | 2048 | False | sdpa_math/std | 1.38x | 1.37x | 0.79 |
| core | none |  | 2048 | True | sdpa/std | 1.35x | 1.33x | 0.42 |
| core | pad |  | 256 | False | sdpa/std | 1.66x | 1.66x | 0.68 |
| core | pad |  | 256 | False | sdpa_cudnn/std | 2.00x | 2.00x | 0.71 |
| core | pad |  | 256 | False | sdpa_memeff/std | 1.62x | 1.63x | 0.68 |
| core | pad |  | 256 | False | sdpa_math/std | 0.99x | 0.98x | 0.89 |
| core | pad |  | 256 | True | sdpa/std | 1.00x | 0.98x | 1.03 |
| core | pad |  | 512 | False | sdpa/std | 2.05x | 2.05x | 0.49 |
| core | pad |  | 512 | False | sdpa_cudnn/std | 2.94x | 2.98x | 0.49 |
| core | pad |  | 512 | False | sdpa_memeff/std | 2.09x | 2.07x | 0.49 |
| core | pad |  | 512 | False | sdpa_math/std | 1.40x | 1.37x | 0.92 |
| core | pad |  | 512 | True | sdpa/std | 0.96x | 0.98x | 0.80 |
| core | pad |  | 1024 | False | sdpa/std | 2.17x | 2.20x | 0.24 |
| core | pad |  | 1024 | False | sdpa_cudnn/std | 3.70x | 3.69x | 0.24 |
| core | pad |  | 1024 | False | sdpa_memeff/std | 2.15x | 2.19x | 0.24 |
| core | pad |  | 1024 | False | sdpa_math/std | 1.22x | 1.23x | 0.82 |
| core | pad |  | 1024 | True | sdpa/std | 0.75x | 0.76x | 0.76 |
| core | pad |  | 2048 | False | sdpa/std | 2.34x | 2.31x | 0.15 |
| core | pad |  | 2048 | False | sdpa_cudnn/std | 3.68x | 3.69x | 0.15 |
| core | pad |  | 2048 | False | sdpa_memeff/std | 2.13x | 2.17x | 0.15 |
| core | pad |  | 2048 | False | sdpa_math/std | 1.29x | 1.30x | 0.82 |
| core | pad |  | 2048 | True | sdpa/std | 0.78x | 0.78x | 0.41 |