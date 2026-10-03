# 一、绝对值（fwd+bwd 训练态，bf16 autocast）

| stage | mask | branch | seq | batch | compile | impl | ms_fp | ms_step | ktok/s | TFLOP/s(满算) | 设备峰值MiB | 分配器峰值MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| capacity | pad | gate | 256 | 256 | False | sdpa | 22.930 | 58.200 | 1125.0 | - | 2016.0 | 1756.0 |
| capacity | pad | gate | 256 | 256 | False | std | 32.840 | 80.900 | 810.6 | - | 2846.0 | 2156.0 |
| core | none | - | 256 | 32 | False | sdpa | 0.470 | 2.020 | 3934.0 | 6.38 | 244.0 | 272.1 |
| core | none | - | 256 | 32 | False | sdpa-cudnn | 0.340 | 1.759 | 4477.0 | 7.33 | 254.0 | 272.1 |
| core | none | - | 256 | 32 | False | std | 1.378 | 4.254 | 1872.0 | 3.03 | 542.0 | 320.1 |
| core | none | - | 256 | 32 | True | sdpa | 0.773 | 2.433 | 3310.0 | 5.29 | 282.0 | 176.0 |
| core | none | - | 256 | 32 | True | std | 0.862 | 2.703 | 2929.0 | 4.77 | 276.0 | 176.0 |
| core | none | - | 512 | 16 | False | sdpa | 0.603 | 2.677 | 2980.0 | 9.62 | 216.0 | 272.1 |
| core | none | - | 512 | 16 | False | sdpa-cudnn | 0.397 | 2.147 | 3767.0 | 12.00 | 220.0 | 272.1 |
| core | none | - | 512 | 16 | False | std | 3.015 | 8.302 | 985.8 | 3.10 | 516.0 | 440.3 |
| core | none | - | 512 | 16 | True | sdpa | 0.926 | 3.117 | 2572.0 | 8.27 | 286.0 | 176.0 |
| core | none | - | 512 | 16 | True | std | 1.253 | 3.788 | 2094.0 | 6.80 | 264.0 | 176.0 |
| core | none | - | 1024 | 8 | False | sdpa | 0.881 | 3.719 | 2144.0 | 13.86 | 212.0 | 272.1 |
| core | none | - | 1024 | 8 | False | sdpa-cudnn | 0.536 | 2.430 | 3327.0 | 21.21 | 216.0 | 272.1 |
| core | none | - | 1024 | 8 | False | std | 5.305 | 13.700 | 600.0 | 3.76 | 836.0 | 697.0 |
| core | none | - | 1024 | 8 | True | sdpa | 1.198 | 4.116 | 1941.0 | 12.52 | 286.0 | 176.0 |
| core | none | - | 1024 | 8 | True | std | 1.700 | 4.495 | 1768.0 | 11.47 | 344.0 | 176.0 |
| core | none | - | 2048 | 4 | False | sdpa | 1.406 | 5.838 | 1368.0 | 17.66 | 148.0 | 272.1 |
| core | none | - | 2048 | 4 | False | sdpa-cudnn | 0.742 | 3.187 | 2511.0 | 32.34 | 152.0 | 272.1 |
| core | none | - | 2048 | 4 | False | std | 10.740 | 25.720 | 320.7 | 4.01 | 1538.0 | 1212.0 |
| core | none | - | 2048 | 4 | True | sdpa | 1.738 | 6.191 | 1294.0 | 16.65 | 286.0 | 176.0 |
| core | none | - | 2048 | 4 | True | std | 2.814 | 8.385 | 975.0 | 12.29 | 680.0 | 176.0 |
| core | pad | - | 256 | 32 | False | sdpa | 0.778 | 2.661 | 2989.0 | 4.84 | 268.0 | 286.1 |
| core | pad | - | 256 | 32 | False | sdpa-cudnn | 0.597 | 2.210 | 3605.0 | 5.83 | 278.0 | 286.1 |
| core | pad | - | 256 | 32 | False | std | 1.513 | 4.421 | 1803.0 | 2.91 | 394.0 | 326.0 |
| core | pad | - | 256 | 32 | True | sdpa | 0.961 | 3.187 | 2589.0 | 4.04 | 310.0 | 178.0 |
| core | pad | - | 256 | 32 | True | std | 1.315 | 3.178 | 2642.0 | 4.05 | 300.0 | 178.0 |
| core | pad | - | 512 | 16 | False | sdpa | 1.179 | 4.353 | 1873.0 | 5.92 | 252.0 | 292.1 |
| core | pad | - | 512 | 16 | False | sdpa-cudnn | 0.697 | 3.030 | 2724.0 | 8.51 | 256.0 | 292.1 |
| core | pad | - | 512 | 16 | False | std | 3.345 | 8.915 | 915.0 | 2.89 | 518.0 | 448.0 |
| core | pad | - | 512 | 16 | True | sdpa | 1.906 | 4.269 | 1855.0 | 6.04 | 318.0 | 180.0 |
| core | pad | - | 512 | 16 | True | std | 1.277 | 4.085 | 1900.0 | 6.31 | 400.0 | 180.0 |
| core | pad | - | 1024 | 8 | False | sdpa | 1.906 | 7.184 | 1156.0 | 7.17 | 204.0 | 304.1 |
| core | pad | - | 1024 | 8 | False | sdpa-cudnn | 1.048 | 4.221 | 1940.0 | 12.21 | 208.0 | 304.1 |
| core | pad | - | 1024 | 8 | False | std | 6.480 | 15.620 | 525.6 | 3.30 | 854.0 | 712.0 |
| core | pad | - | 1024 | 8 | True | sdpa | 2.032 | 6.979 | 1163.0 | 7.38 | 330.0 | 184.0 |
| core | pad | - | 1024 | 8 | True | std | 1.763 | 5.243 | 1533.0 | 9.83 | 432.0 | 184.0 |
| core | pad | - | 2048 | 4 | False | sdpa | 3.538 | 11.470 | 709.1 | 8.98 | 236.0 | 328.1 |
| core | pad | - | 2048 | 4 | False | sdpa-cudnn | 2.123 | 7.292 | 1131.0 | 14.14 | 240.0 | 328.1 |
| core | pad | - | 2048 | 4 | False | std | 11.210 | 26.850 | 307.0 | 3.84 | 1554.0 | 1240.0 |
| core | pad | - | 2048 | 4 | True | sdpa | 3.876 | 12.910 | 632.8 | 7.98 | 354.0 | 192.0 |
| core | pad | - | 2048 | 4 | True | std | 2.972 | 10.060 | 814.5 | 10.25 | 874.0 | 192.0 |
| model | none | prod | 256 | 64 | False | sdpa | 59.690 | 170.300 | 96.2 | - | 4800.0 | 4526.0 |
| model | none | prod | 256 | 64 | False | std | 69.330 | 193.500 | 84.7 | - | 6055.0 | 4865.0 |
| model | none | prod | 256 | 64 | True | sdpa | 34.770 | 105.500 | 153.3 | - | 3998.0 | 1326.0 |
| model | none | prod | 256 | 64 | True | std | 36.210 | 107.300 | 152.3 | - | 4246.0 | 1326.0 |
| model | pad | prod | 256 | 64 | False | sdpa | 69.120 | 173.100 | 93.0 | - | 4890.0 | 4605.0 |
| model | pad | prod | 256 | 64 | False | std | 70.230 | 193.400 | 84.6 | - | 6102.0 | 4895.0 |
| model | pad | prod | 256 | 64 | True | sdpa | 35.620 | 106.700 | 153.3 | - | 3946.0 | 1326.0 |
| model | pad | prod | 256 | 64 | True | std | 36.410 | 105.800 | 154.4 | - | 4156.0 | 1326.0 |
| module | none | gate | 256 | 64 | False | sdpa | 6.296 | 16.570 | 997.5 | - | 592.0 | 515.5 |
| module | none | gate | 256 | 64 | False | std | 10.110 | 24.450 | 670.9 | - | 918.0 | 629.5 |
| module | none | gate | 256 | 64 | True | sdpa | 2.569 | 12.080 | 1348.0 | - | 474.0 | 202.3 |
| module | none | gate | 256 | 64 | True | std | 3.565 | 8.937 | 1786.0 | - | 474.0 | 202.3 |
| module | pad | gate | 256 | 64 | False | sdpa | 6.629 | 16.250 | 1004.0 | - | 628.0 | 543.5 |
| module | pad | gate | 256 | 64 | False | std | 8.796 | 22.210 | 739.0 | - | 834.0 | 645.3 |
| module | pad | gate | 256 | 64 | True | sdpa | 2.905 | 8.891 | 1794.0 | - | 494.0 | 206.3 |
| module | pad | gate | 256 | 64 | True | std | 3.064 | 8.618 | 1891.0 | - | 512.0 | 206.3 |

# 二、capacity：同显存预算下能开多大 batch

| mask | seq | compile | impl | 最大 batch | tokens/step | ms_step | ktok/s | 设备峰值MiB |
|---|---|---|---|---|---|---|---|---|
| pad | 256 | False | std | 256 | 65536 | 80.90 | 810.6 | 2846.0 |
| pad | 256 | False | sdpa | 256 | 65536 | 58.20 | 1125.0 | 2016.0 |
| pad | 256 | False | **sdpa/std** | 1.00x | 1.00x | - | 1.39x | 1.41x |

# 三、比值（一律 std/sdpa：耗时、吞吐 >1 表示 sdpa 更快；显存列 >1 表示 sdpa 更省）

| stage | mask | branch | seq | batch | compile | 对比 | 前向加速 | 单步加速 | 吞吐提升 | 显存节省 |
|---|---|---|---|---|---|---|---|---|---|---|
| core | none | - | 256 | 32 | False | sdpa/std | 2.93x | 2.11x | 2.10x | 2.22x |
| core | none | - | 256 | 32 | False | sdpa-cudnn/std | 4.05x | 2.42x | 2.39x | 2.13x |
| core | none | - | 256 | 32 | False | sdpa-math/std | 0.80x | 0.99x | 0.98x | 1.57x |
| core | none | - | 256 | 32 | False | sdpa-memeff/std | 2.86x | 2.09x | 2.06x | 2.22x |
| core | none | - | 256 | 32 | True | sdpa/std | 1.11x | 1.11x | 1.13x | 0.98x |
| core | none | - | 512 | 16 | False | sdpa/std | 5.00x | 3.10x | 3.02x | 2.39x |
| core | none | - | 512 | 16 | False | sdpa-cudnn/std | 7.60x | 3.87x | 3.82x | 2.35x |
| core | none | - | 512 | 16 | False | sdpa-math/std | 1.10x | 1.34x | 1.32x | 1.13x |
| core | none | - | 512 | 16 | False | sdpa-memeff/std | 4.74x | 3.08x | 2.99x | 2.39x |
| core | none | - | 512 | 16 | True | sdpa/std | 1.35x | 1.22x | 1.23x | 0.92x |
| core | none | - | 1024 | 8 | False | sdpa/std | 6.02x | 3.68x | 3.57x | 3.94x |
| core | none | - | 1024 | 8 | False | sdpa-cudnn/std | 9.90x | 5.64x | 5.54x | 3.87x |
| core | none | - | 1024 | 8 | False | sdpa-math/std | 1.12x | 1.31x | 1.32x | 1.15x |
| core | none | - | 1024 | 8 | False | sdpa-memeff/std | 5.96x | 3.66x | 3.57x | 3.94x |
| core | none | - | 1024 | 8 | True | sdpa/std | 1.42x | 1.09x | 1.10x | 1.20x |
| core | none | - | 2048 | 4 | False | sdpa/std | 7.64x | 4.41x | 4.27x | 10.39x |
| core | none | - | 2048 | 4 | False | sdpa-cudnn/std | 14.47x | 8.07x | 7.83x | 10.12x |
| core | none | - | 2048 | 4 | False | sdpa-math/std | 1.10x | 1.38x | 1.37x | 1.26x |
| core | none | - | 2048 | 4 | False | sdpa-memeff/std | 7.56x | 4.39x | 4.24x | 10.39x |
| core | none | - | 2048 | 4 | True | sdpa/std | 1.62x | 1.35x | 1.33x | 2.38x |
| core | pad | - | 256 | 32 | False | sdpa/std | 1.95x | 1.66x | 1.66x | 1.47x |
| core | pad | - | 256 | 32 | False | sdpa-cudnn/std | 2.54x | 2.00x | 2.00x | 1.42x |
| core | pad | - | 256 | 32 | False | sdpa-math/std | 0.78x | 0.99x | 0.98x | 1.13x |
| core | pad | - | 256 | 32 | False | sdpa-memeff/std | 1.92x | 1.62x | 1.63x | 1.47x |
| core | pad | - | 256 | 32 | True | sdpa/std | 1.37x | 1.00x | 0.98x | 0.97x |
| core | pad | - | 512 | 16 | False | sdpa/std | 2.84x | 2.05x | 2.05x | 2.06x |
| core | pad | - | 512 | 16 | False | sdpa-cudnn/std | 4.80x | 2.94x | 2.98x | 2.02x |
| core | pad | - | 512 | 16 | False | sdpa-math/std | 0.99x | 1.40x | 1.37x | 1.09x |
| core | pad | - | 512 | 16 | False | sdpa-memeff/std | 2.97x | 2.10x | 2.07x | 2.06x |
| core | pad | - | 512 | 16 | True | sdpa/std | 0.67x | 0.96x | 0.98x | 1.26x |
| core | pad | - | 1024 | 8 | False | sdpa/std | 3.40x | 2.17x | 2.20x | 4.19x |
| core | pad | - | 1024 | 8 | False | sdpa-cudnn/std | 6.18x | 3.70x | 3.69x | 4.11x |
| core | pad | - | 1024 | 8 | False | sdpa-math/std | 1.09x | 1.22x | 1.23x | 1.22x |
| core | pad | - | 1024 | 8 | False | sdpa-memeff/std | 3.44x | 2.15x | 2.19x | 4.19x |
| core | pad | - | 1024 | 8 | True | sdpa/std | 0.87x | 0.75x | 0.76x | 1.31x |
| core | pad | - | 2048 | 4 | False | sdpa/std | 3.17x | 2.34x | 2.31x | 6.58x |
| core | pad | - | 2048 | 4 | False | sdpa-cudnn/std | 5.28x | 3.68x | 3.68x | 6.47x |
| core | pad | - | 2048 | 4 | False | sdpa-math/std | 1.02x | 1.29x | 1.30x | 1.22x |
| core | pad | - | 2048 | 4 | False | sdpa-memeff/std | 3.09x | 2.13x | 2.17x | 6.58x |
| core | pad | - | 2048 | 4 | True | sdpa/std | 0.77x | 0.78x | 0.78x | 2.47x |
| model | none | prod | 256 | 64 | False | sdpa/std | 1.16x | 1.14x | 1.14x | 1.26x |
| model | none | prod | 256 | 64 | True | sdpa/std | 1.04x | 1.02x | 1.01x | 1.06x |
| model | pad | prod | 256 | 64 | False | sdpa/std | 1.02x | 1.12x | 1.10x | 1.25x |
| model | pad | prod | 256 | 64 | True | sdpa/std | 1.02x | 0.99x | 0.99x | 1.05x |
| module | none | gate | 256 | 64 | False | sdpa/std | 1.61x | 1.48x | 1.49x | 1.55x |
| module | none | gate | 256 | 64 | True | sdpa/std | 1.39x | 0.74x | 0.75x | 1.00x |
| module | pad | gate | 256 | 64 | False | sdpa/std | 1.33x | 1.37x | 1.36x | 1.33x |
| module | pad | gate | 256 | 64 | True | sdpa/std | 1.05x | 0.97x | 0.95x | 1.04x |
