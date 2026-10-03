# 手写注意力 vs F.scaled_dot_product_attention 实测结论

数据源：`attn_impl_bench_core3_*.csv`（core 级全网格）、`attn_impl_bench_final_*.csv`
（module / model / capacity），机器 RTX 5060 8GB / torch 2.13.0+cu132。
被测对象为工作区现行 `models.py:Attention`（QK-Norm→RoPE、V 不做激活、
softmax 后 attn_dropout=0.05），两侧口径已逐项对齐（含 dropout 与 4D padding mask）。
全部计时为 fwd+bwd 训练态、bf16 autocast、`set_float32_matmul_precision("high")`。

## 一、core 级：只测算子本身（B×L=8192 tokens 固定）

| seq | 实现 | ms_step | 设备峰值 MiB | vs std |
|---|---|---|---|---|
| 256 | std eager | 4.29 | 542 | 1.00x |
| 256 | sdpa eager（默认=mem-efficient） | 2.02 | 244 | 2.11x |
| 256 | sdpa eager 强制 cuDNN | 1.77 | 254 | **2.42x** |
| 256 | sdpa eager 强制 math | 4.69 | 346 | 0.99x |
| 1024 | std eager | 13.70 | 836 | 1.00x |
| 1024 | sdpa eager（默认） | 3.72 | 212 | 3.68x |
| 1024 | sdpa eager 强制 cuDNN | 2.43 | 216 | **5.64x** |
| 1024 | sdpa eager 强制 math | 10.4 | 762 | 1.31x |
| 2048 | std eager | 25.72 | 1538 | 1.00x |
| 2048 | sdpa eager（默认） | 5.84 | 148 | 4.41x |
| 2048 | sdpa eager 强制 cuDNN | 3.19 | 152 | **8.07x** |
| 2048 | sdpa eager 强制 math | 18.6 | 1210 | 1.38x |

开 `max-autotune`（生产口径）之后：

| seq | std compiled | sdpa compiled | sdpa/std |
|---|---|---|---|
| 256 | 3.24 | 2.93 | 1.11x |
| 1024 | 4.50 | 4.12 | 1.09x |
| 2048 | 8.39 | 6.19 | 1.35x |

带 4D padding mask 且开编译时比值进一步落到 **0.75-1.00x（SDPA 反而更慢）**。

## 二、整模型级（MyLM prod 架构，d=512 / 6 层 / n_heads=4 / L=256 / B=64）

| mask | compile | std ms_step | sdpa ms_step | 加速 | std 显存 | sdpa 显存 |
|---|---|---|---|---|---|---|
| none | eager | 193.5 | 170.3 | 1.14x | 6055 | 4800 |
| none | compiled | 107.3 | 105.5 | **1.02x** | 4246 | 3998 |
| pad | eager | 193.4 | 173.1 | 1.12x | 6102 | 4890 |
| pad | compiled | 105.8 | 106.7 | **0.99x** | 4156 | 3946 |

## 三、"换 SDPA 提速 2-4x、省一半计算"的判定

1. **2-4x 只在 eager 裸算子层成立**（实测 2.1-4.4x，强制 cuDNN 可到 8x）。
   但本仓库训练默认开 `mode="max-autotune"`，编译后整模型只剩 **1.02x**，
   带 padding mask 时 **0.99x**，即净收益为零。收益来源不是 SDPA 本身，
   而是"eager 下没人帮你融合"。
2. **"省一半计算"不是加速的真实原因**。把 SDPA 强制按回 math 后端（和手写一样算满
   L²、不做因果剪枝），它仍只比手写快 0-38%；而 causal 剪枝理论上省 50% FLOPs，
   实测加速曲线却随 L 增长才显现（2.11x→4.41x），说明起作用的是**融合内核不物化
   (B,H,L,L) 分数矩阵**，不是少算上三角。
3. **真正拿不走的收益是显存**：L=2048 时 1538→148 MiB（约 10 倍），
   且 std 在长上下文/大 batch 下会先撞 OOM。capacity 阶段同 batch=256 下
   std 80.9ms vs sdpa 58.2ms，吞吐差 1.39x、显存差 1.41x。
   编译能追平时间，追不平 O(L²) 的显存。
4. **本机没有 FlashAttention 内核，且这是构建期决定的**。`torch_cuda.dll` 里
   `pytorch_flash` / `mha_fwd` / `flash_api` 符号全无，而 `#ifndef USE_FLASH_ATTENTION`
   分支的警告文本在 → 这颗 wheel 编译时 `USE_FLASH_ATTENTION` 未开。
   上游 FA 的硬件门限是 `[sm80, sm121]`，sm_120 **在支持范围内**，
   所以不是卡的问题，是 Windows 版 torch 从来不编 FA2（pytorch/pytorch#108175）。
   另外 FA 的资格链含 `check_for_attn_mask`：**传 attn_mask 时 FA 直接不合格**，
   本仓库只要 batch 含 pad 就会构造 4D mask，等于常态性放弃 FA。

5. **默认 SDPA 走的是 mem-efficient，不是 cuDNN；且这条路被 dropout 挡死了**。
   实测（`bench_cudnn_dropout_tmp.py`，fwd+bwd，B×L=8192，bf16）：
   cuDNN 要求 dropout 概率是 **1/16 的整数倍**，生产 `dropout=0.05` 不在格点上，
   于是强制 cuDNN 直接 `No available kernel`，SDPA 静默退回 mem-efficient。

   | seq | mask | mem-eff @0.05（现状） | cuDNN @1/16=0.0625 | cuDNN @0 |
   |---|---|---|---|---|
   | 1024 | none | 3.881 ms | **1.937 ms** | 1.859 ms |
   | 1024 | pad | 6.101 ms | **3.276 ms** | 2.764 ms |
   | 2048 | none | 5.862 ms | **2.600 ms** | 2.519 ms |
   | 2048 | pad | 11.435 ms | **4.996 ms** | 5.301 ms |

   把 attention dropout 从 0.05 改成 **0.0625（=1/16）** 即可解锁 cuDNN，
   算子级再快 **1.9-2.3 倍**，且 0.0625 与 0 的耗时几乎相同（量化本身不额外花钱）。
   代价是改了一个训练超参（正则强度略增），需训练侧确认能否接受；
   模型级收益未测（注意力只占整步的一部分，会被稀释）。
   注意：本文第二节"强制 cuDNN 再快 15-45%"那句只在 dropout=0 的 core 口径成立，
   生产 dropout=0.05 下不成立，须连同 1/16 格点一起改。

## 四、建议

- 若只为提速：不值得改，现有 max-autotune 管线下差异在噪声内。
- 若目标是更长上下文（>1k）或更大 batch：值得改，动机是显存而非时间。
  改时要按顺序做三件事：**先解决 dropout 格点**（0.05 → 1/16=0.0625，否则 cuDNN
  拒绝、SDPA 静默退 mem-efficient），再 `sdpa_kernel([SDPBackend.CUDNN_ATTENTION])`；
  保留"全屏蔽行输出清零"保护（`torch.where(mask.any(-1,keepdim), y, 0)`，
  实测左 padding 场景 std 与该保护版 sdpa 均 0 个 NaN）；
  以及把 dropout 交给 `dropout_p`（与 std 的 softmax 后 attn_dropout 语义一致）。
- 不要为 FA 折腾：Windows torch 不编 FA2（pytorch/pytorch#108175），第三方
  flash-attn 的 sm_120 在 2026 年仍是源码编译 + nvcc segfault + beta wheel 的状态，
  Python 3.14 无对应预编译包；且 FA 的资格链含 `check_for_attn_mask`，
  吃不了本仓库的 4D padding mask。cuDNN 已是本机可用的最优融合内核。
- 数值等价性：关 TF32 时 fp32 相对误差 out 5.6e-7 / dx 1.4e-6 / dW 2.0e-6；
  bf16 autocast 下 5e-3~9e-3；生产开着 TF32 时 std 与 cuDNN 路径差约 4.5e-4。

复现：

```powershell
$env:PYTHONUTF8='1'
uv run python experiments/attn_impl_bench.py --stages probe --probe-compile
uv run python experiments/attn_impl_bench.py --stages core --seqs 256,512,1024,2048 `
  --masks none,pad --forced-backends cudnn,memeff,math --dump-kernels --tag core3
uv run python experiments/attn_impl_bench.py --stages module,model,capacity --seqs 256 `
  --branches gate --masks none,pad --model-arch prod --capacity-seqs 256 --tag final
uv run python experiments/attn_impl_report.py --png

# dropout 格点与内核资格（一次性探测，bench_*_tmp.py 已被 gitignore）
uv run python bench_cudnn_dropout_tmp.py     # cuDNN 要求 dropout 为 1/16 整数倍
```

FA 符号自查（确认是"没编译"而非"硬件不支持"）：

```python
import mmap, os, torch
p = os.path.join(os.path.dirname(torch.__file__), "lib", "torch_cuda.dll")
m = mmap.mmap(os.open(p, os.O_RDONLY), 0, access=mmap.ACCESS_READ)
for s in (b"Torch was not compiled with flash attention", b"pytorch_flash",
          b"mha_fwd", b"flash_api"):
    print(s.decode(), m.find(s) != -1)
# 本机结果：前一个 True（#ifndef USE_FLASH_ATTENTION 分支的文本在），
# 后三个 False（FA 内核符号全无）
```
