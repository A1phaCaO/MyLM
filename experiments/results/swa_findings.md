# SWA（滑动窗口注意力）实现方式实测结论

机器 RTX 5060 8GB / torch 2.13.0+cu132 / triton 3.7.1。脚本 `experiments/swa_bench.py`，
口径：注意力算子级（q,k/v 就绪）、fwd+bwd 训练态、bf16 autocast、
`B×L = 8192 token` 固定（使不同 L 的耗时可直接横比）、编译用 `max-autotune`
对齐 `pre_train.py`。

## 零、前提：L=256 时 SWA 拿不到时间收益（实测）

W ≥ L 时滑动窗口与全因果完全等价，这是定义上的 no-op。但即使把窗口收到
W < L（64 / 128），在 L=256 上也**测不出时间收益**：

compiled / B=32 / mask=none / 单步 ms：

| W | flex | band-std（编译） | flex 相对 |
|---|---|---|---|
| 64 | 2.047 | 3.177 | 1.55x |
| 128 | 2.058 | 2.051 | 1.00x |
| 256（=全因果） | 2.077 | 1.960 | 0.94x |

flex 三个窗口的耗时几乎一模一样（2.047/2.058/2.077，±1.5%），说明 **L=256 这个尺度上是
kernel launch/调度 bound，不是计算量 bound** —— 少算窗口外那部分分数并不省事。
`mask=pad` 同样：2.096 / 2.046 / 2.042ms，与窗口无关。
显存方面省得也不多：band-std 216-240 MiB → flex 158-164 MiB（1.3-1.5x）。
（顺带可见 band-std 编译耗时在 L=256 上噪声极大：同为"必算满 L²"的实现，
W=64/128/256 测出 3.18/2.05/1.96ms，而 flex 稳定到 ±1.5%。）

**结论：SWA 的时间收益要到 L≥1024（显著是 L≥2048）才出现。**
要用 SWA，先把 `generate_dataset_v3.py` 的序列长度提上去，否则它只是给未来铺路。

## 一、五种实现与实测结果

| 实现 | 做法 | 计算量 | 分数矩阵 |
|---|---|---|---|
| band-std | 手写全量 `q@k.T` + 带状 0/1 mask（等于只把现有 tril 换成 band） | O(L²) | 物化 |
| band-sdpa | SDPA + 稠密带状 attn_mask（默认选择） | O(L²) | 不物化 |
| band-sdpa-cudnn | 同上但强制 cuDNN | O(L²) | 不物化 |
| flex | `torch.compile(flex_attention)` + `create_block_mask`（窗口与 pad 都在 mask_mod 里） | **O(L·W)** | 不物化 |
| flex-sm | 静态 BlockMask（只含窗口，可缓存）+ score_mod 承担 pad | **O(L·W)** | 不物化 |

compiled / mask=pad / W=256（单位 ms，括号内是相对 band-std 的加速与显存节省）：

| L | band-std | band-sdpa | cuDNN | flex | flex-sm |
|---|---|---|---|---|---|
| 1024 | 5.15 / 528 MiB | 6.35（0.81x） | 3.64（1.42x） | **1.76 / 174 MiB（2.92x, 省 3.03x）** | — |
| 2048 | 7.97-14.32 / 790-960 MiB | 10.90（0.73x） | 6.11（1.30x） | **1.73-2.65 / 154 MiB（3.0-8.3x, 省 5-6x）** | 2.71（5.29x） |
| 4096 | 14.98-26.97 / 1506 MiB | 22.31（1.21x） | 12.21（2.21x） | **2.33-3.13 / 158 MiB（6.4-8.6x, 省 9.5-9.8x）** | 2.69（5.58x） |

**两个必须知道的读法限制**：
1. band-std 在 compiled 下**跨 run 不稳定**（同一配置测出 7.97 与 14.32ms，差 1.8 倍），
   因为 max-autotune 每次给物化 L² 的路径挑的 kernel 不同；flex 自身耗时始终稳定在
   1.7-3.1ms。所以加速比只能取"同一次 run 内"的比值，看区间不要看单点。
2. `band-sdpa` 在 L=1024/2048 比**编译后的手写实现还慢**（0.73-0.81x）。
   它只是把分数矩阵藏进内核，窗口对它没有任何意义（内核不知道 mask 是带状的，
   照样算满 L²）。想用 SDPA 做窗口必须显式 pin cuDNN，且仍拿不到窗口的时间收益。

## 二、决定性证据：只有块稀疏实现把"窗口变小"兑换成时间

L=4096、compiled、token 数固定，扫窗口：

| W | flex ms_step | band-std ms_step | flex/band-std |
|---|---|---|---|
| 64 | 1.616 | 16.83 | 10.4x |
| 256 | 2.021 | 16.89 | 8.4x |
| 1024 | 4.878 | 15.37 | 3.2x |
| 4096（=全因果） | 7.401 | 15.37 | 2.08x |

flex 随 W 线性缩放（O(L·W) 的字面体现），band-std 完全不随 W 变化（永远 O(L²)）。
最后一行还给出一个独立结论：**即使窗口开成全因果（无任何稀疏），flex 仍比带状手写快 2.08x**
——这部分是融合内核自身的收益，与 SWA 无关。

另一个角度：L 从 1024 涨到 4096（token 总数不变）时，flex 1.86→3.13ms，
band-std 4.35→26.97ms。窗口的真正价值是把每 token 成本封顶。

## 三、BlockMask 不能每 batch 重建（这是落地时最容易踩的性能坑）

实测 `create_block_mask` 耗时：首次 96-108ms（含 JIT），稳态 **5.9ms**。
若把 pad 写进 mask_mod，BlockMask 就跟 batch 绑定，每 step 重建一次 5.9ms，
而 flex 单步只要 1.7-3.1ms —— **重建比注意力本身还贵两倍**。

正解是 `flex-sm`：BlockMask 只表达静态结构（窗口+因果，只依赖 L/W，模型初始化时
构建一次缓存住），pad 用 `score_mod` 逐元素施加。代价实测为 +0.98ms（L=2048）/
+0.36ms（L=4096），仍比 flex 慢 15-57%，但远低于重建开销，且比 band-std 快 5.3-5.6x。

进一步可把 pad 长度按 `BLOCK_SIZE`（默认 128）分桶，同桶样本复用同一份 BlockMask，
可以把这点开销也省掉。

## 四、正确性

- **数值等价**（fp32 关 TF32，以 band-std 为参考）：flex `out=3.16e-07 dq=6.10e-07`，
  flex-sm 与 flex 逐位相同；band-sdpa `6.32e-07 / 2.35e-06`。bf16 下四者都是 5e-3 量级。
- **pad 等价**（bf16，有效 query 行上的最大绝对差）：右 pad 与左 pad 场景下
  flex、flex-sm 相对 band-std 均为 `1.56e-02`（bf16 精度级），等价成立。
- **NaN 安全**：左 pad 会制造"窗口内没有任何有效 key"的行（第 0 个 token 的窗口里
  全是 pad），实测 flex 与 flex-sm 的 **out 与 dq 均 0 个 NaN**，
  不需要像稠密 mask 那样做"全屏蔽行输出清零"的后处理保护。计时表里所有行 NaN=0。
- **sink（前 4 个 key 永远可见）成本在噪声内**：L=2048 pad 下 sink=4 为 2.129ms、
  sink=0 为 2.645ms（更快，说明差异由时钟状态主导），可以按需免费加。

## 五、推荐实现

优先级排序：

1. **`flex_attention` + 缓存的静态 BlockMask + score_mod 承担 pad**（即 flex-sm 形态）。
   唯一把窗口兑换成时间的方案，也是唯一显存 O(L·W) 的方案。
2. 偶数层 `Attention` 先用，奇数层 `CompressedAttention` 保持原样
   （它已经是 S×S/4 的压缩结构，与窗口部分重叠，同时上会重复计数）。
3. 窗口值：从二、的扫描看，`W=256` 在 L≥2048 时已经拿到 8x 级别的相对收益，
   再往下（W=64）时间只多省 20% 但表示能力明显变窄，不建议；
   W ≥ L/4 是性价比区间。
4. RoPE 不需改动（窗口只限制可见 key 的范围，相对距离语义不变）；
   但推理侧若要把 KV cache 截断到 W，position 必须保留绝对偏移。

落地骨架（未写入 `models.py`）：

```python
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

class WindowAttention(Attention):          # 与 Attention 同参数，只换注意力段
    def __init__(self, args, use_gate=False, window=256, base_init_std=0.02):
        super().__init__(args, use_gate=use_gate, base_init_std=base_init_std)
        self.window = window
        self._flex = torch.compile(flex_attention, dynamic=False)
        self._bm = None                    # 只依赖 (L, window) -> 缓存

    def _block_mask(self, L, device):
        if self._bm is None or self._bm._q_length != L:
            W = self.window
            def causal_window(b, h, qi, ki):
                d = qi - ki
                return (d >= 0) & (d < W)
            self._bm = create_block_mask(causal_window, B=None, H=None,
                                         Q_LEN=L, KV_LEN=L, device=device)
        return self._bm

    def forward(self, x, token_ids=None, mask=None, causal=True):
        ...                                # q/k/v 投影、多头变形、q_norm/k_norm、RoPE
                                             # 全部沿用 models.py:Attention 现状
        if not causal:
            ...                            # 非因果分支保持原实现
        key_valid = (ids != self.pad_id)   # (B,L)，从 token_ids 直接取，
                                           # 不要从 (B,1,L,L) 反解
        def score_mod(score, b, h, qi, ki):
            return torch.where(key_valid[b, ki], score, float("-inf"))
        y = self._flex(q, k, v, score_mod=score_mod,
                       block_mask=self._block_mask(L, x.device))
        ...                                # transpose/合并头 -> gate -> o_proj，同现状
```

`Attention.forward` 目前忽略 `token_ids`，这里正好把它用起来（签名已有，调用方
`MyLMDecoderLayer` 已经在传）。

## 六、注意事项

- `flex_attention` 用 triton，`torch.compile` 前必须 `$env:PYTHONUTF8='1'`（同现有约定）。
- `create_block_mask` 不接受 bound method，`mask_mod`/`score_mod` 必须是普通函数或闭包。
- 首次调用有 JIT 成本（实测 96-108ms/形状），每个 (L, W) 组合一次，可接受。
- 换实现会改数值路径（bf16 下 5e-3 量级差异），loss 曲线不能与旧 run 直接对比。
  ckpt 无风险：不增减任何参数。
- Windows 上 inductor 已实测可用（profile 里能看到 `triton_poi_fused_*`），
  所以 flex 这条路在你的环境里是通的，不需要换 Linux。

复现：

```powershell
$env:PYTHONUTF8='1'
uv run python experiments/swa_bench.py --seqs 1024,2048,4096 --window 256 `
  --masks none,pad --compilations eager,compiled --dump-kernels --tag w256
uv run python experiments/swa_bench.py --seqs 4096 --masks none `
  --impls flex,band-std --windows 64,256,1024,4096 --compilations compiled --tag wsweep
uv run python experiments/swa_bench.py --seqs 2048,4096 --masks pad `
  --impls band-std,flex,flex-sm --compilations compiled --tag scoremod
uv run python experiments/swa_bench.py --seqs 256 --masks none,pad `
  --impls band-std,flex --windows 64,128,256 --compilations eager,compiled --tag l256
uv run python experiments/swa_bench.py --seqs 2048 --masks pad --pad-side left `
  --impls band-std,band-sdpa-cudnn,flex --compilations compiled --tag leftpad
```
