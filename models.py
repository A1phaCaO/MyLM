import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from dataclasses import dataclass
import math


@dataclass
class MyLMArgs:
    d_model: int
    d_inner: int
    n_layers: int
    latent_moe: int
    d_latent: int
    vocab_size: int
    seq_max_len: int
    use_moe: bool = False
    n_heads: int = None
    n_experts: int = 4
    n_experts_per_tok: int = 2
    # FixedCap MoE 参数: 每专家最大容量 = ceil(期望装载 * capacity / 16) * 16
    moe_capacity: float = 1.25   # κ: 丢率/吞吐权衡, 实验最优
    moe_ds_gamma: float = 0.003  # DeepSeek loss-free 均衡步长 (无 aux loss);
    # γ=1e-3 太小 (小模型 120 步仍在崩塌), γ=5e-3 实验 400 步均衡到 max 11~13%
    d_conv: int = 3              # CompressedAttention conv 局部分支 kernel 大小（终版配置用 4）
    conv_bias: bool = True       # 【保留】现 conv 分支固定 bias=True（zero-init），不受此字段影响
    compress_ratio: int = 8      # CompressedAttention 窗口压缩比（终版配置 ca8）；
                                 # ratio 改变网络结构，必须入 config json
    ffn_bias: bool = False
    attn_bias: bool = False
    d_head: int = 64
    dropout: float = 0.1
    base_init_std: float = 0.02  # 基础初始化标准差
    emb_init_std: float = None   # embedding 标准差；None = base_init_std * 0.5（实验最优 embr=0.5）
    pad_id: int = 0              # 用于构造显式 attention padding mask 的 token id；
    # 本项目 SFT 复用 EOS(id=0) 作 pad，pretrain 也以 0 右填充，故默认 0。
    # 取代「pad 行隐藏态全零」的隐式假设（attn_bias 的 o_proj.bias
    # 会让 pad 行残差流在第 1 层后非零，导致原 seq_mask 失效）。


class RMSNorm(torch.nn.Module):
    """【已弃用 2026-09】本仓库早期手写 RMSNorm 实现。

    models.py 内所有模型使用点（Attention/CompressedAttention 的 q/k-norm、
    MyLMDecoderLayer 的 input/post_attention layernorm、MyLM.norm）已全部
    切换为 torch 原生 `nn.RMSNorm(eps=1e-6)`。本类保留原调用方式
    （`RMSNorm(hidden_size)`）仅供 bench_module.py 等新旧实现性能/数值对照，
    模型代码勿再实例化。

    与 `nn.RMSNorm(hidden_size, eps=1e-6)` 数值等价：方差同样在 fp32 中计算
    （LLaMA 约定），归一化结果转回输入 dtype 后再乘 weight；两者的
    state_dict 均只有 `weight`(初始为 1) 一个键，形状一致，旧 checkpoint
    可直接继续加载（eps 不入库，使用点均显式传 1e-6 与旧实现一致）。
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._reset_parameters()

    def _reset_parameters(self, base_init_std=0.02):
        # 初始化权重为1
        with torch.no_grad():
            self.weight.fill_(1.0)

    def forward(self, x):
        input_dtype = x.dtype
        # 方差在 fp32 中计算（LLaMA 约定），避免 bf16 累加的舍入噪声直接进入 rsqrt；
        # 同时不再主动把 fp32 激活降成 bf16——use_amp=False 的对比实验数值口径才与正式训练一致。
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=self.pad,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        return self.conv(input)[:, :, : -self.pad]


class GPT2PositionEmbedding(nn.Module):
    def __init__(self, seq_max_len, d_model):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_max_len, d_model)
        self._reset_parameters()

    def _reset_parameters(self, base_init_std=0.02):
        nn.init.normal_(self.pos_emb.weight, std=base_init_std)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        assert (
            seq_len <= self.pos_emb.num_embeddings
            # 检查序列长度是否超限
        ), f"序列长度 {seq_len} 超过预设最大值 {self.pos_emb.num_embeddings}"
        # 生成位置编码并相加
        pos = torch.arange(seq_len).to(x.device)  # (seq_len,)
        pos_emb = self.pos_emb(pos)  # (seq_len, d_model)
        pos_emb = pos_emb.unsqueeze(0)  # (1, seq_len, d_model)
        return x + pos_emb  # 广播到 (batch_size, seq_len, d_model)


class MyPositionEmbedding(nn.Module):
    def __init__(self, d_model, d_out, d_inner=None):
        super().__init__()
        if d_inner is None:
            d_inner = d_model // 16
        self.d_inner = d_inner
        self.d_model = d_model
        self.gru = nn.GRU(d_inner, d_inner, bias=False, batch_first=True)
        self.pos_conv = nn.Conv1d(d_model, d_inner, 1)
        # self.proj_conv = nn.AdaptiveAvgPool1d(d_model - d_inner)
        self.proj_conv = nn.Conv1d(d_model, d_model - d_inner, 1)
        self.linear = nn.Linear(d_model, d_out, bias=False)
        # self.up_proj = nn.Linear(d_inner, d_model, bias=False)

    def _reset_parameters(self, base_init_std=0.02):
        torch.nn.init.normal_(self.linear.weight, std=base_init_std)
        # 初始化其他层的权重
        if hasattr(self, "pos_conv") and self.pos_conv.weight is not None:
            torch.nn.init.normal_(self.pos_conv.weight, std=base_init_std)
        if hasattr(self, "proj_conv") and self.proj_conv.weight is not None:
            torch.nn.init.normal_(self.proj_conv.weight, std=base_init_std)
        # GRU层的初始化
        if hasattr(self, "gru"):
            for name, param in self.gru.named_parameters():
                if "weight" in name:
                    torch.nn.init.normal_(param, std=base_init_std)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)

    def forward(self, x):
        res = x
        x = x.transpose(1, 2)
        x_proj = self.proj_conv(x).transpose(1, 2)  # 下采样非时间特征
        pos = self.pos_conv(x).transpose(1, 2)  # 下采样时间特征
        pos, _ = self.gru(pos)  # 时间特征RNN
        x = self.linear(torch.cat([pos, x_proj], dim=-1))  # 维度融合

        return x + res


class ALiBi(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.Tensor(self._get_slopes(num_heads))
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def forward(self, seq_len, batch_size, device):
        # 生成相对位置矩阵
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = torch.abs(
            context_position - memory_position
        )  # (seq_len, seq_len)

        # 为每个头生成偏置矩阵 (num_heads, seq_len, seq_len)
        bias = relative_position[None, ...] * self.slopes[:, None, None]
        bias = -bias  # ALiBi 的负偏置

        # 扩展为 (batch_size * num_heads, seq_len, seq_len)
        bias = bias.repeat(batch_size, 1, 1)  # 直接复制到每个样本
        return bias


class Attention(nn.Module):
    """统一注意力机制，通过use_gate参数切换门控/标准模式"""

    def __init__(self, args: MyLMArgs, use_gate=False, base_init_std=0.02):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads or (args.d_model // args.d_head)
        self.d_head = args.d_head
        self.seq_max_len = args.seq_max_len
        self.use_gate = use_gate
        self.args = args

        # 注意力投影层
        self.q_proj = nn.Linear(
            args.d_model, args.d_model, bias=args.attn_bias)
        self.k_proj = nn.Linear(
            args.d_model, args.d_model, bias=args.attn_bias)
        self.v_proj = nn.Linear(
            args.d_model, args.d_model, bias=args.attn_bias)
        self.o_proj = nn.Linear(
            args.d_model, args.d_model, bias=args.attn_bias)

        # QK-Norm 逐头沿 head_dim 归一化（不是 d_model），且必须在
        # view 出多头之后的 (batch, heads, seq, d_head) 张量上使用
        self.q_norm = nn.RMSNorm(args.d_head, eps=1e-6)
        self.k_norm = nn.RMSNorm(args.d_head, eps=1e-6)

        # 门控层（仅在使用门控时创建）
        if use_gate:
            self.gate = nn.Linear(args.d_model, args.d_model, bias=False)

        # RoPE位置编码缓存
        self.register_buffer(
            "cos_cached", torch.zeros(1, 1, args.seq_max_len, args.d_head)
        )
        self.register_buffer(
            "sin_cached", torch.zeros(1, 1, args.seq_max_len, args.d_head)
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        # 初始化RoPE
        self._init_rope()
        self._reset_parameters(base_init_std=base_init_std)

    def _reset_parameters(self, base_init_std=0.02, residual_scale=None):
        if residual_scale is None:
            residual_scale = 1.0 / math.sqrt(2 * self.args.n_layers)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            torch.nn.init.normal_(proj.weight, std=base_init_std)
            if proj.bias is not None:
                torch.nn.init.zeros_(proj.bias)
        self.o_proj.weight.data.mul_(residual_scale)
        if self.use_gate and hasattr(self, 'gate'):
            torch.nn.init.normal_(self.gate.weight, std=base_init_std)

    def _init_rope(self):
        """初始化RoPE位置编码"""
        d_head_half = self.d_head // 2
        # 创建频率数组，长度为d_head_half
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, d_head_half,
                      dtype=torch.float) / d_head_half)
        )

        t = torch.arange(self.seq_max_len, dtype=torch.float)
        # 计算位置频率
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # 扩展到完整维度并添加批次和头数维度
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d_head)
        # 使用register_buffer更新缓存，而不是直接赋值
        self.register_buffer(
            "cos_cached", emb.cos().unsqueeze(0).unsqueeze(0)
        )  # (1, 1, seq_len, d_head)
        self.register_buffer(
            "sin_cached", emb.sin().unsqueeze(0).unsqueeze(0)
        )  # (1, 1, seq_len, d_head)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """旋转一半维度"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ):
        """应用RoPE位置编码"""
        # 调整cos和sin的维度以匹配q和k的序列长度
        cos = cos[:, :, : q.size(2), :]  # (1, 1, seq_len, d_head)
        sin = sin[:, :, : q.size(2), :]  # (1, 1, seq_len, d_head)

        # 将q和k分割为两半用于旋转操作
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, x: torch.Tensor, token_ids=None, mask=None, causal=True) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 计算QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 计算门控（仅在使用门控时）
        if self.use_gate:
            gate = F.sigmoid(self.gate(x))

        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (batch, heads, seq, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (batch, heads, seq, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (batch, heads, seq, head_dim)

        # QK-Norm：先归一化再施加 RoPE（主流实现口径）。RoPE 保范数，
        # 但归一化放在旋转后会让可学习权重与位置旋转交叉作用，改变语义。
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 应用RoPE位置编码
        cos = self.cos_cached[:, :, :seq_len, :]  # (1, 1, seq_len, d_head)
        sin = self.sin_cached[:, :, :seq_len, :]  # (1, 1, seq_len, d_head)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # 计算注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        if causal:
            # only causal
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)).view(
                1, 1, seq_len, seq_len
            )
        else:
            causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).view(
                1, 1, seq_len, seq_len
            )

        # 如果提供了外部mask，则将其与因果掩码合并
        if mask is not None:
            # 简单检查mask维度，不符合要求直接抛出异常
            if mask.dim() != 4:
                raise ValueError(
                    f"Mask must be 4-dimensional, got {mask.dim()} dimensions")
            if mask.shape != (batch_size, 1, seq_len, seq_len):
                raise ValueError(
                    f"Mask shape must be {(batch_size, 1, seq_len, seq_len)}, got {mask.shape}")
            # 合并因果掩码和外部mask
            combined_mask = causal_mask & mask
        else:
            # 只使用因果掩码
            combined_mask = causal_mask

        att = att.masked_fill(combined_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        # 全被屏蔽的 query 行（query 本身为 pad）softmax 会产生 NaN，
        # 把 mask 位置重新填 0，保证该行输出为 0 而非 NaN 继续传播
        att = att.masked_fill(combined_mask == 0, 0.0)
        att = self.attn_dropout(att)

        # 应用注意力权重
        if self.use_gate:
            # 门控注意力：不对V激活，而是在输出后应用门控
            y = att @ v  # (batch, heads, seq, head_dim)
            y = (
                y.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, self.d_model)
            )  # 重新组合多头
            y = y * gate  # 应用门控
        else:
            # 标准注意力：V 不做激活（旧版为 att @ F.sigmoid(v)，随 QK-Norm
            # 引入一并移除），直接对 v 加权求和
            y = att @ v  # (batch, heads, seq, head_dim)
            y = (
                y.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, self.d_model)
            )  # 重新组合多头
        # 输出投影
        y = self.resid_dropout(self.o_proj(y))
        return y


class CompressedAttention(nn.Module):
    """仿DeepSeek CSA的压缩部分实现的注意力机制，并通过门控卷积分支增强局部感受野。"""

    def __init__(self, args: MyLMArgs, compress_ratio=None, base_init_std=0.02):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads or (args.d_model // args.d_head)
        self.d_head = args.d_head
        self.seq_max_len = args.seq_max_len
        # 压缩比从 MyLMArgs 读取（配置单一事实源）；显式传参仅供
        # bench/消融脚本（如 ca2/ca4 对照）
        self.compress_ratio = (args.compress_ratio if compress_ratio is None
                               else compress_ratio)
        self.args = args
        # QK-Norm 逐头沿 head_dim 归一化（不是 d_model），且必须在
        # view 出多头之后的 (batch, heads, seq, d_head) 张量上使用
        self.q_norm = nn.RMSNorm(args.d_head, eps=1e-6)
        self.k_norm = nn.RMSNorm(args.d_head, eps=1e-6)

        # 注意力投影层
        self.q_proj = nn.Linear(
            args.d_model, args.d_model, bias=args.attn_bias)
        self.k_proj = nn.Linear(
            args.d_model, args.d_model, bias=args.attn_bias)
        self.v_proj = nn.Linear(
            args.d_model, args.d_model, bias=args.attn_bias)
        self.o_proj = nn.Linear(
            args.d_model, args.d_model, bias=args.attn_bias)
        self.compress_gate = nn.Linear(args.d_head, 1, bias=False)
        self.gate_proj = nn.Linear(args.d_model, args.d_model, bias=False)
        self.conv_mix_alpha = nn.Parameter(torch.zeros(1))
        self.conv = CausalConv1d(
            in_channels=args.d_model,
            out_channels=args.d_model,
            kernel_size=args.d_conv,
            groups=args.d_model,
            bias=True,
        )

        # RoPE位置编码缓存
        self.register_buffer(
            "cos_cached", torch.zeros(1, 1, args.seq_max_len, args.d_head)
        )
        self.register_buffer(
            "sin_cached", torch.zeros(1, 1, args.seq_max_len, args.d_head)
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        # 初始化RoPE
        self._init_rope()
        self._reset_parameters(base_init_std=base_init_std)

    def _reset_parameters(self, base_init_std=0.02, residual_scale=None):
        if residual_scale is None:
            residual_scale = 1.0 / math.sqrt(2 * self.args.n_layers)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            torch.nn.init.normal_(proj.weight, std=base_init_std)
            if proj.bias is not None:
                torch.nn.init.zeros_(proj.bias)
        torch.nn.init.normal_(self.compress_gate.weight, std=base_init_std)
        torch.nn.init.normal_(self.gate_proj.weight, std=base_init_std)
        self.o_proj.weight.data.mul_(residual_scale)
        # conv 权重保留 Conv1d 默认 kaiming（depthwise 下 std≈1/√k，分支
        # 初始即有效）；bias 归零：Conv1d 默认 bias init 为 U(±1/√k)，量级
        # ~0.5 远超初始残差流尺度（~0.02），且 alpha 起步 0.5，不归零会让
        # 卷积分支早期被随机常数主导（对齐仓库 bias 一律 zero-init 约定）
        if self.conv.conv.bias is not None:
            torch.nn.init.zeros_(self.conv.conv.bias)

    def _init_rope(self):
        """初始化RoPE位置编码"""
        d_head_half = self.d_head // 2
        # 创建频率数组，长度为d_head_half
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, d_head_half,
                                   dtype=torch.float) / d_head_half)
        )

        t = torch.arange(self.seq_max_len, dtype=torch.float)
        # 计算位置频率
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # 扩展到完整维度并添加批次和头数维度
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d_head)
        # 使用register_buffer更新缓存，而不是直接赋值
        self.register_buffer(
            "cos_cached", emb.cos().unsqueeze(0).unsqueeze(0)
        )  # (1, 1, seq_len, d_head)
        self.register_buffer(
            "sin_cached", emb.sin().unsqueeze(0).unsqueeze(0)
        )  # (1, 1, seq_len, d_head)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """旋转一半维度"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """对单个张量应用RoPE（q与压缩后k的序列长度不同，需分别调用）"""
        cos = cos[:, :, : x.size(2), :]  # (1, 1, len, d_head)
        sin = sin[:, :, : x.size(2), :]
        return (x * cos) + (self._rotate_half(x) * sin)

    def _compress_kv(self, k: torch.Tensor, v: torch.Tensor):
        """压缩K和V：每 ratio 个相邻token经gate加权融合为1个meta-KV"""
        batch_size, n_heads, seq_len, d_head = k.size()
        ratio = self.compress_ratio

        n_pad = -seq_len % ratio
        if n_pad != 0:
            # 在 seq 维（dim=2）补齐到 ratio 的整数倍
            k = F.pad(k, (0, 0, 0, n_pad))
            v = F.pad(v, (0, 0, 0, n_pad))
        len_pad = seq_len + n_pad
        len_compressed = len_pad // ratio

        # k/v 是 transpose(1,2) 后的非连续视图，view 会抛 stride 错误；
        # 且这里是把 seq 维拆成 (窗口数, 窗口内token)，窗口维必须紧跟 n_heads
        k_windows = k.reshape(
            batch_size, n_heads, len_compressed, ratio, d_head)
        v_windows = v.reshape(
            batch_size, n_heads, len_compressed, ratio, d_head)

        scores = self.compress_gate(k_windows).squeeze(-1)  # (b, h, lc, ratio)
        if n_pad != 0:
            valid = torch.arange(ratio, device=k.device) < (ratio - n_pad)
            win_mask = torch.ones(len_compressed, ratio,
                                  dtype=torch.bool, device=k.device)
            win_mask[-1] = valid                     # 只有最后一个窗口需要 mask
            scores = scores.masked_fill(
                ~win_mask[None, None], float("-inf")
            )
        scores = F.softmax(scores, dim=-1)
        k_compressed = (k_windows * scores.unsqueeze(-1)).sum(-2)
        v_compressed = (v_windows * scores.unsqueeze(-1)).sum(-2)

        # (b, h, len_compressed, d_head)
        return k_compressed, v_compressed, len_compressed

    def forward(self, x: torch.Tensor, token_ids=None, mask=None, causal=True) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 外部token级mask校验与token有效性提取（mask: (b,1,s,s) bool，True=有效）
        if mask is not None:
            # 简单检查mask维度，不符合要求直接抛出异常
            if mask.dim() != 4:
                raise ValueError(
                    f"Mask must be 4-dimensional, got {mask.dim()} dimensions")
            if mask.shape != (batch_size, 1, seq_len, seq_len):
                raise ValueError(
                    f"Mask shape must be {(batch_size, 1, seq_len, seq_len)}, got {mask.shape}")
            key_valid = mask.amax(dim=2).unsqueeze(-1)      # (b, 1, seq, 1)
            query_valid = mask.amax(dim=-1) \
                .squeeze(1).unsqueeze(-1)                   # (b, seq, 1)
        else:
            key_valid = None
            query_valid = None

        # 卷积分支计算：因果卷积感受野覆盖 t-k+1..t，左padding时 pad 行内容
        # 会经 conv 泄漏进后续有效位（padding 隔离守卫实测泄漏 1.43）。先把
        # pad 行的卷积输入清零，pad 内容对有效位输出只剩常数(bias)贡献，不泄漏。
        x_conv_in = x if query_valid is None else x * query_valid.to(x.dtype)
        conv_res = self.conv(x_conv_in.transpose(1, 2)).transpose(1, 2)

        # 计算QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (batch, heads, seq, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (batch, heads, seq, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (batch, heads, seq, head_dim)

        # QK-Norm：逐头归一化提前到 pad 清零/窗口压缩/RoPE 之前。
        # nn.RMSNorm 对零向量仍输出零，pad 位清零语义不变；压缩窗口内
        # 融合的是同尺度（单位 RMS）的 k 向量。
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 先把对所有query都无效的 key位k/v清零，避免pad隐藏态混入窗口压缩
        # 结果（nn.RMSNorm 对零向量仍输出零，pad 位清零语义不变）
        if key_valid is not None:
            k = k * key_valid.to(k.dtype)
            v = v * key_valid.to(v.dtype)

        k, v, len_compressed = self._compress_kv(k, v)

        # 应用RoPE位置编码：q用原位置；k每窗口一个，用窗口起始位置
        # (0, ratio, 2*ratio, ...) 作为landmark位置
        cos = self.cos_cached[:, :, :seq_len, :]  # (1, 1, seq_len, d_head)
        sin = self.sin_cached[:, :, :seq_len, :]  # (1, 1, seq_len, d_head)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos[:, :, ::self.compress_ratio],
                       sin[:, :, ::self.compress_ratio])

        # 计算注意力分数：key维已压缩为 len_compressed
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        if causal:
            # 窗口必须整体不在未来才可见: 以"窗口最后一个 token 下标 <= query 位置"
            # 判定。旧实现按窗口起始 (w*ratio <= t) 判定, 偶数位 query 会看到同窗口
            # 内的 t+1, 而 t 位置的预测目标恰是 token t+1 —— 目标经压缩 meta-KV
            # 直接泄漏进注意力, loss 可假降到抄答案水平（探针实测已确认）。
            # 代价: 偶数位 query 的自身窗口 (含奇数位 t+1) 整窗被屏蔽, 看不到自己,
            # 自身信息退化为仅走残差流; 若这成为质量瓶颈, 应补 NSA/CSA 式局部
            # 细分支（小滑窗全分辨率）而非放宽掩码。
            key_pos = torch.arange(
                len_compressed, device=x.device) * self.compress_ratio
            key_end = (key_pos + self.compress_ratio - 1).clamp(
                max=seq_len - 1)   # 尾窗被 pad 截短, 实际末 token 到 seq_len-1 为止
            causal_mask = (key_end[None, :] <= torch.arange(
                seq_len, device=x.device)[:, None]
            ).view(1, 1, seq_len, len_compressed)
        else:
            causal_mask = torch.ones(
                seq_len, len_compressed, device=x.device, dtype=torch.bool
            ).view(1, 1, seq_len, len_compressed)

        # 如果提供了外部mask，则将其与因果掩码合并：窗口可见性取「窗口内至少
        # 一个 key 有效」。不再按窗口起始位采样——左pad下起始位常是pad，会把
        # 含有效 token 的混合窗口整窗误屏蔽；而 pad 位 k/v 已清零，放行混合
        # 窗口不会引入 pad 内容。pad query 行整行无效，输出保持全零。
        if mask is not None:
            ratio = self.compress_ratio
            win_valid = key_valid.squeeze(-1)              # (b, 1, seq)
            n_pad = -seq_len % ratio
            if n_pad != 0:
                win_valid = F.pad(win_valid, (0, n_pad), value=False)
            win_valid = win_valid.view(
                batch_size, 1, len_compressed, ratio).amax(-1).unsqueeze(2)
            q_valid4 = query_valid.unsqueeze(1)            # (b, 1, seq, 1)
            combined_mask = causal_mask & win_valid & q_valid4
        else:
            # 只使用因果掩码
            combined_mask = causal_mask

        att = att.masked_fill(combined_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        # 全被屏蔽的 query 行（query 本身为 pad）softmax 会产生 NaN，
        # 把 mask 位置重新填 0，保证该行输出为 0 而非 NaN 继续传播
        att = att.masked_fill(combined_mask == 0, 0.0)
        att = self.attn_dropout(att)

        y = att @ v  # (batch, heads, seq, head_dim)
        y = (
            y.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.d_model)
        )  # 重新组合多头
        alpha = F.sigmoid(self.conv_mix_alpha)
        # 卷积分支与注意力输出按 alpha 混合后做 sigmoid 门控
        y = (y*(1-alpha) + conv_res*alpha) * F.sigmoid(self.gate_proj(x))
        # 输出投影
        y = self.resid_dropout(self.o_proj(y))
        return y


class FFN(nn.Module):
    """LLaMA MLP层"""

    def __init__(self, args: MyLMArgs, base_init_std=0.02):
        super().__init__()
        self.args = args
        # 用局部变量而非改写共享 args，避免副作用污染 args.d_latent
        d_in = args.d_latent if args.latent_moe else args.d_model
        self.gate_proj = nn.Linear(d_in, args.d_inner, bias=False)
        self.up_proj = nn.Linear(d_in, args.d_inner, bias=False)
        self.down_proj = nn.Linear(args.d_inner, d_in, bias=False)
        self._reset_parameters(base_init_std=base_init_std)

    def _reset_parameters(self, base_init_std=0.02, residual_scale=None):
        if residual_scale is None:
            residual_scale = 1.0 / math.sqrt(2 * self.args.n_layers)
        torch.nn.init.normal_(self.gate_proj.weight, std=base_init_std)
        torch.nn.init.normal_(self.up_proj.weight, std=base_init_std)
        torch.nn.init.normal_(self.down_proj.weight, std=base_init_std)
        self.down_proj.weight.data.mul_(residual_scale)

    def forward(self, x: torch.Tensor, token_ids=None) -> torch.Tensor:
        y = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        return y


class MoEFFN(nn.Module):
    """稀疏 MoE 混合专家层: FixedCap 固定容量 + DeepSeek loss-free 负载均衡。

    设计结论 (debug_moe.py 实测, 2026-08-08):
    - 数学上与逐专家 mask 版 (legacy models_260808.py) 完全一致 (k=1 误差 0)
    - 固定容量 M = ceil(期望每专家装载 * kappa / 16) * 16, 0 次 CPU 同步
       (分桶/负载统计全在 GPU 上完成, 实测比逐专家循环快 1.6x)
    - 超容量 token 的 prob 置 0 → index_add 归入 padding 槽, 不污染结果
    - kappa=1.25 为横评最优: 训练丢率 1~7%, loss 稳定不劣化
    - DS bias: bias 加进 topk 输入, 权重仍出自原始 logits; 步长 moe_ds_gamma
      自动按负载差更新, 完全无 aux loss (aux=5e-4 实测拖慢 11x, 弃用)
    - M_cap 首个 forward 时定下, 之后不再变化
    """

    def __init__(self, args: MyLMArgs, base_init_std=0.02, kappa=None, ds_gamma=None):
        super().__init__()
        self.args = args
        self.kappa = args.moe_capacity if kappa is None else kappa
        self.ds_gamma = args.moe_ds_gamma if ds_gamma is None else ds_gamma
        self.N = args.n_experts
        self.Kk = args.n_experts_per_tok
        self.d = args.d_latent if args.latent_moe else args.d_model
        self.MCap = None          # 首个 forward 定下: M = ceil(期望装载 * κ / 16) * 16
        # 预分配 0 维 buffer 存丢率：compile+CUDAGraph 下 forward 内直接赋值
        # 的 tensor 是图输出，图外（utils._hook 采集端）读取会报
        # "CUDAGraphs output has been overwritten"；copy_ 写入持久 buffer 则安全。
        # persistent=False：不参与 state_dict（纯运行时统计）
        self.register_buffer("last_drop", torch.zeros(()), persistent=False)
        self.router = nn.Linear(args.d_model, args.n_experts, bias=False)
        if args.latent_moe:
            self.latent_down = nn.Linear(
                args.d_model, args.d_latent, bias=False)
            self.latent_up = nn.Linear(args.d_latent, args.d_model, bias=False)
        # 专家权重: [N, d_in, d_inner] ×3, batched bmm 一次算完全部专家
        w = torch.empty(self.N, self.d, args.d_inner)
        self.w_gate = nn.Parameter(w)
        self.w_up = nn.Parameter(w.clone())
        self.w_down = nn.Parameter(
            torch.empty(self.N, args.d_inner, self.d)
        )
        if self.ds_gamma > 0:
            self.register_buffer("expert_bias", torch.zeros(args.n_experts))
        self._reset_parameters(base_init_std)

    def _reset_parameters(self, base_init_std=0.02, residual_scale=None):
        if residual_scale is None:
            residual_scale = 1.0 / math.sqrt(2 * self.args.n_layers)
        torch.nn.init.normal_(self.router.weight, std=base_init_std*0.5)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)
        # latent 投影与专家保持一致的初始化分布
        if self.args.latent_moe:
            torch.nn.init.normal_(self.latent_down.weight, std=base_init_std)
            torch.nn.init.normal_(self.latent_up.weight, std=base_init_std)
            self.latent_up.weight.data.mul_(residual_scale)
        # 3D 专家权重共享同一分布 (原版为逐专家 FFN 独立初始化, 分布一致)
        torch.nn.init.normal_(self.w_gate, std=base_init_std)
        torch.nn.init.normal_(self.w_up, std=base_init_std)
        torch.nn.init.normal_(self.w_down, std=base_init_std)
        self.w_down.data.mul_(residual_scale)

    def forward(self, x, token_ids=None):
        # 路由块: sqrt(softplus(.)) 单调, 直接对 router logits 做 topk，
        # 省去对全量 [B,S,n_experts] 的 softplus+sqrt
        router_logits = self.router(x)  # [B, S, N]
        # DeepSeek loss-free 均衡: bias 只影响 topk 选择, 权重引用原始 logits
        logits_sel = router_logits + self.expert_bias if self.ds_gamma > 0 else router_logits
        top_k_logits, top_k_idx = torch.topk(logits_sel, self.Kk, dim=-1)
        top_k_probs = torch.sqrt(F.softplus(
            torch.gather(router_logits, -1, top_k_idx)))
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # DeepSeek loss-free 负载均衡更新: 仅在训练时按负载差符号调整 bias
        # 计数用 index_add_ 而非 torch.bincount: bincount 带 kwarg 在 dynamo
        # 中不识别会 graph break, 恢复子图在训练(aot joint)下触发 torch 2.13
        # _replay_alias 的 "shape '[]' is invalid" 崩溃
        if self.ds_gamma > 0 and self.training:
            ones = torch.ones_like(
                top_k_idx.reshape(-1), dtype=torch.float32)
            load = torch.zeros(
                self.N, device=ones.device, dtype=torch.float32)
            load.index_add_(0, top_k_idx.reshape(-1), ones)
            self.expert_bias.data.add_(
                -self.ds_gamma * torch.sign(load - load.mean()))

        N, K, d = self.N, self.Kk, self.d
        B, S, _ = x.shape
        T = B * S
        if self.training:
            # 训练: 容量由首个训练 batch 定下后固定（不随 step 变化，保持
            # torch.compile 图稳定）。训练 batch 很大，容量充足、几乎不丢 token。
            if self.MCap is None:
                self.MCap = max(1, math.ceil(
                    (T * K / N) * self.kappa / 16) * 16)
            M = self.MCap
        else:
            # 推理/验证: 优先复用训练 MCap——validate 的 batch 与训练同形状，
            # 复用可避免 eval 容量小于实际装载导致大量 token 被丢弃、val 失真，
            # 且与训练图同形状、不会触发重编译。
            # 纯推理（如 run_model，MCap 未被训练定下）回退到「基于 seq_max_len」
            # 的固定容量：与 T 无关、逐 step 形状恒定，对任意长度单序列
            # （期望装载 = seq_max_len*K/N ≤ M）都充足、不丢 token。
            # 关键修复——旧版 eval 分支无条件覆盖 self.MCap，任何 eval 前向
            # （validate/generate_test）都把训练容量污染成 seq_max_len 容量，
            # 之后训练分支 if self.MCap is None 永假，MoE 训练静默丢弃大量
            # token（SFT 配置下约 94.5%），且批量验证自身同样丢弃大量 token。
            M = self.MCap
            if M is None:
                M = max(
                    1, math.ceil((self.args.seq_max_len * K / N)
                                 * self.kappa / 16) * 16
                )

        # Latent 投影
        if self.args.latent_moe:
            x = self.latent_down(x)
        # (token, expert) 对按行排列, 每个 token 展平为 K 行
        flat_x = x.reshape(T, 1, d).expand(T, K, d).reshape(T * K, d)

        # (token, expert) 对扁平化并按专家归桶:
        # 每对 = (槽位-T=序号, 专家), 组内编号 = 该专家内第几个 token
        pair_exp = top_k_idx.reshape(-1)           # [T*K]
        pair_prob = top_k_probs.reshape(-1)        # [T*K]
        pair_idx = torch.arange(T * K, device=x.device)  # 展平对序号
        order = torch.argsort(pair_exp, stable=True)
        es = pair_exp[order]
        # 与上一处 bincount 同理, 用 index_add_ 计数 (保持 int64 以维持下游
        # target = es.long()*M + pos_safe 的整型运算链)
        ones = torch.ones_like(es, dtype=torch.long)
        counts = torch.zeros(N, device=es.device, dtype=torch.long)
        counts.index_add_(0, es, ones)
        cs = torch.cumsum(counts, dim=0)
        starts = cs - counts
        pos = pair_idx - starts.gather(0, es)      # 组内序号
        keep = pos < M
        pos_safe = torch.clamp(pos, max=M - 1)
        target = es.long() * M + pos_safe          # 块内全局槽位 [N*M]
        indirect = order.to(torch.long)
        total = N * M

        # 输入进桶: 超容量 token 的权重置 0 后 index_add 到 padding 槽
        block = torch.zeros(total, d, device=x.device, dtype=flat_x.dtype)
        block.index_add_(
            0, target, flat_x[indirect] * keep.to(flat_x.dtype).unsqueeze(-1))
        block_p = torch.zeros(total, device=x.device, dtype=flat_x.dtype)
        block_p.index_add_(0, target, pair_prob[indirect].to(
            flat_x.dtype) * keep.to(flat_x.dtype))
        block = block.view(N, M, d)

        # 专家前向: batched bmm 一次算完全部 N 个专家
        h = F.silu(torch.bmm(block, self.w_gate)) * torch.bmm(block, self.w_up)
        out = torch.bmm(h, self.w_down).view(total, d)
        out = out * block_p.unsqueeze(-1)

        # 槽位结果按 token 累加 (index_add 自动处理 K>1 同 token 多专家)
        token_of_slot = torch.zeros(total, device=x.device, dtype=torch.long)
        token_of_slot.index_add_(
            0, target, (pair_idx // K)[indirect] * keep.to(torch.long))
        y = torch.zeros(T, d, device=x.device, dtype=out.dtype)
        y.index_add_(0, token_of_slot, out)

        # 存 0 维 tensor 而非 .item(): torch.compile 下 .item() 会 graph break
        # (采集端 utils._hook 已用 float() 转 Python 标量)
        self.last_drop.copy_((1 - keep.to(torch.float32).mean()).detach())
        return self.latent_up(y.view(B, S, d)) if self.args.latent_moe else y.view(B, S, d)


class MyLMDecoderLayer(nn.Module):
    """
    混合Transformer块
    门控注意力 : v激活注意力 = 1:1
    """

    def __init__(self, args: MyLMArgs, layer_idx=0, base_init_std=0.02):
        super().__init__()
        self.args = args
        if layer_idx % 2 == 0:
            self.attn = Attention(args, use_gate=True,
                                  base_init_std=base_init_std)
        else:
            # compress_ratio 由 MyLMArgs 注入（终版配置 ca8=8）
            self.attn = CompressedAttention(args, base_init_std=base_init_std)

        self.mlp = (
            MoEFFN(args, base_init_std=base_init_std)
            if args.use_moe
            else FFN(args, base_init_std=base_init_std)
        )
        self.input_layernorm = nn.RMSNorm(args.d_model, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(args.d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, token_ids=None, padding_mask=None) -> torch.Tensor:
        # 注意力部分
        residual = x
        x = self.input_layernorm(x)
        # padding_mask: (batch_size, seq_len) bool，True=真实 token；由 MyLM.forward
        # 从输入 token id 用 (id != pad_id) 显式构造后逐层传入，取代原先
        # seq_mask = (x.sum(-1) != 0) 的「pad 行隐藏态全零」隐式假设。
        # 该假设在 attn_bias=True 时因 o_proj.bias 可训练而在第 1 层起失效，
        # 导致左 padding 下 pad 行被误判为有效 token、深度参与全部层注意力
        # （即 code review 中的 S2 污染）。显式 mask 与隐藏态是否为零无关，
        # 左/右 padding 均正确；padding_mask=None 时仅用 causal mask（推理无 padding）。
        x = self.attn(x, token_ids=token_ids, mask=padding_mask)
        x = residual + x

        # MLP部分
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, token_ids=token_ids)
        x = residual + x

        return x


def exclude_moe_from_compile(model: nn.Module) -> int:
    """标记模型中所有 MoEFFN 层，使其在 torch.compile 时以 eager 模式执行。

    【已过时】旧版逐专家循环（mask/nonzero）数据依赖 Python 循环，compile 反而更慢；
    现版 FixedCap 分桶已改为纯 tensor 操作（bincount/index_add/bmm），
    实测 inductor 编译整模型（MoE 不排除）38.6ms vs 排除 44.2ms，包含 MoE 更快。
    保留此函数仅为兼容旧配置，新训练建议 exclude_moe_from_compile=False。
    返回被排除的 MoE 层数。
    """
    n_disabled = 0
    for module in model.modules():
        if isinstance(module, MoEFFN):
            module.forward = torch.compiler.disable(
                module.forward, recursive=True,
                reason="遗留: 旧逐专家循环时代结论, 新 FixedCap 已可编译",
            )
            n_disabled += 1
    return n_disabled


class MyLM(nn.Module):
    """
    简化版LLaMA架构实现
    兼容MyLMArgs配置参数
    """

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.args = args
        base_init_std = args.base_init_std
        # 词嵌入层
        self.token_embedding = nn.Embedding(args.vocab_size, args.d_model)

        # Transformer块
        self.blocks = nn.ModuleList(
            [
                MyLMDecoderLayer(args, layer_idx, base_init_std=base_init_std)
                for layer_idx in range(args.n_layers)
            ]
        )

        # 输出层
        self.norm = nn.RMSNorm(args.d_model, eps=1e-6)
        self.head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self._reset_parameters(base_init_std=args.base_init_std)

    def _reset_parameters(self, base_init_std=0.02):
        emb_std = (
            self.args.emb_init_std
            if self.args.emb_init_std is not None
            else base_init_std * 0.5
        )
        torch.nn.init.normal_(self.token_embedding.weight, std=emb_std)
        torch.nn.init.normal_(self.head.weight, std=base_init_std)
        # for module in self.modules():
        # if isinstance(module, nn.Linear):
        #     if module.bias is not None:
        #         module.bias.data.zero_()
        # if isinstance(module, nn.Embedding):
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, token_ids=None, padding_mask=None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量（token id），形状为(batch_size, seq_len)
            token_ids: token ID，保留位（当前未使用），与 x 相同
            padding_mask: 可选，(batch_size, seq_len) 的 bool 张量，True=真实 token。
                由调用方用 (x != pad_id) 构造后传入；为 None 时仅使用 causal mask
                （推理无 padding 场景）。该显式 mask 取代原先依赖「pad 行隐藏态全零」
                的隐式 padding 屏蔽，使左/右 padding 下的注意力屏蔽都正确无误。

        Returns:
            输出张量，形状为(batch_size, seq_len, vocab_size)
        """
        # 显式 attention padding mask：同时屏蔽 key 与 query 维的 pad 位置。
        # 只屏蔽 key 维时，左 padding 下 pad 位置的 query 行全部 key 被 -inf 屏蔽，
        # softmax 产生 NaN 并经残差继续传播，故 key/query 双向屏蔽。
        if padding_mask is not None:
            key_mask = padding_mask.unsqueeze(1).unsqueeze(2)    # (B, 1, 1, S)
            query_mask = padding_mask.unsqueeze(
                1).unsqueeze(-1)  # (B, 1, S, 1)
            # (B, 1, S, S)
            attn_pad_mask = (key_mask & query_mask).bool()
        else:
            attn_pad_mask = None

        # 词嵌入
        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)

        # 通过Transformer块
        for block in self.blocks:
            x = block(x, token_ids=token_ids, padding_mask=attn_pad_mask)

        # 最终归一化和输出投影
        x = self.norm(x)
        logits = self.head(x)

        return logits


if __name__ == "__main__":
    # 测试

    # 创建模型参数
    args = MyLMArgs(
        d_model=512,
        d_inner=2048,
        n_layers=8,
        d_head=64,
        vocab_size=10000,
        seq_max_len=512,
        use_moe=False,
        dropout=0.1,
        base_init_std=0.02,  # 基础初始化标准差
        latent_moe=0,        # DeepNet 残差缩放已内置于 _reset_parameters
        d_latent=64,
        pad_id=0,
    )

    # 实例化模型
    model = MyLM(args)

    # 初始化参数
    model._reset_parameters()

    # 创建测试输入
    bsz = 32
    seq_length = 128
    input_ids = torch.randint(0, args.vocab_size, (bsz, seq_length))

    # 前向传播
    outputs = model(input_ids)

    # 检查输出形状
    assert outputs.shape == (
        bsz,
        seq_length,
        args.vocab_size,
    ), f"Output shape错误: {outputs.shape}"
    print("测试通过!")
    print(f"输出形状: {outputs.shape}")
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")

    # padding mask 隔离回归：左 padding 下真实 token 不应受 pad 内容影响。
    # 这是此前 seq_mask 零值假设失效（attn_bias 的 o_proj.bias 使 pad 行残差流
    # 第 1 层后非零，左右 pad 均污染注意力）的根因回归守卫。
    # 注：必须在 eval() 下比较数值——train 模式的 dropout 会让两次独立前向
    # 产生随机差异，与 pad 无关。
    model.eval()
    with torch.no_grad():
        a = torch.tensor([[10, 20, 5, 6, 7]])
        b = torch.tensor([[30, 40, 5, 6, 7]])
        pm = torch.tensor([[False, False, True, True, True]])  # 前两位是 pad
        oa = model(a, padding_mask=pm)
        ob = model(b, padding_mask=pm)
        leak = (oa[0, 2:] - ob[0, 2:]).abs().max().item()
        assert leak < 1e-5, f"padding mask 未隔离 pad 内容，泄漏 {leak}"
        # 无 mask 时仅 causal（推理路径），不应抛错
        _ = model(torch.tensor([[5, 6, 7]]))
    print("padding mask 隔离测试通过!")

    # 测试不同的缩放参数
    print("\n测试不同缩放参数...")
    args_scaled = MyLMArgs(
        d_model=256,
        d_inner=1024,
        n_layers=4,
        d_head=64,
        vocab_size=10000,
        seq_max_len=256,
        use_moe=False,
        dropout=0.1,
        base_init_std=0.02,
        latent_moe=0,
        d_latent=64,
        pad_id=0,
    )

    model_scaled = MyLM(args_scaled)
    model_scaled._reset_parameters()
    outputs_scaled = model_scaled(input_ids[:, :64])  # 使用较短序列
    print(f"缩放模型输出形状: {outputs_scaled.shape}")
    print("缩放模型测试通过!")

    # 新旧 RMSNorm 数值等价 + ckpt 键兼容守卫：模型现用 nn.RMSNorm，
    # 旧实现（上方已弃用的 RMSNorm 类）保留用于对照。两者 state_dict 均只有
    # weight 一个键，同权重同 eps 下输出应在 fp32 容差内一致。
    torch.manual_seed(0)
    legacy_norm = RMSNorm(64, eps=1e-6)
    native_norm = nn.RMSNorm(64, eps=1e-6)
    native_norm.load_state_dict(legacy_norm.state_dict())
    probe = torch.randn(4, 17, 64, dtype=torch.float32)
    diff = (legacy_norm(probe) - native_norm(probe)).abs().max().item()
    assert diff < 1e-6, f"nn.RMSNorm 与旧实现数值不一致，最大差 {diff}"
    assert set(native_norm.state_dict().keys()) == set(legacy_norm.state_dict().keys()), \
        "新旧实现 state_dict 键不一致，旧 ckpt 将无法加载"
    print("RMSNorm 新旧实现等价性测试通过!")

    # 架构配置接线守卫：奇数层必须全为 CompressedAttention，且
    # compress_ratio 从 MyLMArgs 注入（不是构造器默认值）；conv bias 已归零
    ca_layers = [blk.attn for blk in model_scaled.blocks
                 if isinstance(blk.attn, CompressedAttention)]
    assert len(ca_layers) == args_scaled.n_layers // 2, "奇数层应全为 CompressedAttention"
    assert all(l.compress_ratio == args_scaled.compress_ratio for l in ca_layers), \
        "compress_ratio 未从 MyLMArgs 读取"
    assert all(torch.count_nonzero(l.conv.conv.bias) == 0 for l in ca_layers), \
        "conv bias 未 zero-init"
    print("架构配置接线守卫通过!")

    # 训练反向 NaN 回归守卫：窗口因果判定使 query t<ratio-1 无任何可见窗口
    # （全 -inf 行），前向靠 softmax 后填 0；反向经 masked_fill/softmax
    # backward 在 torch 2.13 eager+CUDA 实测安全，此处守住未来 torch 升级
    # 或算子改动引入 NaN 梯度污染共享 q/k 投影的不变量
    model_scaled.train()
    ids_probe = torch.randint(1, args_scaled.vocab_size, (4, 33))
    ids_probe[:, -7:] = 0  # 右 pad + 33%8=1 覆盖尾窗与全屏蔽行
    logits_probe = model_scaled(ids_probe, padding_mask=(ids_probe != 0))
    logits_probe.pow(2).mean().backward()
    nan_params = [n for n, p in model_scaled.named_parameters()
                  if p.grad is not None and torch.isnan(p.grad).any()]
    assert not nan_params, f"反向产生 NaN 梯度: {nan_params}"
    print("训练方向 NaN 回归测试通过!")
