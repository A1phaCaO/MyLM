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
    d_conv: int = 3
    conv_bias: bool = True
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
            # 标准注意力：对V应用sigmoid激活
            y = att @ F.sigmoid(v)  # (batch, heads, seq, head_dim)
            y = (
                y.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, self.d_model)
            )  # 重新组合多头

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
                self.MCap = max(1, math.ceil((T * K / N) * self.kappa / 16) * 16)
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
                    1, math.ceil((self.args.seq_max_len * K / N) * self.kappa / 16) * 16
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
        self.attn = Attention(
            args,
            use_gate=(layer_idx % 2 == 0),  # 偶数层使用门控注意力，奇数层使用标准注意力
            base_init_std=base_init_std
        )
        self.mlp = (
            MoEFFN(args, base_init_std=base_init_std)
            if args.use_moe
            else FFN(args, base_init_std=base_init_std)
        )
        self.input_layernorm = RMSNorm(args.d_model)
        self.post_attention_layernorm = RMSNorm(args.d_model)

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
        self.norm = RMSNorm(args.d_model)
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
            query_mask = padding_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, S, 1)
            attn_pad_mask = (key_mask & query_mask).bool()        # (B, 1, S, S)
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
