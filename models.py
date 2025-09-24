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
    vocab_size: int
    seq_max_len: int
    use_moe: bool = False
    n_heads: int = None
    n_experts: int = 4
    n_experts_per_tok: int = 2
    d_conv: int = 3
    conv_bias: bool = True
    ffn_bias: bool = False
    attn_bias: bool = False
    d_head: int = 64
    dropout: float = 0.1


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
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

    def _reset_parameters(self):
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        # nn.init.normal_(self.up_proj.weight, std=0.01)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        assert (
            seq_len <= self.pos_emb.num_embeddings
        ), f"序列长度 {seq_len} 超过预设最大值 {self.pos_emb.num_embeddings}"  # 检查序列长度是否超限
        # 生成位置编码并相加
        pos = torch.arange(seq_len).to(x.device)  # (seq_len,)
        pos_emb = self.pos_emb(pos)  # (seq_len, d_model)
        pos_emb = pos_emb.unsqueeze(0)  # (1, seq_len, d_model)
        return x + pos_emb  # 广播到 (batch_size, seq_len, d_model)


class MyPositionEmbedding(nn.Module):
    def __init__(self, d_model, d_out, d_inner=None):
        super().__init__()
        if d_inner is None:
            d_inner = d_model // 8
        self.d_inner = d_inner
        self.d_model = d_model
        self.gru = nn.GRU(d_inner, d_inner, bias=False, batch_first=True)
        self.pos_conv = nn.Conv1d(d_model, d_inner, 1)
        # self.proj_conv = nn.AdaptiveAvgPool1d(d_model - d_inner)
        self.proj_conv = nn.Conv1d(d_model, d_model - d_inner, 1)
        self.linear = nn.Linear(d_model, d_out, bias=False)
        # self.up_proj = nn.Linear(d_inner, d_model, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        res = x
        x = x.transpose(1, 2)
        x_proj = self.proj_conv(x).transpose(1, 2)  # 下采样非时间特征
        pos = self.pos_conv(x).transpose(1, 2)  # 下采样时间特征
        pos, _ = self.gru(pos)  # 时间特征RNN
        x = self.linear(torch.cat([pos, x_proj], dim=-1))  # 维度融合

        return x + res


class TokenShift(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 定义填充层：在时间维度上向后滑动一位（如[0,0,1,-1]表示在第三维前补1，后删1）
        self.pad = nn.ZeroPad2d((0, 0, 1, -1))  # 适用于输入形状为 (B, T, C)
        self.mix_k = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mix_k, mean=0, std=0.01)

    def forward(self, x):
        xx = self.pad(x) - x  # x_t + xx = x_t-1
        return x + xx * self.mix_k


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
    """LLaMA 注意力机制"""

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads or (args.d_model // args.d_head)
        self.d_head = args.d_head
        assert args.d_model % self.n_heads == 0, "d_model 必须是 n_heads 的整数倍"
        self.seq_max_len = args.seq_max_len

        # 注意力投影层
        self.q_proj = nn.Linear(args.d_model, args.d_model)
        self.k_proj = nn.Linear(args.d_model, args.d_model)
        self.v_proj = nn.Linear(args.d_model, args.d_model)
        self.o_proj = nn.Linear(args.d_model, args.d_model)

        # RoPE位置编码缓存
        self.register_buffer("cos_cached", torch.zeros(args.seq_max_len, args.d_head))
        self.register_buffer("sin_cached", torch.zeros(args.seq_max_len, args.d_head))
        self._init_rope()

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化参数"""
        nn.init.normal_(self.q_proj.weight, std=1/(4*(self.d_model**0.5)))
        nn.init.normal_(self.k_proj.weight, std=1/(4*(self.d_model**0.5)))
        nn.init.normal_(self.v_proj.weight, std=1/(4*(self.d_model**0.5)))
        nn.init.normal_(self.o_proj.weight, std=1/(4*(self.d_model**0.5)))


    def _init_rope(self):
        """初始化RoPE位置编码"""
        d_head_half = self.d_head // 2
        # 创建频率数组，长度为d_head_half
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, d_head_half, dtype=torch.float) / d_head_half)
        )

        t = torch.arange(self.seq_max_len, dtype=torch.float)
        # 计算位置频率
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # 扩展到完整维度并添加批次和头数维度
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d_head)
        self.cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_head)
        self.sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_head)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """旋转一半维度"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
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

    def forward(self, x: torch.Tensor, token_ids=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

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

        # 应用RoPE位置编码
        cos = self.cos_cached[:, :, :seq_len, :]  # (1, 1, seq_len, d_head)
        sin = self.sin_cached[:, :, :seq_len, :]  # (1, 1, seq_len, d_head)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # 计算注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=att.device)).view(
            1, 1, seq_len, seq_len
        )
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 应用注意力权重
        y = att @ v  # (batch, heads, seq, head_dim)
        y = (
            y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )  # 重新组合多头

        # 输出投影
        y = self.resid_dropout(self.o_proj(y))
        return y


class FFN(nn.Module):
    """LLaMA MLP层"""

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.args = args
        self.gate_proj = nn.Linear(args.d_model, args.d_inner, bias=False)
        self.up_proj = nn.Linear(args.d_model, args.d_inner, bias=False)
        self.down_proj = nn.Linear(args.d_inner, args.d_model, bias=False)
        self.act_fn = nn.SiLU()
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.normal_(self.gate_proj.weight, std=1/((self.args.d_model**0.5)))
        torch.nn.init.normal_(self.up_proj.weight, std=1/((self.args.d_model**0.5)))
        torch.nn.init.normal_(self.down_proj.weight, std=1/((self.args.d_inner**0.5)))

    def forward(self, x: torch.Tensor, token_ids=None) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEFFN(nn.Module):
    def __init__(self, args: MyLMArgs):
        """
        激活参数量为 n_experts_per_token * ffn
        总参数量为 n_experts * ffn
        """
        super().__init__()
        self.args = args
        self.router = nn.Linear(args.d_model, args.n_experts)
        self.experts = nn.ModuleList([FFN(args) for _ in range(args.n_experts)])
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.router.weight, nonlinearity="linear")

    def forward(self, x, token_ids=None):
        # 路由块
        probs = F.softmax(self.router(x), dim=-1)
        top_k_probs, top_k_indices = torch.topk(
            probs, self.args.n_experts_per_tok, dim=-1
        )
        # 归一化概率
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 专家块
        expert_inputs = x.view(
            -1, x.shape[-1]
        )  # 展平出所有token [batch_size*seq_len, hidden_size]
        expert_inputs = expert_inputs.repeat_interleave(
            self.args.n_experts_per_tok, dim=0
        )
        flat_top_k_idx = top_k_indices.view(-1)  # 展平出所有token对应的专家索引
        expert_outputs = torch.zeros_like(expert_inputs)  # 分配专家输出

        # 遍历所有专家 处理对应token
        for expert_idx, expert in enumerate(self.experts):
            mask = flat_top_k_idx == expert_idx  # 该专家
            if mask.any():
                expert_outputs[mask] = expert(expert_inputs[mask], token_ids=token_ids)
        expert_outputs = (
            expert_outputs.view(*top_k_probs.shape, -1) * top_k_probs.unsqueeze(-1)
        ).sum(
            dim=2
        )  # 乘以对应权重求和得到最终输出

        return expert_outputs


class MyLMDecoderLayer(nn.Module):
    """LLaMA Transformer块"""

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.attn = Attention(args)
        self.mlp = MoEFFN(args) if args.use_moe else FFN(args)
        self.input_layernorm = RMSNorm(args.d_model)
        self.post_attention_layernorm = RMSNorm(args.d_model)

    def forward(self, x: torch.Tensor, token_ids=None) -> torch.Tensor:
        # 注意力部分
        residual = x
        x = self.input_layernorm(x)
        x = self.attn(x, token_ids=token_ids)
        x = residual + x

        # MLP部分
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, token_ids=token_ids)
        x = residual + x

        return x


class MyLM(nn.Module):
    """
    简化版LLaMA架构实现
    兼容MyLMArgs配置参数
    """

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.args = args

        # 词嵌入层
        self.token_embedding = nn.Embedding(args.vocab_size, args.d_model)

        # Transformer块
        self.blocks = nn.ModuleList(
            [MyLMDecoderLayer(args) for _ in range(args.n_layers)]
        )

        # 输出层
        self.norm = RMSNorm(args.d_model)
        self.head = nn.Linear(args.d_model, args.vocab_size, bias=False)

        # 权重初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        # 特殊处理输出层权重，与嵌入层共享
        self.head.weight = self.token_embedding.weight

    def forward(self, x: torch.Tensor, token_ids=None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为(batch_size, seq_len)
            token_ids: token ID，用于某些特殊操作

        Returns:
            输出张量，形状为(batch_size, seq_len, vocab_size)
        """
        # 词嵌入
        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)

        # 通过Transformer块
        for block in self.blocks:
            x = block(x, token_ids=token_ids)

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
    )

    # 实例化模型
    model = MyLM(args)

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
