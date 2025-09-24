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
    n_heads: int
    n_layers: int
    vocab_size: int
    seq_max_len: int
    use_moe: bool = False
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
        self.pos_emb = nn.Embedding(seq_max_len, d_model // 8)
        self.up_proj = nn.Linear(d_model // 8, d_model, bias=False)

    def _reset_parameters(self):
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        nn.init.normal_(self.up_proj.weight, std=0.01)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        assert (
            seq_len <= self.pos_emb.num_embeddings
        ), f"序列长度 {seq_len} 超过预设最大值 {self.pos_emb.num_embeddings}"  # 检查序列长度是否超限
        # 生成位置编码并相加
        pos = torch.arange(seq_len).to(x.device)  # (seq_len,)
        pos_emb = self.pos_emb(pos)  # (seq_len, d_model)
        pos_emb = self.up_proj(pos_emb)
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
        self.pos_pool = nn.AdaptiveAvgPool1d(d_inner)
        self.proj_pool = nn.AdaptiveAvgPool1d(d_model - d_inner)
        self.linear = nn.Linear(d_model, d_out, bias=False)
        # self.up_proj = nn.Linear(d_inner, d_model, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        res = x
        x_proj = self.proj_pool(x)
        pos = self.pos_pool(x)
        pos, _ = self.gru(pos)
        x = self.linear(torch.cat([pos, x_proj], dim=-1))

        return (x + res).permute(1, 0, 2)


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
    def __init__(self, embed_dim, head_dim, num_heads, vocab_size=None, dropout=0, bias=True):
        """
        Args:
            embed_dim: 模型的总维度
            num_heads: 并行注意力头的数量
            dropout: dropout概率
            bias: 是否在线性变换中使用偏置
        """
        super().__init__()
        self.embed_dim = embed_dim
        if head_dim is None:
            head_dim = 64
        if num_heads is None:
            num_heads = embed_dim // head_dim
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # 线性变换矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # self.q_shift = TokenShift(self.embed_dim)
        # self.k_shift = TokenShift(self.embed_dim)
        # self.v_shift = TokenShift(self.embed_dim)
        self.q_pos = MyPositionEmbedding(embed_dim, embed_dim)
        self.k_pos = MyPositionEmbedding(embed_dim, embed_dim)

        if vocab_size is not None:
            # self.k_emb = nn.Embedding(vocab_size, embed_dim)
            self.v_emb = nn.Embedding(vocab_size, embed_dim)

        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)


        self.dropout = dropout
        self.bias = bias
        # 初始化权重
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier/Kaiming初始化对于注意力权重效果较好
        # torch.nn.init.xavier_uniform_(self.q_proj.weight)
        # torch.nn.init.xavier_uniform_(self.k_proj.weight)
        # torch.nn.init.xavier_uniform_(self.v_proj.weight)
        torch.nn.init.kaiming_normal_(self.out_proj.weight)
        # 判断是否存在v_emb
        if hasattr(self, "v_emb"):
            torch.nn.init.ones_(self.v_emb.weight)

        # 如果有偏置项，初始化为小的常数
        if self.bias is True:
            torch.nn.init.constant_(self.q_proj.bias, 0.0)
            torch.nn.init.constant_(self.k_proj.bias, 0.0)
            # torch.nn.init.constant_(self.v_proj.bias, 0.0)
            torch.nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x,
        token_ids=None,
        attn_bias=None,
        batch_first=True,
        need_weights=False,
        attn_mask=None,
        is_causal=True,
    ):
        """
        Args:
            query: [T, B, E] 或 [B, T, E] 取决于 batch_first
            key: [S, B, E] 或 [B, S, E] 取决于 batch_first
            value: [S, B, E] 或 [B, S, E] 取决于 batch_first
            need_weights: 是否返回注意力权重
            attn_mask: [T, S] 或 [T, S], 会被加到计算出的注意力权重上
        Returns:
            attn_output: [T, B, E] 或 [B, T, E] 取决于 batch_first
            attn_weights: [B, T, S] (如果need_weights=True)
        """
        if batch_first:
            # 如果 batch_first 为 True，输入形状为 [B, T, E]，需要转置为 [T, B, E]
            x = x.transpose(0, 1)

        seq_len, bsz, embed_dim = x.size()
        # q = self.q_shift(x)
        # k = self.k_shift(x)
        # v = self.v_shift(x)
        # 线性变换
        # q = self.q_proj(x)
        # k = self.k_proj(x)
        # v = self.v_proj(x)
        q = self.q_pos(x)
        k = self.k_pos(x)
        v = F.silu(x) * self.v_emb(token_ids).permute(1, 0, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # 重排形状以支持多头 [T, B, H, D]
        q = (
            q.contiguous()
            .view(seq_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(seq_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(seq_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # if need_weights == False and attn_bias is None:
        #     NotImplementedError("弃用")
        #     # 使用PyTorch内置的缩放点积注意力(自动使用FlashAttention如果可用)
        #     # attn_output = F.scaled_dot_product_attention(
        #     #     q,
        #     #     k,
        #     #     v,
        #     #     attn_mask=attn_mask,
        #     #     dropout_p=self.dropout if self.training else 0.0,
        #     #     is_causal=True,
        #     # )  # 如果没有显式mask，假设是因果注意力
        # else:
        if is_causal:
            mask = (
                torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(
                    x.device
                )
                if attn_mask is None
                else attn_mask
            )
        else:
            mask = attn_mask

        mask = mask.to(torch.bool)
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) # QK/sqrt(d_k)

        # 应用掩码
        if attn_bias is not None:
            attn_weights += attn_bias

        if mask is not None:
            attn_weights += attn_weights.masked_fill(mask, float("-inf"))


        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ v  # 加权求和


        # 恢复形状 [B, T, H*D]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
        )
        # 最终输出投影
        attn_output = self.out_proj(attn_output)

        if batch_first:
            # 如果 batch_first 为 True，输出形状需要转回 [B, T, E]
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output


class FFN(nn.Module):
    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.d_model, args.d_inner, bias=args.ffn_bias)
        self.down_proj = nn.Linear(args.d_inner, args.d_model, bias=args.ffn_bias)
        self.up_proj = nn.Linear(args.d_model, args.d_inner, bias=args.ffn_bias)
        self.dropout = nn.Dropout(args.dropout)
        self.deep_emb = nn.Embedding(args.vocab_size, args.d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.gate_proj.weight)
        nn.init.kaiming_normal_(self.down_proj.weight)
        nn.init.kaiming_normal_(self.up_proj.weight)
        nn.init.ones_(self.deep_emb.weight)

    def telu(self, x):
        return F.tanh(torch.exp(x)) * x

    def forward(self, x, token_ids=None):

        # FFN Block
        # SwiGLU
        # x = self.up_proj(x) * F.silu(self.gate_proj(x))
        x = self.up_proj(x) * self.telu(self.gate_proj(x))
        x = self.down_proj(x)
        x = self.dropout(x)
        x = x * self.deep_emb(token_ids)
        return x


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

    def forward(self, x):
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
                expert_outputs[mask] = expert(expert_inputs[mask])
        expert_outputs = (
            expert_outputs.view(*top_k_probs.shape, -1) * top_k_probs.unsqueeze(-1)
        ).sum(
            dim=2
        )  # 乘以对应权重求和得到最终输出

        return expert_outputs


class MyLMDecoderLayer(nn.Module):

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.d_inner = args.d_inner
        self.dropout = nn.Dropout(args.dropout)
        # self.alibi = ALiBi(args.d_model // 64)
        self.mha: Attention = Attention(
            args.d_model,
            args.d_head,
            args.n_heads,
            dropout=args.dropout,
            bias=args.attn_bias,
            vocab_size=args.vocab_size,
        )
        
        self.ffn = MoEFFN(args) if args.use_moe else FFN(args)

        self.pre_attn_norm = RMSNorm(args.d_model)
        self.pre_ffn_norm = RMSNorm(args.d_model)
        self.ffn_alpha = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.attn_alpha = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self._reset_parameters()

    def _reset_parameters(self): ...

    def forward(self, x, token_ids=None):
        # attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
        #     x.device
        # )

        # Attention Block
        res = x
        x = self.pre_attn_norm(x)

        x = self.mha(
            x,
            token_ids=token_ids,
            # attn_bias=alibi_bias,
            is_causal=True,
            batch_first=True,
        )
        x = x + res * self.attn_alpha
        # FFN Block
        res = x
        x = self.pre_ffn_norm(x)
        if token_ids is not None:
            x = self.ffn(x, token_ids)
        else:
            x = self.ffn(x)

        x = x + res * self.ffn_alpha
        # x = res * self.res_alpha + x
        return x


class MyLM(nn.Module):

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.model_args = args

        # 模型层
        self.emb = nn.Embedding(args.vocab_size, args.d_model)
        # self.pos = GPT2PositionEmbedding(args.seq_max_len, args.d_model)
        # self.pos = MyPositionEmbedding(args.d_model)
        self.blocks = nn.ModuleList([MyLMDecoderLayer(args) for _ in range(args.n_layers)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self._reset_parameters()
        self.lm_head.weight = self.emb.weight

    def _reset_parameters(self):
        # 初始化
        # nn.init.kaiming_normal_(self.emb.weight, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.emb.weight, mean=0, std=0.02)

    def forward(self, x):
        token_ids = x
        x = self.emb(x)
        # x = self.pos(x)

        assert not torch.isnan(x).any(), f"emb nan"
        # 每经过一层block也进行检查
        for i, layer in enumerate(self.blocks):
            x = layer(x, token_ids)
            assert not torch.isnan(x).any(), f"After block {i+1} contains NaN"

        x = self.norm_f(x)
        x = self.lm_head(x)

        return x


if __name__ == "__main__":
    # 测试

    # 创建模型参数
    args = MyLMArgs(
        d_model=512,
        d_inner=2048,
        n_layers=8,
        n_heads=512//64,
        vocab_size=10000,
        seq_max_len=512,
        use_moe=False,
        dropout=0.1,
    )

    # 实例化模型
    model = MyLM(args)

    # 创建测试输入
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_length))

    # 前向传播
    outputs = model(input_ids)

    # 检查输出形状
    assert outputs.shape == (
        batch_size,
        seq_length,
        args.vocab_size,
    ), f"Output shape错误: {outputs.shape}"
    print("测试通过!")
    print(f"输出形状: {outputs.shape}")
