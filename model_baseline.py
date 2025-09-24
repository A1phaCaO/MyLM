import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from models import MyLMArgs


class GPT2Baseline(nn.Module):
    """
    简化版GPT-2架构实现
    兼容MyLMArgs配置参数
    """

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.args = args

        # 词嵌入层
        self.token_embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.position_embedding = nn.Embedding(args.seq_max_len, args.d_model)

        # LayerNorm层
        self.embedding_ln = nn.LayerNorm(args.d_model)

        # Transformer块
        self.blocks = nn.ModuleList([GPT2Block(args) for _ in range(args.n_layers)])

        # 输出层
        self.ln_f = nn.LayerNorm(args.d_model)
        self.head = nn.Linear(args.d_model, args.vocab_size, bias=False)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        # 特殊处理输出层权重，与嵌入层共享
        self.head.weight = self.token_embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为(batch_size, seq_len)

        Returns:
            输出张量，形状为(batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()

        # 词嵌入和位置嵌入
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(
            0
        )  # (1, seq_len)
        pos_emb = self.position_embedding(pos)  # (1, seq_len, d_model)

        # 合并嵌入并归一化
        x = token_emb + pos_emb
        x = self.embedding_ln(x)

        # 通过Transformer块
        for block in self.blocks:
            x = block(x)

        # 最终LayerNorm和输出投影
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


class GPT2Block(nn.Module):
    """GPT-2 Transformer块"""

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(args.d_model)
        self.attn = GPT2Attention(args)
        self.ln_2 = nn.LayerNorm(args.d_model)
        self.mlp = GPT2MLP(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Attention(nn.Module):
    """GPT-2 注意力机制"""

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads or (args.d_model // args.d_head)
        self.d_head = args.d_head
        self.seq_max_len = args.seq_max_len

        # 注意力投影层
        self.c_attn = nn.Linear(args.d_model, 3 * args.d_model)
        self.c_proj = nn.Linear(args.d_model, args.d_model)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 计算QKV
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # 重塑为多头形式
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (batch, heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (batch, heads, seq, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (batch, heads, seq, head_dim)

        # 计算注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=att.device)).view(1, 1, seq_len, seq_len)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 应用注意力权重
        y = att @ v  # (batch, heads, seq, head_dim)
        y = (
            y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )  # 重新组合多头

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y


class GPT2MLP(nn.Module):
    """GPT-2 MLP层"""

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.c_fc = nn.Linear(args.d_model, args.d_inner)
        self.c_proj = nn.Linear(args.d_inner, args.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LLaMABaseline(nn.Module):
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
        self.blocks = nn.ModuleList([LLaMABlock(args) for _ in range(args.n_layers)])

        # 输出层
        self.norm = RMSNorm(args.d_model)
        self.head = nn.Linear(args.d_model, args.vocab_size, bias=False)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # 特殊处理输出层权重，与嵌入层共享
        self.head.weight = self.token_embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为(batch_size, seq_len)

        Returns:
            输出张量，形状为(batch_size, seq_len, vocab_size)
        """
        # 词嵌入
        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)

        # 通过Transformer块
        for block in self.blocks:
            x = block(x)

        # 最终归一化和输出投影
        x = self.norm(x)
        logits = self.head(x)

        return logits


class LLaMABlock(nn.Module):
    """LLaMA Transformer块"""

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.attn = LLaMAAttention(args)
        self.mlp = LLaMAMLP(args)
        self.input_layernorm = RMSNorm(args.d_model)
        self.post_attention_layernorm = RMSNorm(args.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力部分
        residual = x
        x = self.input_layernorm(x)
        x = self.attn(x)
        x = residual + x

        # MLP部分
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class LLaMAAttention(nn.Module):
    """LLaMA 注意力机制"""

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads or (args.d_model // args.d_head)
        self.d_head = args.d_head
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用RoPE位置编码"""
        # 调整cos和sin的维度以匹配q和k的序列长度
        cos = cos[:, :, : q.size(2), :]  # (1, 1, seq_len, d_head)
        sin = sin[:, :, : q.size(2), :]  # (1, 1, seq_len, d_head)

        # 将q和k分割为两半用于旋转操作
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=att.device)).view(1, 1, seq_len, seq_len)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
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


class LLaMAMLP(nn.Module):
    """LLaMA MLP层"""

    def __init__(self, args: MyLMArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.d_model, args.d_inner, bias=False)
        self.up_proj = nn.Linear(args.d_model, args.d_inner, bias=False)
        self.down_proj = nn.Linear(args.d_inner, args.d_model, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class RMSNorm(nn.Module):
    """RMS归一化"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
