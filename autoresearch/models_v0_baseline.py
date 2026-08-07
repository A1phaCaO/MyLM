"""V0 — Original baseline: separate Q/K/V, manual attention (Q@K.T, tril, softmax), FP32."""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    base_init_std: float = 0.02
    resid_pdrop: float = 0.1
    resid_scale: float = 1.0
    layer_scale: float = 1.0
    use_deepnet_scaling: bool = True


class Attention(nn.Module):
    def __init__(self, args: MyLMArgs, use_gate=False, base_init_std=0.02):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads or (args.d_model // args.d_head)
        self.d_head = args.d_head
        self.seq_max_len = args.seq_max_len
        self.use_gate = use_gate

        self.q_proj = nn.Linear(args.d_model, args.d_model, bias=args.attn_bias)
        self.k_proj = nn.Linear(args.d_model, args.d_model, bias=args.attn_bias)
        self.v_proj = nn.Linear(args.d_model, args.d_model, bias=args.attn_bias)
        self.o_proj = nn.Linear(args.d_model, args.d_model, bias=args.attn_bias)

        if use_gate:
            self.gate = nn.Linear(args.d_model, args.d_model, bias=False)

        self.register_buffer(
            "cos_cached", torch.zeros(1, 1, args.seq_max_len, args.d_head)
        )
        self.register_buffer(
            "sin_cached", torch.zeros(1, 1, args.seq_max_len, args.d_head)
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        self._init_rope()
        self._reset_parameters(base_init_std=base_init_std)

    def _reset_parameters(self, base_init_std=0.02):
        torch.nn.init.normal_(self.q_proj.weight, std=base_init_std)
        torch.nn.init.normal_(self.k_proj.weight, std=base_init_std)
        torch.nn.init.normal_(self.v_proj.weight, std=base_init_std)
        torch.nn.init.normal_(self.o_proj.weight, std=base_init_std)
        if self.use_gate and hasattr(self, 'gate'):
            torch.nn.init.normal_(self.gate.weight, std=base_init_std)

    def _init_rope(self):
        d_head_half = self.d_head // 2
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, d_head_half, dtype=torch.float) / d_head_half)
        )
        t = torch.arange(self.seq_max_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer(
            "sin_cached", emb.sin().unsqueeze(0).unsqueeze(0)
        )

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ):
        cos = cos[:, :, : q.size(2), :]
        sin = sin[:, :, : q.size(2), :]
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, x: torch.Tensor, token_ids=None, mask=None, causal=True) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_gate:
            gate = F.sigmoid(self.gate(x))

        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        if causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)).view(
                1, 1, seq_len, seq_len
            )
        else:
            causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).view(
                1, 1, seq_len, seq_len
            )

        if mask is not None:
            if mask.dim() != 4:
                raise ValueError(f"Mask must be 4-dimensional, got {mask.dim()} dimensions")
            if mask.shape != (batch_size, 1, seq_len, seq_len):
                raise ValueError(f"Mask shape must be {(batch_size, 1, seq_len, seq_len)}, got {mask.shape}")
            combined_mask = causal_mask & mask
        else:
            combined_mask = causal_mask

        att = att.masked_fill(combined_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        if self.use_gate:
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            y = y * gate
        else:
            y = att @ F.sigmoid(v)
            y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        y = self.resid_dropout(self.o_proj(y))
        return y


if __name__ == "__main__":
    args = MyLMArgs(
        d_model=512, d_inner=2048, n_layers=1, vocab_size=10000,
        seq_max_len=256, d_head=64,
    )
    m = Attention(args, use_gate=False)
    x = torch.randn(2, 128, 512)
    out = m(x, causal=True)
    print(f"output shape: {out.shape}")
    print("OK")
