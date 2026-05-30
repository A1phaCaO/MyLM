import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from prepare import (
    BenchmarkConfig, BenchmarkResult, run_single_benchmark,
    print_results, make_bench_input, make_causal_mask, get_dtype
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = BenchmarkConfig()

# ============================================================
# REFERENCE (baseline) — do NOT modify this class.
# Gated attention from models.py (Attention with use_gate=True)
# ============================================================

class GatedAttentionBaseline(nn.Module):
    """Reference implementation — do NOT remove or modify."""
    def __init__(self, D: int, H: int):
        super().__init__()
        self.D = D
        self.H = H
        head_dim = D // H
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.gate = nn.Linear(D, D, bias=False)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H = self.H
        head_dim = D // H

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = torch.sigmoid(self.gate(x))

        q = q.view(B, S, H, head_dim).transpose(1, 2)
        k = k.view(B, S, H, head_dim).transpose(1, 2)
        v = v.view(B, S, H, head_dim).transpose(1, 2)

        pos = torch.arange(S, device=x.device, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        cos = freqs.cos().view(1, 1, S, head_dim // 2).expand(-1, -1, -1, 2).reshape(1, 1, S, head_dim)
        sin = freqs.sin().view(1, 1, S, head_dim // 2).expand(-1, -1, -1, 2).reshape(1, 1, S, head_dim)
        q = q * cos + torch.cat((-q[..., head_dim//2:], q[..., :head_dim//2]), dim=-1) * sin
        k = k * cos + torch.cat((-k[..., head_dim//2:], k[..., :head_dim//2]), dim=-1) * sin

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        y = y * g
        y = self.o_proj(y)
        return y


# ============================================================
# AGENT: Add your optimized implementations below.
# Each class must have: forward(self, x, mask) -> Tensor
#   x:    (B, S, D)
#   mask: (1, 1, S, S) causal bool mask
# Register in BENCHMARKS dict below.
# ============================================================

# --- (agent implementations go here) ---


# ============================================================
# Registration
# ============================================================
BENCHMARKS = {
    "baseline": GatedAttentionBaseline,
}


def main():
    torch.manual_seed(42)
    print(f"Device: {DEVICE}")
    print(f"Config: B={CONFIG.B}, S={CONFIG.S}, D={CONFIG.D}, H={CONFIG.H}")
    print(f"Benchmark iterations: warmup={CONFIG.warmup_iters}, bench={CONFIG.bench_iters}")
    print()

    x = make_bench_input(CONFIG, DEVICE)
    mask = make_causal_mask(CONFIG.S, DEVICE)

    # Compute reference output from a seeded baseline (same weights for all comparisons)
    torch.manual_seed(0)
    ref_mod = GatedAttentionBaseline(CONFIG.D, CONFIG.H).to(DEVICE, get_dtype(CONFIG.dtype))
    ref_mod.eval()
    with torch.no_grad():
        ref_out = ref_mod(x, mask)

    results = []
    for name, cls in BENCHMARKS.items():
        print(f"Benchmarking {name}...")
        ref_for_this = None if name == "baseline" else ref_out
        result = run_single_benchmark(name, lambda c=cls: c(CONFIG.D, CONFIG.H), x, mask, CONFIG, ref_for_this)
        results.append(result)

    print_results(results)


if __name__ == "__main__":
    main()
