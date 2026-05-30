# Gated Attention Operator Optimization

## Goal

Write a **high-performance PyTorch gated attention operator** that is faster and more memory-efficient than `GatedAttentionBaseline` in `benchmark.py`. Target **2x+ speedup**.

Use any technique available: `torch.compile`, kernel fusion, Flash Attention, Triton kernels, CUDA graphs, `scaled_dot_product_attention`, memory-efficient attention, etc.

## Setup

This project uses a `.venv` virtual environment (managed by `uv`). Before running benchmarks:

```bash
# Activate the environment:
.venv\Scripts\activate
```

Or run directly without activation:

```bash
uv run python benchmark.py
```

## Files

| File | Role |
|---|---|
| **`prepare.py`** | Fixed utilities: config, benchmark harness, correctness checker. **Do NOT modify.** |
| **`benchmark.py`** | The file you edit. Contains the baseline + your optimized implementations. |
| **`program.md`** | This file — instructions for the agent. |

## Gated Attention Definition

```
Given x: (B, S, D), mask: (1, 1, S, S) causal:
  Q = x @ W_q          # Linear, no bias
  K = x @ W_k          # Linear, no bias
  V = x @ W_v          # Linear, no bias
  G = sigmoid(x @ W_gate)  # Linear, no bias
  Q, K = apply_rope(Q, K)
  attn = softmax(Q @ K^T / sqrt(d_head) + mask)
  O = (attn @ V) * G   # gate applied element-wise after attention
  out = O @ W_o        # Linear, no bias
```

## How to Iterate

1. Read `benchmark.py` to understand the baseline and harness.
2. Add a new optimized class in the `# AGENT:` section of `benchmark.py`.
3. Register it in the `BENCHMARKS` dict.
4. Run: `python benchmark.py` in the `experiment/` directory.
5. Check throughput (tok/s), memory (MB), and correctness (cosine sim > 0.99).
6. Keep the best implementation, discard regressions. Repeat.

## Constraints

- Must be a valid `nn.Module` with `forward(self, x, mask) -> Tensor`.
- `x: (B, S, D)`, `mask: (1, 1, S, S)` causal bool mask.
- Numerically correct: cosine similarity > 0.99 with baseline.
- Keep `GatedAttentionBaseline` unchanged. Do NOT modify `prepare.py`.

## Scoring

`score = fwd_tok_s * correct + 0.5 * fwd_bwd_tok_s * correct`

Higher is better. A correct 2x speedup scores 2x + 1x = 3x baseline.
