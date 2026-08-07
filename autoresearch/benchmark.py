import sys, os

if __package__:
    from .models import Attention, MyLMArgs
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import Attention, MyLMArgs

import torch
import time
from typing import Optional, Tuple


DEFAULT_BENCHMARK_CONFIG = {
    "d_model": 512,
    "d_head": 64,
    "n_heads": None,
    "seq_len": 256,
    "batch_size": 16,
    "dropout": 0.0,
    "n_warmup": 20,
    "n_iters": 100,
}


def create_attention(
    d_model: int = 512,
    d_head: int = 64,
    n_heads: Optional[int] = None,
    seq_max_len: int = 512,
    use_gate: bool = False,
    dropout: float = 0.0,
) -> Attention:
    n_heads = n_heads or (d_model // d_head)
    args = MyLMArgs(
        d_model=d_model,
        d_inner=d_model * 4,
        n_layers=1,
        vocab_size=10000,
        seq_max_len=seq_max_len,
        d_head=d_head,
        n_heads=n_heads,
        dropout=dropout,
    )
    return Attention(args, use_gate=use_gate)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark_attention(
    attn_module: Attention,
    batch_size: int = 16,
    seq_len: int = 256,
    device: Optional[torch.device] = None,
    n_warmup: int = 20,
    n_iters: int = 100,
) -> float:
    if device is None:
        device = get_device()
    attn_module = attn_module.to(device).train()

    x = torch.randn(
        batch_size, seq_len, attn_module.d_model,
        device=device, requires_grad=True,
    )

    def _step():
        out = attn_module(x, causal=True)
        loss = out.sum()
        loss.backward()
        if x.grad is not None:
            x.grad = None

    for _ in range(n_warmup):
        _step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(n_iters):
        _step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - start
    tokens = batch_size * seq_len * n_iters
    return tokens / total


def verify_correctness(
    attn_module: Attention,
    batch_size: int = 4,
    seq_len: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[bool, str]:
    if device is None:
        device = get_device()
    attn_module = attn_module.to(device).eval()
    d_model = attn_module.d_model

    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    with torch.set_grad_enabled(True):
        out = attn_module(x, causal=True)

    if out.shape != (batch_size, seq_len, d_model):
        return False, f"shape mismatch: expected {(batch_size, seq_len, d_model)}, got {out.shape}"

    if torch.isnan(out).any():
        return False, "output contains NaN"

    if torch.isinf(out).any():
        return False, "output contains Inf"

    try:
        loss = out.sum()
        loss.backward()
    except Exception as e:
        return False, f"backward failed: {e}"

    if x.grad is None:
        return False, "input.grad is None after backward"

    if x.grad.abs().sum().item() == 0:
        return False, "input.grad is all zero after backward"

    return True, "OK"


def compute_score(throughput: float, is_correct: bool) -> float:
    return throughput if is_correct else 0.0


if __name__ == "__main__":
    cfg = DEFAULT_BENCHMARK_CONFIG
    print(f"AutoResearch Benchmark | {cfg['d_model']}d-{cfg['d_head']}dh-{cfg['seq_len']}seq-{cfg['batch_size']}bs")
    print(f"Device: {get_device()}")
    print()

    attn = create_attention(
        d_model=cfg["d_model"],
        d_head=cfg["d_head"],
        seq_max_len=cfg["seq_len"],
    )

    ok, msg = verify_correctness(attn, device=get_device())
    print(f"Correctness: {'PASS' if ok else 'FAIL'} ({msg})")

    if ok:
        tput = benchmark_attention(
            attn,
            batch_size=cfg["batch_size"],
            seq_len=cfg["seq_len"],
            device=get_device(),
            n_warmup=cfg["n_warmup"],
            n_iters=cfg["n_iters"],
        )
        score = compute_score(tput, ok)
        print(f"Throughput:  {tput:,.0f} tokens/sec")
        print(f"Score:       {score:,.0f}")
