import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from typing import Optional, Callable, Tuple


@dataclass
class BenchmarkConfig:
    B: int = 8
    S: int = 1024
    D: int = 1024
    H: int = 16
    dtype: str = "bfloat16"
    warmup_iters: int = 3
    bench_iters: int = 20


@dataclass
class BenchmarkResult:
    name: str
    fwd_us: float
    fwd_bwd_us: float
    fwd_tok_s: float
    fwd_bwd_tok_s: float
    peak_mem_mb: float
    correct: bool
    cos_sim: float
    params_m: float


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]


def make_causal_mask(S: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(S, S, device=device, dtype=torch.bool)).view(1, 1, S, S)


def compute_cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)).item()


def check_correctness(ref: torch.Tensor, test: torch.Tensor, rtol=1e-3, atol=1e-3) -> Tuple[bool, float]:
    cos_sim = compute_cos_sim(ref, test)
    if not torch.allclose(ref.float(), test.float(), rtol=rtol, atol=atol):
        return False, cos_sim
    return True, cos_sim


def peak_memory_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def count_params(module: nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def run_single_benchmark(
    name: str,
    make_fn: Callable[[], nn.Module],
    x: torch.Tensor,
    mask: torch.Tensor,
    config: BenchmarkConfig,
    ref_output: Optional[torch.Tensor] = None,
) -> BenchmarkResult:
    device = x.device
    dtype = get_dtype(config.dtype)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    mod = make_fn().to(device, dtype)
    mod.train()
    params_m = count_params(mod)

    for _ in range(config.warmup_iters):
        out = mod(x, mask)
        out.sum().backward()

    # Benchmark forward
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        out = mod(x, mask)
    torch.cuda.synchronize()
    fwd_s = (time.perf_counter() - start) / config.bench_iters

    # Benchmark forward + backward
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        out = mod(x, mask)
        out.sum().backward()
    torch.cuda.synchronize()
    fwd_bwd_s = (time.perf_counter() - start) / config.bench_iters

    peak_mem = peak_memory_mb()
    tokens_per_s = config.B * config.S / fwd_s
    fwd_bwd_tokens_per_s = config.B * config.S / fwd_bwd_s

    correct = True
    cos_sim = 1.0
    if ref_output is not None:
        mod.eval()
        with torch.no_grad():
            test_out = mod(x, mask)
        correct, cos_sim = check_correctness(ref_output, test_out)

    return BenchmarkResult(
        name=name,
        fwd_us=fwd_s * 1e6,
        fwd_bwd_us=fwd_bwd_s * 1e6,
        fwd_tok_s=tokens_per_s,
        fwd_bwd_tok_s=fwd_bwd_tokens_per_s,
        peak_mem_mb=peak_mem,
        correct=correct,
        cos_sim=cos_sim,
        params_m=params_m,
    )


def print_results(results: list[BenchmarkResult]):
    print(f"\n{'='*120}")
    print(f"{'Name':<25} {'Fwd(us)':<12} {'Fwd+Bwd(us)':<14} {'Fwd(tok/s)':<14} {'Fwd+Bwd(tok/s)':<16} {'Mem(MB)':<10} {'Correct':<10} {'CosSim':<10} {'Params(M)':<10}")
    print(f"{'='*120}")
    for r in results:
        print(f"{r.name:<25} {r.fwd_us:<12.1f} {r.fwd_bwd_us:<14.1f} {r.fwd_tok_s:<14.1f} {r.fwd_bwd_tok_s:<16.1f} {r.peak_mem_mb:<10.1f} {str(r.correct):<10} {r.cos_sim:<10.4f} {r.params_m:<10.2f}")
    print(f"{'='*120}")

    baseline = results[0]
    print(f"\n{'Speedup vs Baseline':>25}")
    for r in results[1:]:
        fwd_speedup = r.fwd_tok_s / baseline.fwd_tok_s
        bwd_speedup = r.fwd_bwd_tok_s / baseline.fwd_bwd_tok_s
        mem_ratio = r.peak_mem_mb / baseline.peak_mem_mb
        status = "PASS" if r.correct else "FAIL"
        print(f"  {r.name:<22} fwd={fwd_speedup:.3f}x  fwd+bwd={bwd_speedup:.3f}x  mem={mem_ratio:.3f}x  [{status}]")


def make_bench_input(config: BenchmarkConfig, device: torch.device):
    dtype = get_dtype(config.dtype)
    return torch.randn(config.B, config.S, config.D, device=device, dtype=dtype)
