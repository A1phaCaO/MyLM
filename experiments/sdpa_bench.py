"""SDPA vs 标准 Attention 基准实验：速度 + 显存占据（训练态 fwd+bwd）。

对比 4 种组合 × 序列长度（可选两种分支）：
- impl: std（models.py Attention 原实现，显式物化 (B,H,L,L) 分数矩阵）
- impl: sdpa（同结构，forward 换 F.scaled_dot_product_attention，flash 内核）

每组分别测 eager / torch.compile(max-autotune，对齐 pre_train.py)。

用法（Windows 下 compile 需要 UTF-8 环境）:
$env:PYTHONUTF8='1'; uv run python experiments/sdpa_bench.py
$env:PYTHONUTF8='1'; uv run python experiments/sdpa_bench.py --seqs 256,512,1024,2048 --mode default
$env:PYTHONUTF8='1'; uv run python experiments/sdpa_bench.py --branch sigmoid --check --model-level

结果写入 experiments/results/sdpa_bench_<ts>.csv。
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import argparse
import datetime
import gc
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MyLM, MyLMArgs, Attention
from common import set_seed

RESULTS = Path(__file__).resolve().parent / "results"
COMPILE_MODE = "max-autotune"


class SDPAAttention(Attention):
    """与 models.Attention 参数结构完全相同，仅 forward 改用 SDPA（flash 内核）。

    - gate 分支：std 为 att@v 后乘 sigmoid(gate(x))，SDPA 输出同样后乘 gate
    - sigmoid 分支：std 为 att@F.sigmoid(v)，这里先对 v 激活再进 SDPA
    - 无 mask 时用 is_causal=True（零 mask 物化）；有 mask 时合并出 (B,1,L,L) bool
    """

    def forward(self, x, token_ids=None, mask=None, causal=True):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        if self.use_gate:
            gate = F.sigmoid(self.gate(x))
            v = self.v_proj(x)
        else:
            v = F.sigmoid(self.v_proj(x))

        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        attn_mask = None
        is_causal = causal
        if mask is not None:
            if mask.dim() != 4 or mask.shape != (batch_size, 1, seq_len, seq_len):
                raise ValueError(
                    f"Mask must be 4D (B,1,L,L), got {tuple(mask.shape)}")
            causal_mask = torch.tril(torch.ones(
                seq_len, seq_len, device=x.device, dtype=torch.bool)).view(
                1, 1, seq_len, seq_len
            ) if causal else torch.ones(
                seq_len, seq_len, device=x.device, dtype=torch.bool).view(
                1, 1, seq_len, seq_len
            )
            attn_mask = (causal_mask & mask).bool()
            is_causal = False

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(self.d_head),
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        if self.use_gate:
            y = y * gate
        y = self.resid_dropout(self.o_proj(y))
        return y


def make_sdpa_model(model: MyLM) -> MyLM:
    """整模型实验：把每个 decoder block 的 Attention 换成 SDPAAttention，
    用 load_state_dict 拷贝权重与 RoPE 缓存（同结构直接复制）。"""
    args = model.args
    for block in model.blocks:
        src = block.attn
        dst = SDPAAttention(args, use_gate=src.use_gate)
        dst.load_state_dict(src.state_dict())
        dst.to(next(src.parameters()).device)
        block.attn = dst
    return model


def module_args(seq_max, d_model=512, n_layers=8, d_head=128, dropout=0.05):
    return MyLMArgs(
        d_model=d_model,
        d_inner=int((d_model * (8 / 3) // 64) * 64),
        n_layers=n_layers,
        latent_moe=False,
        d_latent=d_model,
        vocab_size=2000,
        seq_max_len=seq_max,
        use_moe=False,
        n_heads=d_model // d_head,
        d_head=d_head,
        dropout=dropout,
        attn_bias=True,
        base_init_std=0.02,
    )


def _fwd_bwd(mod, x, L, B, args):
    mod.zero_grad(set_to_none=True)
    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        y = mod(x)
        loss = y.float().pow(2).mean()
    loss.backward()


@torch.no_grad()
def _fwd_eval(mod, x):
    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        return mod(x).float()


def bench_module(impl, use_compile, use_gate, args, L, B, steps, warmup,
                 compile_mode=COMPILE_MODE):
    """单个 Attention 模块的训练态（fwd+bwd）计时与峰值显存。
    OOM 时返回 (None, None)，由调用方打印 OOM。"""
    set_seed(42)
    mod = Attention(args, use_gate=use_gate) if impl == "std" \
        else SDPAAttention(args, use_gate=use_gate)
    mod = mod.to("cuda").train()
    if use_compile:
        mod = torch.compile(mod, mode=compile_mode)
    x = torch.randn(B, L, args.d_model, device="cuda")
    try:
        for _ in range(warmup):                      # 触发 compile + autotune
            _fwd_bwd(mod, x, L, B, args)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(steps):
            _fwd_bwd(mod, x, L, B, args)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / steps
        peak = torch.cuda.max_memory_allocated() / 2**20
        return dt * 1000, peak
    except torch.OutOfMemoryError:
        return None, None
    finally:
        del mod, x
        gc.collect()
        torch.cuda.empty_cache()


def model_args(seq_max, d_model=512, n_layers=8, d_head=128, dropout=0.05):
    a = module_args(seq_max, d_model, n_layers, d_head, dropout)
    a.vocab_size = 2000
    a.use_moe = False
    return a


def bench_model(use_sdpa, use_compile, args, L, B, steps, warmup,
                compile_mode=COMPILE_MODE):
    """整模型 MyLM（dense 8 层）训练态计时与峰值显存。OOM 时返回 (None, None)。"""
    set_seed(42)
    model = MyLM(args).to("cuda").train()
    if use_sdpa:
        model = make_sdpa_model(model)
    if use_compile:
        model = torch.compile(model, mode=compile_mode)
    ids = torch.randint(1, args.vocab_size, (B, L), device="cuda")
    tgt = torch.randint(1, args.vocab_size, (B, L), device="cuda")
    try:
        for _ in range(warmup):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                out = model(ids)
                loss = F.cross_entropy(out.view(-1, args.vocab_size), tgt.view(-1))
            loss.backward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                out = model(ids)
                loss = F.cross_entropy(out.view(-1, args.vocab_size), tgt.view(-1))
            loss.backward()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / steps
        peak = torch.cuda.max_memory_allocated() / 2**20
        return dt * 1000, peak
    except Exception as e:                # 含编译期 OOM（InductorError 等非 OutOfMemoryError）
        print(f"  [note] {type(e).__name__}: {e}", flush=True)
        return None, None
    finally:
        del model, ids, tgt
        gc.collect()
        torch.cuda.empty_cache()


def check_equivalence(args, use_gate, L=256, B=2):
    """数值一致性：std vs SDPA（dropout=0 时输出/梯度应基本一致）"""
    args_zero = module_args(L, dropout=0.0, d_model=args.d_model,
                            n_layers=args.n_layers, d_head=args.d_head)
    set_seed(42)
    std = Attention(args_zero, use_gate=use_gate).to("cuda").train()
    sdpa = SDPAAttention(args_zero, use_gate=use_gate).to("cuda").train()
    sdpa.load_state_dict(std.state_dict())
    x = torch.randn(B, L, args.d_model, device="cuda")
    y_std = _fwd_eval(std, x)
    y_sdpa = _fwd_eval(sdpa, x)
    out_rel = (y_std - y_sdpa).norm() / y_std.norm()

    loss = _fwd_bwd_tensor(std, sdpa, x)
    g_std = std.q_proj.weight.grad.detach().clone()
    g_sdpa = sdpa.q_proj.weight.grad.detach().clone()
    grad_rel = (g_std - g_sdpa).norm() / g_std.norm()
    del std, sdpa, x
    torch.cuda.empty_cache()
    return out_rel.item(), grad_rel.item(), loss


def _fwd_bwd_tensor(std, sdpa, x):
    for m in (std, sdpa):
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            loss = m(x).float().pow(2).mean()
        loss.backward()
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="256,1024",
                    help="逗号分隔序列长度列表")
    ap.add_argument("--mode", default=COMPILE_MODE,
                    help="torch.compile mode（对齐 pre_train 默认 max-autotune）")
    ap.add_argument("--branch", default="both", choices=["gate", "sigmoid", "both"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--d-head", type=int, default=128)
    ap.add_argument("--check", action="store_true", help="跑数值一致性检查")
    ap.add_argument("--model-level", action="store_true", help="附带整模型对比")
    ap.add_argument("--model-only", action="store_true", help="只跑整模型对比")
    ap.add_argument("--model-batch", type=int, default=None,
                    help="整模型 batch（默认同 --batch）")
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"torch {torch.__version__}  flash_sdp={torch.backends.cuda.flash_sdp_enabled()} "
          f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"d={args.d_model} n_layers={args.n_layers} d_head={args.d_head} "
          f"B={args.batch} compile_mode={args.mode}")

    seqs = [int(s) for s in args.seqs.split(",")]
    branches = ["gate", "sigmoid"] if args.branch == "both" else [args.branch]
    rows = []
    total_t0 = time.perf_counter()

    if args.check:
        print("\n=== 数值一致性（dropout=0, L=256, B=2）===")
        for b in branches:
            a = module_args(256, d_model=args.d_model, n_layers=args.n_layers,
                            d_head=args.d_head)
            o, g, _ = check_equivalence(a, use_gate=(b == "gate"))
            print(f"  {b:8s} out_rel_err={o:.2e}  grad_rel_err={g:.2e}"
                  + ("  (<1e-2 即等价)" if o < 1e-2 else "  [偏差过大!]"))

    print("\n=== 模块级: 单 Attention 训练态(fwd+bwd) ===")
    if not args.model_only:
        for b in branches:
            print(f"\n-- branch={b} --")
            header = f"{'seq':>5s} {'impl':>5s} {'compile':>9s} {'ms/step':>9s} {'peak_MiB':>10s} {'ktok/s':>8s}"
            print(header)
            for L in seqs:
                a = module_args(L, d_model=args.d_model, n_layers=args.n_layers,
                                d_head=args.d_head)
                for impl in ("std", "sdpa"):
                    for use_compile in (False, True):
                        ms, peak = bench_module(
                            impl, use_compile, use_gate=(b == "gate"), args=a,
                            L=L, B=args.batch, steps=args.steps, warmup=args.warmup,
                            compile_mode=args.mode)
                        if ms is None:
                            rows.append((b, impl, use_compile, L, args.batch,
                                         "OOM", "OOM", 0))
                            print(f"{L:5d} {impl:>5s} {str(use_compile):>9s} {'OOM':>9s} {'OOM':>10s}",
                                  flush=True)
                            continue
                        ktok = args.batch * L / (ms / 1000) / 1e6
                        rows.append((b, impl, use_compile, L, args.batch,
                                     ms, peak, ktok))
                        print(f"{L:5d} {impl:>5s} {str(use_compile):>9s} {ms:9.2f} {peak:10.1f} {ktok:8.2f}",
                              flush=True)

    if args.model_level or args.model_only:
        print("\n=== 整模型: MyLM(dense 8 层) 训练态 ===")
        mb = args.model_batch or args.batch
        for L in seqs:
            a = model_args(L, d_model=args.d_model, n_layers=args.n_layers,
                           d_head=args.d_head)
            for name, use_sdpa in (("std", False), ("sdpa", True)):
                for use_compile in (False, True):
                    ms, peak = bench_model(
                        use_sdpa, use_compile, a, L, mb,
                        steps=max(3, args.steps // 2), warmup=args.warmup,
                        compile_mode=args.mode)
                    if ms is None:
                        rows.append(("model-" + name, "full", use_compile, L,
                                     mb, "OOM", "OOM", 0))
                        print(f"{L:5d} {name:>5s} {str(use_compile):>9s} {'OOM':>9s} {'OOM':>10s}",
                              flush=True)
                        continue
                    rows.append(("model-" + name, "full", use_compile, L,
                                 mb, ms, peak, mb * L / (ms / 1000) / 1e6))
                    print(f"{L:5d} {name:>5s} {str(use_compile):>9s} {ms:9.2f} {peak:10.1f}",
                          flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"sdpa_bench_{ts}.csv"
    import csv
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["branch", "impl", "compile", "seq", "batch", "ms_step",
                    "peak_mib", "ktok_s"])
        w.writerows(rows)
    print(f"\n结果已保存: {out}")
    print(f"总耗时 {time.perf_counter() - total_t0:.0f}s")


if __name__ == "__main__":
    main()