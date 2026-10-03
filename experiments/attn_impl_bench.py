# -*- coding: utf-8 -*-
"""手写注意力 vs F.scaled_dot_product_attention —— 速度 / 吞吐 / 显存 全量对比。

要回答的问题（来自"上三角白算一半 FLOPs，换 SDPA 能提速 2-4x"这个论断）:
1. core    : 只测注意力算子本身（q,k,v 已就绪），手写全量 matmul+mask 与 SDPA 差多少？
2. compile : 开 torch.compile(max-autotune, 对齐 pre_train.py) 后差距是被抹平还是反转？
3. memory  : O(L^2) 分数矩阵 vs 不物化，显存差多少？换算成"同显存能开多大 batch"？
4. padmask : 生产路径带 4D padding mask（mask 不为 None）时 SDPA 还剩多少优势？
5. backend : 本 GPU 上 SDPA 实际用了哪个内核（flash 不可用时退到 cuDNN / mem-efficient）？

对照组实现严格复刻 models.py:Attention 的数值语义（含 softmax 后把被屏蔽位置
再填 0 的"全屏蔽行不出 NaN"保护）；SDPA 组用行级 where 补齐同一保护，保证两者
在 padding 场景下输出可互相替换。

用法（Windows compile 需 UTF-8）:
  $env:PYTHONUTF8='1'; uv run python experiments/attn_impl_bench.py --stages probe
  $env:PYTHONUTF8='1'; uv run python experiments/attn_impl_bench.py --stages core --seqs 256,1024
  $env:PYTHONUTF8='1'; uv run python experiments/attn_impl_bench.py --stages module --seqs 256,1024
  $env:PYTHONUTF8='1'; uv run python experiments/attn_impl_bench.py --stages model,capacity

结果写入 experiments/results/attn_impl_bench_<tag>_<ts>.csv / .md
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import argparse
import contextlib
import csv
import datetime
import gc
import math
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from models import Attention, MyLM, MyLMArgs  # noqa: E402
from common import set_seed  # noqa: E402

RESULTS = ROOT / "experiments" / "results"

# 生产口径（pre_train.py: d_model=512, d_head=128 -> n_heads=4, n_layers=6,
# seq_max_len=256, batch=64, dropout=0.05, bf16 autocast）
D_MODEL = 512
D_HEAD = 128
N_HEADS = D_MODEL // D_HEAD
N_LAYERS = 6
VOCAB = 7160
TRAIN_TOKENS = 64 * 256          # 一个训练 micro-batch 的 token 数


# ----------------------------- 被测实现 -----------------------------

def _combined_mask(causal: bool, mask, L: int, device):
    """复刻 models.py:Attention 里 combined_mask 的构造（含每次 forward 重建 tril）。"""
    if causal:
        causal_mask = torch.tril(torch.ones(L, L, device=device,
                                           dtype=torch.bool)).view(1, 1, L, L)
    else:
        causal_mask = torch.ones(L, L, device=device,
                                 dtype=torch.bool).view(1, 1, L, L)
    return causal_mask if mask is None else (causal_mask & mask)


class StdCore(nn.Module):
    """models.py:Attention 第 339-390 行的注意力段逐行复刻（物化 (B,H,L,L)）。"""

    def __init__(self, causal=True):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v, mask=None):
        L = q.size(-2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        combined = _combined_mask(self.causal, mask, L, q.device)
        att = att.masked_fill(combined == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(combined == 0, 0.0)
        return att @ v


class SdpaCore(nn.Module):
    """同一数学，交给 SDPA。backend=None 表示让 PyTorch 自己按优先级选。

    dropout 对齐 models.py:Attention 的口径：它是在 softmax **之后**对注意力概率
    做 attn_dropout（被屏蔽位此时已是 0，dropout 不会复活它们），与 SDPA 的
    dropout_p（同样作用于 softmax 后的 p）语义一致。若这里不传 dropout_p，
    sdpa 侧就省掉了生产里真实存在的 (B,H,L,L) RNG 开销，对比会偏向 sdpa。
    """

    def __init__(self, causal=True, backend=None, dropout=0.0):
        super().__init__()
        self.causal = causal
        self.backend = backend
        self.dropout = float(dropout)

    def forward(self, q, k, v, mask=None):
        L = q.size(-2)
        attn_mask = None
        is_causal = self.causal
        if mask is not None:
            attn_mask = _combined_mask(self.causal, mask, L, q.device)
            is_causal = False
        ctx = (sdpa_kernel([self.backend]) if self.backend is not None
               else contextlib.nullcontext())
        with ctx:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                dropout_p=self.dropout if self.training else 0.0,
                scale=1.0 / math.sqrt(q.size(-1)))
        if mask is not None:
            # 对齐 std 的"全屏蔽行输出 0"保护：flash/cuDNN 对整行被屏蔽的 query
            # 直接给 0 或 NaN 依内核而定，这里显式清零，保证可替换性。
            valid_row = attn_mask.any(dim=-1, keepdim=True)
            y = torch.where(valid_row, y, torch.zeros_like(y))
        return y


class SDPAAttention(Attention):
    """与 models.Attention 结构/超参完全相同，只把注意力段换成 SDPA。

    与旧 experiments/sdpa_bench.py 的区别：这里严格按现行 models.py 的
    QK-Norm -> RoPE 顺序、且 V 不做激活（2026-09 工作区已移除 sigmoid），
    并同样承担 attn_dropout（旧脚本漏了 q_norm/k_norm，数值不可比）。
    """

    def __init__(self, args, use_gate=False, base_init_std=0.02):
        super().__init__(args, use_gate=use_gate, base_init_std=base_init_std)
        self.core = SdpaCore(causal=True, dropout=args.dropout)

    def forward(self, x, token_ids=None, mask=None, causal=True):
        B, L, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.use_gate:
            gate = F.sigmoid(self.gate(x))

        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        q = self.q_norm(q)                      # 现行口径：先 norm 再 RoPE
        k = self.k_norm(k)
        cos = self.cos_cached[:, :, :L, :]
        sin = self.sin_cached[:, :, :L, :]
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        y = self.core(q, k, v, mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        if self.use_gate:
            y = y * gate
        return self.resid_dropout(self.o_proj(y))


# ----------------------------- 计时工具 -----------------------------

class EventTimer:
    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.s.record()
        return self

    def __exit__(self, *exc):
        self.e.record()
        torch.cuda.synchronize()
        return False

    def ms(self):
        return self.s.elapsed_time(self.e)


def _top_kernels(label, fn, iters=4):
    def autocast_step():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            fn()
    for _ in range(3):
        autocast_step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            autocast_step()
        torch.cuda.synchronize()
    evs = sorted([e for e in prof.key_averages() if e.device_time_total > 0],
                 key=lambda e: -e.device_time_total)[:5]
    print(f"  [{label}] " + " | ".join(
        f"{e.key.split('(')[0][:34]}:{e.device_time_total / iters:.0f}us"
        f" x{max(1, e.count // iters)}" for e in evs), flush=True)


def _median(ts):
    return statistics.median(ts) if ts else float("nan")


class Case:
    """一个待测组合：module + 输入 + loss 构造，负责 warmup/计时/显存统计。"""

    def __init__(self, module, inputs, tokens, label, mem0=0):
        self.mod = module
        self.inputs = inputs
        self.tokens = tokens
        self.label = label
        self.mem0 = mem0          # 构造本 case 之前的设备空闲量
        self._objs = None
        self.mem_probe = 0.0

    def grad_objs(self):
        if self._objs is None:
            objs = (list(self.mod.parameters())
                    if isinstance(self.mod, nn.Module) else [])
            objs += [t for t in self.inputs if isinstance(t, torch.Tensor)
                     and t.requires_grad]
            self._objs = [p for p in objs if p.requires_grad]
        return self._objs

    def prealloc(self):
        """预分配 .grad 缓冲：CUDAGraph 要求反向图的梯度输出落在已有缓冲上
        （pre_train.py 同一条约定），否则报 "gradient tensor output of
        CUDAGraphs that has been overwritten"。"""
        for p in self.grad_objs():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

    def zero(self):
        # 口径同 pre_train.py：保留 .grad 缓冲置零（CUDAGraph 友好），
        # core 级没有参数，梯度落在 q/k/v 叶子上，一并清理
        for p in self.grad_objs():
            if p.grad is not None:
                p.grad.zero_()

    def fwd(self):
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            return self.mod(*self.inputs)

    def step(self, sample=False):
        self.zero()
        out = self.fwd()
        loss = out.float().pow(2).mean()
        loss.backward()
        if sample:
            self.mem_probe = self.dev_used()
        del out, loss

    def dev_used(self):
        """设备级已用量相对基线的增量（含 CUDA Graph 私有池）。"""
        return (self.mem0 - torch.cuda.mem_get_info()[0]) / 2**20

    def run(self, warmup, iters):
        """先跑计时轮（干净口径），再单独跑显存轮（不计时），避免采样污染时间。"""
        self.prealloc()
        for _ in range(max(2, warmup // 2)):
            with torch.no_grad():
                self.fwd()
        torch.cuda.synchronize()
        fp_ms = []
        for _ in range(iters):
            with torch.no_grad(), EventTimer() as t:
                self.fwd()
            fp_ms.append(t.ms())
        for _ in range(warmup):
            self.step()
        torch.cuda.synchronize()
        step_ms = []
        for _ in range(iters):
            with EventTimer() as t:
                self.step()
            step_ms.append(t.ms())
        # ---- 显存轮（不参与计时，用后台线程抓 step 内部峰值）----
        torch.cuda.reset_peak_memory_stats()
        with DevPeakSampler(self.mem0) as s:
            for _ in range(4):
                with torch.no_grad():
                    out = self.fwd()
                del out
        fp_dev = s.mib
        fp_peak = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with DevPeakSampler(self.mem0) as s:
            for _ in range(4):
                self.step()
        st_dev = s.mib
        return {
            "ms_step": _median(step_ms),
            "ms_step_mean": statistics.fmean(step_ms),
            "ms_step_std": statistics.stdev(step_ms) if len(step_ms) > 1 else 0.0,
            "ms_fp": _median(fp_ms),
            "step_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
            "step_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "step_dev_mib": st_dev,
            "fp_peak_mib": fp_peak,
            "fp_dev_mib": fp_dev,
            "ktok_s": self.tokens / statistics.fmean(step_ms),
        }


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    # 关键：torch.compile 的缓存按 code object 共享，跨 case 累积不同形状/mask
    # 会打满 recompile_limit 并**静默退回 eager**（表现为"编译无收益"），
    # 每个 case 之后必须 reset，配合 inductor 磁盘缓存仍比首次编译快得多。
    torch._dynamo.reset()
    torch.cuda.synchronize()


def free_bytes() -> int:
    """设备级空闲显存字节数（含 CUDA Graph 私有池的影响，
    而 max_memory_allocated 看不到图池，只有 mem_get_info 才公平）。"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0]


def free_mib() -> float:
    return free_bytes() / 2**20


# ----------------------------- 输入构造 -----------------------------

def pad_mask(B, L, valid_frac, device, left=False):
    """生产同款 (B,1,L,L) bool padding mask（key&query 相交后的形态，见 MyLM.forward）。"""
    n_keep = max(1, int(L * valid_frac))
    valid = torch.zeros(B, L, dtype=torch.bool, device=device)
    if left:
        valid[:, L - n_keep:] = True
    else:
        valid[:, :n_keep] = True
    key = valid.unsqueeze(1).unsqueeze(2)       # (B,1,1,L)
    qry = valid.unsqueeze(1).unsqueeze(-1)      # (B,1,L,1)
    return (key & qry).bool()


def module_args(seq_max, d_model=D_MODEL, n_layers=N_LAYERS, d_head=D_HEAD,
                dropout=0.05):
    return MyLMArgs(
        d_model=d_model,
        d_inner=int((d_model * (8 / 3) // 64) * 64),
        n_layers=n_layers,
        latent_moe=False,
        d_latent=d_model,
        vocab_size=VOCAB,
        seq_max_len=seq_max,
        use_moe=False,
        n_heads=d_model // d_head,
        d_head=d_head,
        dropout=dropout,
        attn_bias=True,
        base_init_std=0.02,
    )


# ----------------------------- stages -----------------------------

BACKENDS = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "memeff": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


def stage_probe(opt):
    print("\n" + "=" * 78)
    print("STAGE probe: SDPA 内核可用性 / 实际内核 / 数值一致性")
    print("=" * 78)
    print(f"GPU: {torch.cuda.get_device_name(0)} cc{torch.cuda.get_device_capability(0)}"
          f"  torch {torch.__version__} cuda {torch.version.cuda}")
    print("sdp flags: " + " ".join(
        f"{n}={f()}" for n, f in (
            ("cudnn", torch.backends.cuda.cudnn_sdp_enabled),
            ("flash", torch.backends.cuda.flash_sdp_enabled),
            ("memeff", torch.backends.cuda.mem_efficient_sdp_enabled),
            ("math", torch.backends.cuda.math_sdp_enabled))))
    B, H, L, D = 4, N_HEADS, 256, D_HEAD
    q, k, v = (torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    am = torch.ones(1, 1, L, L, dtype=torch.bool, device="cuda")
    print("\n-- 强制内核可用性 (bf16, head_dim=128) --")
    for name, be in BACKENDS.items():
        for tag, causal, m in (("causal", True, None), ("attn_mask", False, am)):
            try:
                with sdpa_kernel([be]):
                    F.scaled_dot_product_attention(
                        q, k, v, attn_mask=m, is_causal=causal)
                torch.cuda.synchronize()
                print(f"  {name:7s} {tag:9s} OK")
            except Exception as e:
                print(f"  {name:7s} {tag:9s} FAIL {type(e).__name__}: {str(e)[:70]}")

    print("\n-- profiler 实测单步(fwd+bwd, bf16 autocast)热 kernel, "
          f"L={L} B={B} --")
    qf, kf, vf = (torch.randn(B, H, L, D, device="cuda", requires_grad=True)
                  for _ in range(3))
    for mask_mode in ("none", "pad"):
        m = pad_mask(B, L, 0.75, torch.device("cuda")) if mask_mode == "pad" \
            else None
        for compiled in ((False, True) if opt.probe_compile else (False,)):
            for tag, mod in (("std", StdCore()), ("sdpa", SdpaCore())):
                if compiled:
                    mod = torch.compile(mod, mode=opt.compile_mode, dynamic=False)

                def one(mod=mod, m=m):
                    out = mod(qf, kf, vf, m)
                    out.float().pow(2).mean().backward()

                _top_kernels(f"{tag} {'compiled' if compiled else 'eager'}"
                             f" mask={mask_mode}", one)
                del mod
                cleanup()

    print("\n-- 数值一致性（dropout=0, L=256, B=2）--")
    # TF32 会让 std 的显式 matmul 与 cuDNN 融合内核以不同方式舍入，
    # 这里先关 TF32 验纯数学等价，再在 fp32/TF32/bf16-autocast 三口径下对比
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    for prec in ("fp32", "bf16-autocast"):
        for use_gate in (True, False):
            a = module_args(256, dropout=0.0)
            set_seed(42)
            std = Attention(a, use_gate=use_gate).cuda().float().train()
            sdp = SDPAAttention(a, use_gate=use_gate).cuda().float().train()
            sdp.load_state_dict(std.state_dict())
            x = torch.randn(2, 256, D_MODEL, device="cuda", requires_grad=True)
            res = {}
            for name, m in (("std", std), ("sdpa", sdp)):
                m.zero_grad(set_to_none=True)
                x.grad = None
                with (torch.autocast("cuda", dtype=torch.bfloat16)
                      if prec == "bf16-autocast" else contextlib.nullcontext()):
                    y = m(x)
                y.float().pow(2).mean().backward()
                res[name] = (y.detach().float(),
                             x.grad.detach().float().clone(),
                             m.q_proj.weight.grad.detach().float().clone())
            rel_o = ((res["std"][0] - res["sdpa"][0]).norm()
                     / res["std"][0].norm()).item()
            rel_g = ((res["std"][1] - res["sdpa"][1]).norm()
                     / res["std"][1].norm()).item()
            rel_w = ((res["std"][2] - res["sdpa"][2]).norm()
                     / res["std"][2].norm()).item()
            tag = "✓" if max(rel_o, rel_g, rel_w) < (2e-2 if prec != "fp32"
                                                     else 1e-5) else "✗"
            print(f"  {prec:13s} branch={'gate' if use_gate else 'sigmoid'}"
                  f"  out={rel_o:.2e} dx={rel_g:.2e} dW={rel_w:.2e}  {tag}")
    torch.backends.cuda.matmul.allow_tf32 = tf32

    print("\n-- 左 padding 全屏蔽行（SDPA 是否出 NaN）--")
    mask = pad_mask(2, 256, 0.6, torch.device("cuda"), left=True)
    set_seed(42)
    std = Attention(module_args(256, dropout=0.0), use_gate=False).cuda().float().train()
    sdp = SDPAAttention(module_args(256, dropout=0.0), use_gate=False).cuda().float().train()
    sdp.load_state_dict(std.state_dict())
    x = torch.randn(2, 256, D_MODEL, device="cuda", requires_grad=True)
    for name, m in (("std", std), ("sdpa", sdp)):
        m.zero_grad(set_to_none=True)
        y = m(x, mask=mask)
        y.pow(2).mean().backward()
        nan_out = torch.isnan(y).sum().item()
        nan_w = sum(int(torch.isnan(p.grad).sum()) for p in m.parameters()
                    if p.grad is not None)
        print(f"  {name:5s} 输出 NaN 元素={nan_out}  权重梯度 NaN 元素={nan_w}"
              + ("  ✓安全" if nan_out == 0 and nan_w == 0 else "  ✗NaN 污染"))
    del std, sdp, x
    cleanup()


def _build_core(impl, L, B, mask_mode, backend):
    mem0 = free_bytes()                       # 基线：本 case 什么都没分配
    if impl == "std":
        mod = StdCore()
    else:
        mod = SdpaCore(backend=BACKENDS.get(backend))
    mod = mod.cuda()
    set_seed(42)
    q, k, v = (torch.randn(B, N_HEADS, L, D_HEAD, device="cuda",
                           requires_grad=True) for _ in range(3))
    m = pad_mask(B, L, 0.75, torch.device("cuda")) if mask_mode == "pad" else None
    return Case(mod, (q, k, v, m), B * L, f"core/{impl}", mem0=mem0)


def stage_core(opt):
    print("\n" + "=" * 78)
    print("STAGE core: 只测注意力算子 (q,k,v 就绪)  fwd+bwd")
    print("=" * 78)
    rows = []
    for mask_mode in opt.masks:
        for L in opt.seqs:
            B = max(1, opt.core_tokens // L)
            # 注意力理论 GEMM 工作量：fwd = 4*B*H*L^2*D FLOPs（QK^T 与 PV 各 2 MAC->FLOP）
            # step(fwd+bwd) ≈ 3x fwd；"full"=算满 L^2，"causal"=只算下三角
            flops_full = 4 * B * N_HEADS * L * L * D_HEAD * 3
            plan = []
            if False in opt.compiles:
                plan += [("std", None, False), ("sdpa", None, False)]
                plan += [(f"sdpa-{b}", b, False) for b in opt.forced_backends]
            if True in opt.compiles and not (opt.skip_compiled_mask
                                             and mask_mode != "none"):
                plan += [("std", None, True), ("sdpa", None, True)]
            print(f"\n-- mask={mask_mode} L={L} B={B} (heads={N_HEADS},"
                  f" d_head={D_HEAD}) tokens={B*L}  理论 step: full="
                  f"{flops_full / 1e9:.1f} GFLOP causal≈{flops_full / 2e9:.1f} GFLOP --")
            print(f"{'impl':>12s} {'compile':>8s} {'ms_fp':>8s} {'ms_step':>9s}"
                  f" {'ktok/s':>8s} {'TF/s_full':>9s} {'step设备MiB':>11s}"
                  f" {'step分配器MiB':>12s}")
            for impl, backend, compiled in plan:
                if compiled and backend is not None:
                    continue
                mod = None
                t0 = time.perf_counter()
                try:
                    mod = _build_core(impl, L, B, mask_mode, backend)
                    if compiled:
                        mod.mod = torch.compile(mod.mod, mode=opt.compile_mode,
                                                dynamic=False)
                    r = mod.run(opt.warmup * 2 if compiled else opt.warmup,
                                opt.iters)
                    if opt.dump_kernels:
                        _top_kernels(f"core {impl} "
                                     f"{'compiled' if compiled else 'eager'}"
                                     f" mask={mask_mode} L={L}",
                                     mod.step, iters=4)
                except torch.OutOfMemoryError:
                    print(f"{impl:>12s} {str(compiled):>8s} {'OOM':>8s} {'OOM':>9s}",
                          flush=True)
                    rows.append(dict(stage="core", mask=mask_mode, seq=L, batch=B,
                                     impl=impl, compile=compiled, ms_step="OOM",
                                     step_peak_mib="OOM"))
                    continue
                finally:
                    wall = time.perf_counter() - t0
                    del mod
                    cleanup()
                r.update(stage="core", mask=mask_mode, seq=L, batch=B, impl=impl,
                         compile=compiled, wall_s=round(wall, 1),
                         tflops_full=flops_full / (r["ms_step"] / 1000) / 1e12)
                rows.append(r)
                print(f"{impl:>12s} {str(compiled):>8s} {r['ms_fp']:8.3f}"
                      f" {r['ms_step']:9.3f} {r['ktok_s']:8.2f}"
                      f" {r['tflops_full']:9.2f}"
                      f" {r['step_dev_mib']:11.1f} {r['step_peak_mib']:12.1f}"
                      + (f"  [{wall:.0f}s]" if compiled else ""), flush=True)
    return rows


def _build_module(impl, L, B, use_gate, mask_mode, compiled, mode):
    mem0 = free_bytes()
    args = module_args(L)
    set_seed(42)
    mod = (Attention(args, use_gate=use_gate) if impl == "std"
           else SDPAAttention(args, use_gate=use_gate))
    mod = mod.cuda().train()
    x = torch.randn(B, L, args.d_model, device="cuda")
    m = pad_mask(B, L, 0.75, torch.device("cuda")) if mask_mode == "pad" else None
    if compiled:
        mod = torch.compile(mod, mode=mode, dynamic=False)
    return Case(mod, (x, None, m), B * L, f"module/{impl}", mem0=mem0)


def stage_module(opt):
    print("\n" + "=" * 78)
    print("STAGE module: 完整 Attention 模块(含 QKV/O 投影+RoPE+QKNorm+门控) fwd+bwd")
    print("=" * 78)
    rows = []
    for mask_mode in opt.masks:
        for branch in opt.branches:
            for L in opt.seqs:
                B = max(1, TRAIN_TOKENS // L)
                print(f"\n-- mask={mask_mode} branch={branch} L={L} B={B} --")
                print(f"{'impl':>6s} {'compile':>8s} {'ms_fp':>8s} {'ms_step':>9s}"
                      f" {'ktok/s':>8s} {'step设备MiB':>11s} {'step分配器MiB':>12s}")
                for impl in ("std", "sdpa"):
                    for compiled in opt.compiles:
                        if compiled and mask_mode != "none" and opt.skip_compiled_mask:
                            continue
                        case = None
                        t0 = time.perf_counter()
                        try:
                            case = _build_module(impl, L, B, branch == "gate",
                                                 mask_mode, compiled,
                                                 opt.compile_mode)
                            r = case.run(opt.warmup * 2 if compiled else opt.warmup,
                                         opt.iters)
                            if opt.dump_kernels:
                                _top_kernels(f"module {impl} branch={branch} "
                                             f"{'compiled' if compiled else 'eager'}"
                                             f" mask={mask_mode} L={L}",
                                             case.step, iters=3)
                        except torch.OutOfMemoryError:
                            print(f"{impl:>6s} {str(compiled):>8s} {'OOM':>8s}"
                                  f" {'OOM':>9s}", flush=True)
                            rows.append(dict(stage="module", mask=mask_mode,
                                             branch=branch, seq=L, batch=B,
                                             impl=impl, compile=compiled,
                                             ms_step="OOM", step_peak_mib="OOM"))
                            continue
                        except Exception as e:
                            print(f"{impl:>6s} {str(compiled):>8s} ERR "
                                  f"{type(e).__name__}: {str(e)[:60]}", flush=True)
                            continue
                        finally:
                            compile_s = time.perf_counter() - t0
                            del case
                            cleanup()
                        r.update(stage="module", mask=mask_mode, branch=branch,
                                 seq=L, batch=B, impl=impl, compile=compiled,
                                 wall_s=round(compile_s, 1))
                        rows.append(r)
                        print(f"{impl:>6s} {str(compiled):>8s} {r['ms_fp']:8.3f}"
                              f" {r['ms_step']:9.3f} {r['ktok_s']:8.2f}"
                              f" {r['step_dev_mib']:11.1f}"
                              f" {r['step_peak_mib']:12.1f}"
                              + (f"  [{compile_s:.0f}s]" if compiled else ""),
                              flush=True)
    return rows


def _dev(model):
    return next(model.parameters()).device


class DevPeakSampler:
    """后台线程轮询 mem_get_info，抓 step **内部** 的设备级显存峰值。

    必须这样做有两个原因：
    1. 峰值出现在 step 执行期间，step 结束（反向激活释放）后再采样必然读到 ~0；
    2. inductor / CUDA Graph 的私有内存池不进 torch 分配器统计，
       max_memory_allocated 看不到，只有设备级 mem_get_info 才公平。
    轮询只发生在显存轮，不污染计时轮。
    """

    def __init__(self, mem0_bytes: int, dev=0, interval=0.001):
        self.mem0 = mem0_bytes
        self.dev = dev
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._th = None

    def _loop(self):
        while not self._stop.is_set():
            used = self.mem0 - torch.cuda.mem_get_info(self.dev)[0]
            if used > self.peak:
                self.peak = used
            self._stop.wait(self.interval)

    def __enter__(self):
        torch.cuda.synchronize()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self._stop.set()
        self._th.join()
        self.peak = max(self.peak,
                        self.mem0 - torch.cuda.mem_get_info(self.dev)[0])
        return False

    @property
    def mib(self):
        return self.peak / 2**20


def _make_model(args, use_sdpa, arch):
    """构建两种实现共用的模型（同一 seed -> 权重逐位相同）。

    arch="prod" : 保留 MyLM 奇数层的 CompressedAttention（分数矩阵 S x S/4），
                  只把偶数层 Attention 换成 SDPA —— 反映真实架构里的收益稀释。
    arch="dense": 奇数层也换成标准 Attention，两种实现全层对齐 —— 隔离出纯效应。
    """
    set_seed(42)
    model = MyLM(args).cuda().train()
    for i, blk in enumerate(model.blocks):
        src = blk.attn
        if not isinstance(src, Attention):
            if arch != "dense":
                continue
            src = Attention(args, use_gate=False).to(_dev(model))
            blk.attn = src
        if use_sdpa:
            dst = SDPAAttention(args, use_gate=src.use_gate).to(_dev(model))
            dst.load_state_dict(src.state_dict())   # 权重与 RoPE 缓存原样复制
            blk.attn = dst
    return model


def _build_model(use_sdpa, compiled, mode, L, B, mask_mode="none", arch="prod"):
    mem0 = free_bytes()
    args = module_args(L)
    model = _make_model(args, use_sdpa, arch)
    if compiled:
        model = torch.compile(model, mode=mode, dynamic=False)
    ids = torch.randint(1, args.vocab_size, (B, L), device="cuda")
    tgt = torch.randint(1, args.vocab_size, (B, L), device="cuda")
    pm = None
    if mask_mode == "pad":
        # 生产口径：右 pad，尾部 0~30% 置为 pad_id=0（同 generate_dataset_v3 的短句）
        n = (L * (0.7 + 0.3 * torch.rand(B, device="cuda"))).long().clamp(1, L)
        ar = torch.arange(L, device="cuda").unsqueeze(0)
        pad_at = ar >= n.unsqueeze(1)
        ids = ids.masked_fill(pad_at, 0)
        pm = ids != 0
        tgt = tgt.masked_fill(pad_at, -100)

    class ModelCase(Case):
        def fwd(self):
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                out = self.mod(ids, padding_mask=pm)
                return F.cross_entropy(out.view(-1, args.vocab_size),
                                       tgt.view(-1))

        def step(self, sample=False):
            self.zero()
            self.fwd().backward()
            if sample:
                self.mem_probe = self.dev_used()

    return ModelCase(model, (ids, tgt, pm), B * L, "model", mem0=mem0)


def stage_model(opt):
    print("\n" + "=" * 78)
    print("STAGE model: 整模型 MyLM(dense 6层, d=512) 真实训练口径 L=256 B=64")
    print("=" * 78)
    rows = []
    L = opt.model_seq
    B = opt.model_batch
    print(f"{'impl':>6s} {'mask':>5s} {'arch':>6s} {'compile':>8s} {'ms_step':>9s}"
          f" {'ktok/s':>8s} {'step设备MiB':>11s} {'step分配器MiB':>12s} {'耗时':>6s}")
    for mask_mode in opt.masks:
        for arch in opt.model_arch.split(","):
            for impl, use_sdpa in (("std", False), ("sdpa", True)):
                for compiled in opt.compiles:
                    case = None
                    t0 = time.perf_counter()
                    try:
                        case = _build_model(use_sdpa, compiled,
                                            opt.compile_mode, L, B, mask_mode,
                                            arch)
                        r = case.run(opt.warmup * 2 if compiled else opt.warmup,
                                     max(6, opt.iters))
                        if opt.dump_kernels:
                            _top_kernels(f"model {impl} arch={arch} "
                                         f"{'compiled' if compiled else 'eager'}"
                                         f" mask={mask_mode}",
                                         case.step, iters=2)
                    except torch.OutOfMemoryError:
                        print(f"{impl:>6s} {mask_mode:>5s} {arch:>6s} "
                              f"{str(compiled):>8s} OOM", flush=True)
                        continue
                    finally:
                        wall = time.perf_counter() - t0
                        del case
                        cleanup()
                    r.update(stage="model", mask=mask_mode, branch=arch, seq=L,
                             batch=B, impl=impl, compile=compiled,
                             wall_s=round(wall, 1))
                    rows.append(r)
                    print(f"{impl:>6s} {mask_mode:>5s} {arch:>6s}"
                          f" {str(compiled):>8s} {r['ms_step']:9.3f}"
                          f" {r['ktok_s']:8.2f} {r['step_dev_mib']:11.1f}"
                          f" {r['step_peak_mib']:12.1f}"
                          f" {wall:6.0f}s", flush=True)
    return rows


def stage_capacity(opt):
    """同显存能开多大 batch：显存收益换算成吞吐收益。"""
    print("\n" + "=" * 78)
    print("STAGE capacity: 逐次翻倍 batch 直到 OOM（module 级, mask=pad）")
    print("=" * 78)
    rows = []
    for L in opt.capacity_seqs:
        for compiled in ([False, True] if opt.capacity_compiled else [False]):
            for impl in ("std", "sdpa"):
                best = None
                B = opt.capacity_start_batch
                while B <= opt.capacity_max_batch:
                    case = None
                    try:
                        case = _build_module(impl, L, B, False, "pad", compiled,
                                             opt.compile_mode)
                        r = case.run(opt.warmup if not compiled else opt.warmup * 2,
                                     max(4, opt.iters // 2))
                        best = (B, r)
                    except torch.OutOfMemoryError:
                        break
                    except Exception as e:
                        print(f"  [note] {impl} L={L} B={B}: "
                              f"{type(e).__name__}: {str(e)[:60]}", flush=True)
                        break
                    finally:
                        del case
                        cleanup()
                    B *= 2
                if best is None:
                    print(f"  L={L} compile={compiled} {impl}: 起始 batch 即 OOM")
                    continue
                Bm, r = best
                r.update(stage="capacity", mask="pad", branch="gate", seq=L,
                         batch=Bm, impl=impl, compile=compiled)
                rows.append(r)
                print(f"  L={L:5d} compile={str(compiled):5s} {impl:>5s}: "
                      f"最大 batch={Bm:4d} tokens={Bm*L:>7d}  "
                      f"ms_step={r['ms_step']:8.2f} ktok/s={r['ktok_s']:6.2f} "
                      f"设备峰值={r['step_dev_mib']:8.1f} MiB", flush=True)
    return rows


# ----------------------------- 汇总 -----------------------------

def summarize(rows, out_md):
    """按 (stage, mask, seq, branch/impl 组) 输出 std vs sdpa 的加速比与显存节省。"""
    lines = ["\n## 汇总（以 std 为基准，>1 表示 sdpa 更快/更省）\n"]
    groups = {}
    for r in rows:
        if r.get("ms_step") == "OOM":
            continue
        key = (r["stage"], r.get("mask", ""), r.get("branch", ""), r["seq"],
               r["compile"])
        groups.setdefault(key, {})[r["impl"].replace("sdpa-", "sdpa_")] = r
    lines.append("| stage | mask | branch | seq | compile | 对比 | ms_step 比 | ktok/s 比 | 设备显存比 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, d in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                         str(kv[0][2]), kv[0][3],
                                                         str(kv[0][4]))):
        stage, mask, branch, seq, compiled = key
        base = d.get("std")
        for name, r in d.items():
            if name == "std" or base is None:
                continue
            lines.append(
                f"| {stage} | {mask} | {branch} | {seq} | {compiled} | {name}/std "
                f"| {base['ms_step'] / r['ms_step']:.2f}x "
                f"| {r['ktok_s'] / base['ktok_s']:.2f}x "
                f"| {r.get('step_dev_mib', r['step_peak_mib']) / base.get('step_dev_mib', base['step_peak_mib']):.2f} |")
        if "capacity" == stage and "std" in d and "sdpa" in d:
            lines.append(f"| {stage} | {mask} | {branch} | {seq} | {compiled} "
                         f"| 最大 batch sdpa/std | - | - | "
                         f"{d['sdpa']['batch'] / d['std']['batch']:.2f}x |")
    text = "\n".join(lines)
    print(text)
    out_md.write_text(text, encoding="utf-8")


def _num(v):
    return f"{v:.4g}" if isinstance(v, float) else v


def write_csv(rows, out_csv):
    if not rows:
        return
    cols = ["stage", "mask", "branch", "seq", "batch", "impl", "compile",
            "ms_fp", "ms_step", "ms_step_mean", "ms_step_std", "ktok_s",
            "step_dev_mib", "step_peak_mib", "step_reserved_mib",
            "fp_dev_mib", "fp_peak_mib", "wall_s"]
    keys = set()
    for r in rows:
        keys |= set(r)
    use = [c for c in cols if c in keys] + sorted(keys - set(cols))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=use)
        w.writeheader()
        for r in rows:
            w.writerow({k: _num(r.get(k, "")) for k in use})
    print(f"\n结果已保存: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="probe,core,module")
    ap.add_argument("--seqs", default="256,512,1024")
    ap.add_argument("--masks", default="none", help="none,pad（pad=4D padding mask）")
    ap.add_argument("--branches", default="gate,sigmoid")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--compile-mode", default="max-autotune")
    ap.add_argument("--core-tokens", type=int, default=8192)
    ap.add_argument("--forced-backends", default="",
                    help="eager 下额外强制内核列表，如 cudnn,memeff,math")
    ap.add_argument("--skip-compiled-mask", action="store_true")
    ap.add_argument("--probe-compile", action="store_true",
                    help="probe 阶段额外 profile 编译后的实际 kernel")
    ap.add_argument("--compilations", default="eager,compiled",
                    help="eager,compiled 两类计时场景选择")
    ap.add_argument("--model-seq", type=int, default=256)
    ap.add_argument("--model-batch", type=int, default=64)
    ap.add_argument("--model-arch", default="prod,dense",
                    help="prod=保留奇数层 CompressedAttention；dense=全层标准注意力")
    ap.add_argument("--capacity-seqs", default="256,1024")
    ap.add_argument("--capacity-start-batch", type=int, default=8)
    ap.add_argument("--capacity-compiled", action="store_true",
                    help="capacity 阶段也测编译版（每个 batch 都会重编译，很慢）")
    ap.add_argument("--capacity-max-batch", type=int, default=512)
    ap.add_argument("--dump-kernels", action="store_true",
                    help="每组额外用 profiler 打印实际执行的 top kernel（定位 SDPA 到底选了哪个后端）")
    ap.add_argument("--tag", default="")
    opt = ap.parse_args()

    opt.seqs = [int(s) for s in opt.seqs.split(",")]
    opt.masks = [s for s in opt.masks.split(",")]
    opt.branches = [s for s in opt.branches.split(",")]
    opt.forced_backends = [s for s in opt.forced_backends.split(",") if s]
    opt.capacity_seqs = [int(s) for s in opt.capacity_seqs.split(",")]
    opt.compiles = sorted({c == "compiled"
                           for c in opt.compilations.split(",") if c})

    torch.set_float32_matmul_precision("high")   # 对齐 PreTrainer.__init__
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"attn_impl_bench_{(opt.tag + '_') if opt.tag else ''}{ts}"
    out_csv = RESULTS / f"{stem}.csv"
    out_md = RESULTS / f"{stem}.md"

    stages = opt.stages.split(",")
    if "probe" in stages:
        stage_probe(opt)
    rows = []
    for st in ("core", "module", "model", "capacity"):
        if st in stages:
            rows += globals()[f"stage_{st}"](opt)
    if rows:
        write_csv(rows, out_csv)
        summarize(rows, out_md)


if __name__ == "__main__":
    main()
