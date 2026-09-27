# -*- coding: utf-8 -*-
"""SWA（滑动窗口注意力）三种实现的速度 / 显存 / 数值对比。

要回答的问题：在这个仓库（d_head=128, n_heads=4, bf16 autocast, fwd+bwd）里，
滑动窗口注意力怎么实现才好？

三种做法都严格实现"只看最近 W 个 key（含自身）"的因果窗口：
1. band-std         手写全量 q@k.T + 带状 0/1 mask。相当于只把现有 Attention 的
                    tril 换成 band：分数矩阵 O(L^2) 全物化，窗口外的分数照样算完
                    再丢，计算量仍是 O(L^2)。
2. band-sdpa        F.scaled_dot_product_attention + 稠密带状 attn_mask。
                    分数不物化（融合内核省显存），但内核不知道 mask 是带状的，
                    计算量仍是 O(L^2) —— 时间上拿不到 SWA 的理论收益。
3. flex             torch.compile(flex_attention) + create_block_mask(mask_mod)。
                    mask 以**函数**给出，块级跳过窗口外的 KV tile，
                    是唯一把计算量真正降到 O(L*W) 的实现；pad 也不用物化 bool。

背景：L=256（当前 SENTENCE_MAXLEN）时 W>=L，三者全是 no-op，所以本实验只做
L>=1024。B*L 固定 8192 token，使不同长度的耗时可直接横比。

用法:
  $env:PYTHONUTF8='1'; uv run python experiments/swa_bench.py
  $env:PYTHONUTF8='1'; uv run python experiments/swa_bench.py --seqs 2048 --dump-kernels
结果写 experiments/results/swa_bench_<tag>_<ts>.csv
"""
import contextlib
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import argparse
import csv
import datetime
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# 复用 attn_impl_bench 的计时 / 显存采样 / 清理口径，保证两份实验可直接对比
from attn_impl_bench import (D_HEAD, N_HEADS, Case, _top_kernels, cleanup,
                            free_bytes)

RESULTS = ROOT / "experiments" / "results"
CORE_TOKENS = 8192
class FlexScoreMod(nn.Module):
    """生产可落地的形态：BlockMask 只表达**静态**结构（窗口+因果），整次训练复用；
    动态的 pad 走 score_mod。

    动机是实测：BlockMask 若含 pad 就必须每个 batch 重建，create_block_mask 要
    5.9ms（首次含 JIT 96ms），而单步 flex 只要 2.0-3.1ms —— 重建比计算还贵。
    静态结构只依赖 (L, W)，可以在模型初始化时算一次缓存住。

    inputs[3] 传的是 (B,L) 的 key-valid bool（而不是 (B,1,L,L)），
    因为 pad 在这里只需要 key 维。
    """

    def __init__(self, W, compiled=True, sink=0, dev="cuda"):
        super().__init__()
        self.W = W
        self.sink = sink
        self.dev = dev
        self.fn = (torch.compile(flex_attention, dynamic=False)
                   if compiled else flex_attention)
        self.bm = None
        self.build_ms = float("nan")

    def prepare(self, L):
        """只依赖 (L, W, sink)，与 batch 内容无关 -> 可缓存复用。"""
        if self.bm is not None and self.bm._q_length == L:
            return
        W, sink = self.W, self.sink

        def window(b, h, qi, ki):
            delta = qi - ki
            ok = (delta >= 0) & (delta < W)
            return ok | (ki < sink) if sink else ok

        t0 = time.perf_counter()
        self.bm = create_block_mask(window, B=None, H=None, Q_LEN=L, KV_LEN=L,
                                    device=self.dev)
        torch.cuda.synchronize()
        self.build_ms = (time.perf_counter() - t0) * 1000

    def forward(self, q, k, v, valid=None):
        if valid is None:
            return self.fn(q, k, v, block_mask=self.bm)

        def score_mod(score, b, h, qi, ki):
            return torch.where(valid[b, ki], score, float("-inf"))

        return self.fn(q, k, v, score_mod=score_mod, block_mask=self.bm)


IMPLS = ("band-std", "band-sdpa", "band-sdpa-cudnn", "flex", "flex-sm")


def band_mask(L, W, device, sink=0):
    """因果带状稠密 mask：True = 可见（i-W < j <= i）。

    sink>0 时额外放开前 sink 个 key（StreamingLLM 式 attention sink）：
    纯滑窗下第 0 个 token 只能看自己、窗口左半边的 token 上下文极短，
    留几个"永远可见"的全局 key 可以补上这个缺陷，代价是每行多 sink 列。
    """
    i = torch.arange(L, device=device).unsqueeze(1)
    j = torch.arange(L, device=device).unsqueeze(0)
    ok = ((i - j) >= 0) & ((i - j) < W)
    if sink:
        ok = ok | (j < sink)      # 广播到 (L,L)：前 sink 列永远可见
    return ok


class BandStd(nn.Module):
    """手写全量分数 + 带状 mask（口径同 models.py:Attention，含二次 masked_fill）。"""

    def __init__(self, W, sink=0):
        super().__init__()
        self.W = W
        self.sink = sink

    def forward(self, q, k, v, mask=None):
        L = q.size(-2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        combined = band_mask(L, self.W, q.device, self.sink).view(1, 1, L, L)
        if mask is not None:
            combined = combined & mask
        att = att.masked_fill(combined == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(combined == 0, 0.0)
        return att @ v


class BandSdpa(nn.Module):
    """SDPA + 稠密带状 attn_mask。backend=None 让 torch 自己选。"""

    def __init__(self, W, backend=None, sink=0):
        super().__init__()
        self.W = W
        self.backend = backend
        self.sink = sink

    def forward(self, q, k, v, mask=None):
        L = q.size(-2)
        attn_mask = band_mask(L, self.W, q.device, self.sink).view(1, 1, L, L)
        if mask is not None:
            attn_mask = attn_mask & mask
        ctx = (sdpa_kernel([self.backend]) if self.backend is not None
               else contextlib.nullcontext())
        with ctx:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False,
                scale=1.0 / math.sqrt(q.size(-1)))


class FlexCore(nn.Module):
    """flex_attention + BlockMask：窗口与 pad 都以函数形式给出。

    注意：create_block_mask 拒绝接收 bound method（"requires a mask_mod function"），
    必须传普通闭包，所以下面用 _make_mask_mod 造闭包而不是 self._mask_mod。
    """

    def __init__(self, W, compiled=True, valid_mask=None, B=None, dev="cuda",
                 sink=0):
        super().__init__()
        self.W = W
        self.fn = (torch.compile(flex_attention, dynamic=False)
                   if compiled else flex_attention)
        self.valid_mask = valid_mask
        self.B = B
        self.dev = dev
        self.sink = sink
        self.bm = None
        self.build_ms = float("nan")
        self.mask_mod = self._make_mask_mod(W, valid_mask, sink)

    @staticmethod
    def _make_mask_mod(W, valid_mask, sink=0):
        def window(b, h, qi, ki):
            delta = qi - ki
            ok = (delta >= 0) & (delta < W)
            if sink:
                ok = ok | (ki < sink)   # attention sink：前 sink 个 key 永远可见
            if valid_mask is not None:
                ok = ok & valid_mask[b, ki]   # pad 位不可见（左右 pad 通吃）
            return ok
        return window

    def prepare(self, L):
        t0 = time.perf_counter()
        self.bm = create_block_mask(self.mask_mod,
                                    B=(None if self.valid_mask is None else self.B),
                                    H=None, Q_LEN=L, KV_LEN=L, device=self.dev)
        torch.cuda.synchronize()
        self.build_ms = (time.perf_counter() - t0) * 1000

    def forward(self, q, k, v, mask=None):
        return self.fn(q, k, v, block_mask=self.bm)


def build(impl, L, B, W, mask_mode, compiled, sink=0, pad_left=False):
    """构造一个待测 case。返回 (Case, module)。"""
    mem0 = free_bytes()
    dev = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    q, k, v = (torch.randn(B, N_HEADS, L, D_HEAD, device=dev,
                           dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    m = None
    valid = None
    if mask_mode == "pad":
        n_keep = max(1, int(L * 0.75))
        valid = torch.zeros(B, L, dtype=torch.bool, device=dev)
        if pad_left:
            valid[:, L - n_keep:] = True       # 左 pad（SFT 批次的常见形态）
        else:
            valid[:, :n_keep] = True           # 右 pad（pretrain 的形态）
        m = valid.unsqueeze(1).unsqueeze(2) & valid.unsqueeze(1).unsqueeze(-1)

    if impl == "band-std":
        mod = BandStd(W, sink=sink)
    elif impl == "band-sdpa":
        mod = BandSdpa(W, sink=sink)
    elif impl == "band-sdpa-cudnn":
        mod = BandSdpa(W, backend=SDPBackend.CUDNN_ATTENTION, sink=sink)
    elif impl == "flex":
        mod = FlexCore(W, compiled=compiled, valid_mask=valid, B=B, sink=sink)
        mod.prepare(L)
        m = None                      # pad 已在 mask_mod 内处理，不需要稠密 mask
    elif impl == "flex-sm":
        mod = FlexScoreMod(W, compiled=compiled, sink=sink)
        mod.prepare(L)                # 静态结构，一次构建后复用
        m = valid                     # pad 交给 score_mod：传 (B,L) key-valid
    else:
        raise ValueError(impl)
    if compiled and impl not in ("flex", "flex-sm"):
        mod = torch.compile(mod, mode="max-autotune", dynamic=False)
    return Case(mod, (q, k, v, m), B * L, f"swa/{impl}", mem0=mem0), mod
def nan_check(case):
    """跑一次 fwd+bwd 数 NaN。左 pad 时窗口内可能一个有效 key 都没有，
    softmax 全 -inf 会产出 NaN 并污染梯度——这是 SWA 落地最容易踩的正确性坑。"""
    case.prealloc()
    case.zero()
    out = case.fwd()
    no = int(torch.isnan(out).sum())
    out.float().pow(2).mean().backward()
    nd = sum(int(torch.isnan(p.grad).sum()) for p in case.grad_objs()
             if p.grad is not None)
    del out
    return no, nd


def numeric_check(W):
    """三种实现与 band-std 的数值一致性。

    fp32 一轮（严格对拍纯数学等价，cuDNN 不支持 fp32 故跳过并注明），
    bf16 一轮（生产实际精度，容差放宽）。
    """
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    L, B = 1024, 2
    for prec, tol, dtype in (("fp32", 2e-5, torch.float32),
                             ("bf16-autocast", 2e-2, torch.float32)):
        print("\n" + "=" * 78)
        print(f"数值一致性：以 band-std 为参考（{prec}, L={L} W={W}）")
        print("=" * 78)
        ref = None
        for impl in IMPLS:
            if impl == "band-sdpa-cudnn" and prec == "fp32":
                print(f"  {impl:16s} 跳过（cuDNN attention 只接受 fp16/bf16，"
                      f"fp32 下会 No available kernel）")
                continue
            torch.manual_seed(0)
            q, k, v = (torch.randn(B, N_HEADS, L, D_HEAD, device="cuda",
                                   dtype=dtype, requires_grad=True)
                       for _ in range(3))
            if impl == "band-std":
                mod = BandStd(W)
            elif impl == "band-sdpa":
                mod = BandSdpa(W)
            elif impl == "band-sdpa-cudnn":
                mod = BandSdpa(W, backend=SDPBackend.CUDNN_ATTENTION)
            elif impl == "flex-sm":
                mod = FlexScoreMod(W, compiled=True)
                mod.prepare(L)
            else:
                mod = FlexCore(W, compiled=True)
                mod.prepare(L)
            try:
                if prec == "bf16-autocast":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = mod(q, k, v, None)
                else:
                    out = mod(q, k, v, None)
                out.float().pow(2).mean().backward()
            except Exception as e:
                print(f"  {impl:16s} FAIL {type(e).__name__}: {str(e)[:60]}")
                del q, k, v, mod
                cleanup()
                continue
            cur = (out.detach().float(), q.grad.detach().float().clone())
            if ref is None:
                ref = cur
                print(f"  {impl:16s} 参考")
            else:
                ro = ((cur[0] - ref[0]).norm() / ref[0].norm()).item()
                rg = ((cur[1] - ref[1]).norm() / ref[1].norm()).item()
                print(f"  {impl:16s} out={ro:.2e} dq={rg:.2e}  "
                      + ("✓" if max(ro, rg) < tol else "✗(超容差)"))
            del q, k, v, mod, out, cur
            cleanup()
    torch.backends.cuda.matmul.allow_tf32 = tf32


def pad_equivalence_check(W):
    """pad 场景的等价性：静态 BlockMask + score_mod-pad 是否等于稠密 4D mask。

    分别测右 pad（pretrain）与左 pad（SFT），在**有效 query 行**上比输出，
    并统计 out / dq 的 NaN。左 pad 会制造"窗口内一个有效 key 都没有"的行，
    这类行是否出 NaN 决定实现里要不要加保护。
    """
    print("\n" + "=" * 78)
    print(f"pad 等价性检查（bf16 autocast, L=2048 W={W}）")
    print("=" * 78)
    L, B = 2048, 4
    for side in ("right", "left"):
        torch.manual_seed(7)
        q, k, v = (torch.randn(B, N_HEADS, L, D_HEAD, device="cuda",
                              dtype=torch.bfloat16, requires_grad=True)
                   for _ in range(3))
        n_keep = int(L * 0.75)
        valid = torch.zeros(B, L, dtype=torch.bool, device="cuda")
        if side == "left":
            valid[:, L - n_keep:] = True
        else:
            valid[:, :n_keep] = True
        dense = valid.unsqueeze(1).unsqueeze(2) & valid.unsqueeze(1).unsqueeze(-1)
        res = {}
        for impl, mk in (("band-std", lambda: BandStd(W)),
                         ("flex", lambda: FlexCore(W, compiled=True,
                                                   valid_mask=valid, B=B)),
                         ("flex-sm", lambda: FlexScoreMod(W, compiled=True))):
            mod = mk()
            if impl != "band-std":
                mod.prepare(L)
            inp = (q, k, v, None if impl == "flex" else
                   (dense if impl == "band-std" else valid))
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = mod(*inp)
                out.float().pow(2).mean().backward()
                res[impl] = (out.detach().float(), q.grad.detach().float().clone())
            except Exception as e:
                print(f"  {impl:10s} FAIL {type(e).__name__}: {str(e)[:60]}")
                continue
            q.grad = None
        if "band-std" not in res or len(res) < 2:
            continue
        ref = res["band-std"]
        rows_ok = valid                       # (B,L) 有效 query 位置
        for impl, cur in res.items():
            if impl == "band-std":
                continue
            d = (cur[0] - ref[0]).abs()
            dv = d[:, :, :, :].permute(0, 2, 1, 3).reshape(B, L, -1)
            valid_err = dv[rows_ok].max().item()
            nan_o = int(torch.isnan(cur[0]).sum())
            nan_g = int(torch.isnan(cur[1]).sum())
            print(f"  pad={side:5s} {impl:8s} 有效行最大绝对差={valid_err:.2e}"
                  f"  NaN(out)={nan_o} NaN(dq)={nan_g}"
                  + ("  ✓" if valid_err < 5e-2 and nan_o == 0 and nan_g == 0
                     else "  ✗"))
        del q, k, v, res, dense, valid
        cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="1024,2048,4096")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--masks", default="none,pad")
    ap.add_argument("--compilations", default="eager,compiled")
    ap.add_argument("--impls", default=",".join(IMPLS))
    ap.add_argument("--sink", type=int, default=0,
                    help="额外让前 sink 个 key 永远可见（attention sink）")
    ap.add_argument("--pad-side", default="right", choices=("right", "left"),
                    help="pad 方向；left 会制造窗口内无有效 key 的行，用于查 NaN")
    ap.add_argument("--windows", default="",
                    help="窗口扫描，如 64,256,1024；留空则用 --window")
    ap.add_argument("--skip-numeric", action="store_true")
    ap.add_argument("--dump-kernels", action="store_true")
    ap.add_argument("--tag", default="")
    opt = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    opt.seqs = [int(s) for s in opt.seqs.split(",")]
    opt.masks = opt.masks.split(",")
    opt.impls = [s for s in opt.impls.split(",") if s]
    opt.compiles = sorted({c == "compiled"
                           for c in opt.compilations.split(",") if c})
    opt.windows = ([int(x) for x in opt.windows.split(",")] if opt.windows
                   else [opt.window])
    opt.pad_left = (opt.pad_side == "left")
    RESULTS.mkdir(parents=True, exist_ok=True)

    if not opt.skip_numeric:
        numeric_check(opt.windows[0])
        pad_equivalence_check(opt.windows[0])

    rows = []
    print("\n" + "=" * 78)
    print(f"STAGE swa-core: W={opt.windows}, B*L={CORE_TOKENS} token 固定, "
          f"fwd+bwd, bf16 autocast, sink={opt.sink}, pad_side={opt.pad_side}")
    print("=" * 78)
    for mask_mode in opt.masks:
        for W in opt.windows:
            for L in opt.seqs:
                B = max(1, CORE_TOKENS // L)
                vis_band = L * min(L, W)
                vis_causal = L * (L + 1) // 2
                print(f"\n-- mask={mask_mode} W={W} L={L} B={B}  "
                      f"窗口可见量/因果可见量 = {vis_band / vis_causal:.2f} --")
                print(f"{'impl':>16s} {'compile':>8s} {'ms_fp':>8s}"
                      f" {'ms_step':>9s} {'ktok/s':>8s} {'step设备MiB':>11s}"
                      f" {'分配器MiB':>10s} {'NaN':>6s}")
                for impl in opt.impls:
                    for compiled in opt.compiles:
                        if impl in ("flex", "flex-sm") and not compiled:
                            continue  # eager flex 是刻意物化的退化路径
                        case = None
                        t0 = time.perf_counter()
                        try:
                            case, mod = build(impl, L, B, W, mask_mode,
                                              compiled, sink=opt.sink,
                                              pad_left=opt.pad_left)
                            build_ms = getattr(mod, "build_ms", None)
                            r = case.run(opt.warmup * 2 if compiled
                                         else opt.warmup, opt.iters)
                            r["nan_out"], r["nan_dq"] = nan_check(case)
                            if build_ms:
                                r["blockmask_build_ms"] = build_ms
                            if opt.dump_kernels:
                                _top_kernels(f"{impl} {'cmp' if compiled else 'egr'}"
                                             f" L={L}", case.step, iters=3)
                        except torch.OutOfMemoryError:
                            print(f"{impl:>16s} {str(compiled):>8s} OOM",
                                  flush=True)
                            rows.append(dict(stage="swa", mask=mask_mode,
                                             seq=L, batch=B, impl=impl,
                                             window=W, sink=opt.sink,
                                             compile=compiled, ms_step="OOM",
                                             step_peak_mib="OOM"))
                            continue
                        except Exception as e:
                            print(f"{impl:>16s} {str(compiled):>8s} ERR "
                                  f"{type(e).__name__}: {str(e)[:70]}",
                                  flush=True)
                            continue
                        finally:
                            wall = time.perf_counter() - t0
                            del case
                            cleanup()
                        r.update(stage="swa", mask=mask_mode, seq=L, batch=B,
                                 impl=impl, window=W, sink=opt.sink,
                                 pad_side=opt.pad_side, compile=compiled,
                                 wall_s=round(wall, 1))
                        rows.append(r)
                        print(f"{impl:>16s} {str(compiled):>8s}"
                              f" {r['ms_fp']:8.3f} {r['ms_step']:9.3f}"
                              f" {r['ktok_s']:8.2f} {r['step_dev_mib']:11.1f}"
                              f" {r['step_peak_mib']:10.1f}"
                              f" {r['nan_out'] + r['nan_dq']:6d}"
                              + (f"  [{wall:.0f}s]" if compiled else ""),
                              flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"swa_bench_{(opt.tag + '_') if opt.tag else ''}{ts}.csv"
    cols = ["stage", "mask", "seq", "window", "batch", "impl", "compile",
            "ms_fp", "ms_step", "ktok_s", "step_dev_mib", "step_peak_mib",
            "blockmask_build_ms", "wall_s"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\n结果已保存: {out}")

    print("\n## 汇总（以 band-std 为基线，>1 表示更快 / 更省）\n")
    print(f"{'mask':>5s} {'seq':>6s} {'win':>6s} {'compile':>8s} {'impl':>16s} "
          f"{'单步加速':>9s} {'显存节省':>9s}")
    idx = {}
    for r in rows:
        if r.get("ms_step") == "OOM":
            continue
        idx.setdefault((r["mask"], int(r["seq"]), int(r["window"]),
                        r["compile"]), {})[r["impl"]] = r
    for key in sorted(idx, key=lambda k: (k[0], k[2], k[1], k[3])):
        mask, seq, win, compiled = key
        d = idx[key]
        base = d.get("band-std")
        if base is None:
            continue
        for impl, r in d.items():
            if impl == "band-std":
                continue
            print(f"{mask:>5s} {seq:6d} {win:6d} {str(compiled):>8s} {impl:>16s} "
                  f"{base['ms_step'] / r['ms_step']:8.2f}x "
                  f"{base['step_dev_mib'] / r['step_dev_mib']:8.2f}x"
                  f"  NaN={r.get('nan_out', 0) + r.get('nan_dq', 0)}")


if __name__ == "__main__":
    main()
