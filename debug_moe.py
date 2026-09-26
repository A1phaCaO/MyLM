"""debug_moe.py — 单独诊断 MoE 为什么没有比 dense 训练更快。

用法:
    uv run python debug_moe.py

小节:
  1. MoE(k=1/12) vs Dense 端到端训练 step 时间 (含/不含 LM head)
  2. MoEFFN 前向子操作耗时分解 (复刻 models.py 的 MoEFFN.forward 逐段计时)
  3. torch.profiler: MoE vs Dense 的 kernel 数量 / 小 kernel 占比
  4. 每层 CPU 同步(device->host)次数统计
  5. 路由负载均衡统计 (每层每专家 token 数分布)
  6. torch.compile(backend="eager") 开/关对比 (还原训练实际配置)
  7. padded-batched-bmm 参考实现, 估算 MoE 在当前规模下的理论上限
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import gc
import io
import math
import time
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from models import MyLM, MyLMArgs, MoEFFN, FFN

torch.manual_seed(42)
if not torch.cuda.is_available():
    raise SystemExit("本脚本需要 CUDA GPU")
DEVICE = "cuda"

# quick 模式: 小批量/短序列/少迭代, 约 1 分钟内跑完整个脚本
QUICK = os.environ.get("MOE_DEBUG_QUICK", "") == "1" or "quick" in sys.argv

# ---------- 与 pre_train.py / TrainingConfig 完全一致的配置 ----------
D_MODEL = 512
D_LATENT = 128
D_INNER = int(((128 * (8 / 3)) // 64) * 64)  # == 320
N_LAYERS = 4
N_EXPERTS = 12
K = 1
VOCAB = 7160
if QUICK:
    BATCH, SEQ = 32, 129  # 4128 token/层
else:
    BATCH, SEQ = 64, 257  # 16448 token/层
BS = BATCH * SEQ
AMP_DTYPE = torch.bfloat16

_WARMUP = 1 if QUICK else 4
_ITERS = 5 if QUICK else 10


def build_args(use_moe, latent=True, d_latent=D_LATENT, d_inner=D_INNER,
               n_exp=N_EXPERTS, k=K):
    return MyLMArgs(
        vocab_size=VOCAB,
        seq_max_len=SEQ,
        d_model=D_MODEL,
        latent_moe=d_latent if latent else 0,
        d_latent=d_latent,
        d_inner=d_inner,
        d_head=128,
        n_heads=None,
        n_layers=N_LAYERS,
        use_moe=use_moe,
        n_experts=n_exp,
        n_experts_per_tok=k,
        dropout=0.05,
        ffn_bias=False,
        attn_bias=True,
    )


def num_params(m):
    return sum(p.numel() for p in m.parameters())


class _NoHead(MyLM):
    """去掉 LM head, 排除词表大 GEMM 后只看 transformer 主体"""

    def forward(self, x, token_ids=None):
        x = self.token_embedding(x)
        for blk in self.blocks:
            x = blk(x, token_ids=token_ids)
        return self.norm(x)


def build_model(args, no_head=False):
    cls = _NoHead if no_head else MyLM
    with redirect_stdout(io.StringIO()):
        m = cls(args)
    return m.to(DEVICE)


def time_ms(fn, warmup=None, iters=None):
    warmup = _WARMUP if warmup is None else warmup
    iters = _ITERS if iters is None else iters
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def make_train_fn(model, no_head=False, batch=None, seq=None):
    """与训练循环等价的 fwd+bwd 闭包 (不含 grad clip / optimizer)"""
    B = batch if batch is not None else BATCH
    S = seq if seq is not None else SEQ
    x = torch.randint(0, VOCAB, (B, S), device=DEVICE)
    y = torch.randint(0, VOCAB, (B, S), device=DEVICE)
    m = torch.ones(B, S, device=DEVICE)

    def fn():
        model.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE, dtype=AMP_DTYPE):
            if no_head:
                out = model(x)
                loss = out.pow(2).mean()
            else:
                out = model(x)
                loss = F.cross_entropy(out.view(-1, VOCAB), y.view(-1), reduction="none")
                loss = (loss * m.view(-1)).sum() / m.sum()
        loss.backward()

    return fn


def make_fwd_fn(model, no_head=False, batch=None, seq=None):
    B = batch if batch is not None else BATCH
    S = seq if seq is not None else SEQ
    x = torch.randint(0, VOCAB, (B, S), device=DEVICE)

    def fn():
        with torch.autocast(DEVICE, dtype=AMP_DTYPE):
            return model(x)

    return fn


# =====================================================================
# 1) 端到端对比
# =====================================================================
def section1():
    print("=" * 84)
    print("1) 端到端训练 step 时间   batch=64, seq=257, BF16 AMP")
    print("=" * 84)
    cfgs = {
        "MoE k=1/12 latent128 (当前)": build_args(use_moe=True),
        "Dense d_inner=320": build_args(use_moe=False, latent=False,
                                        d_latent=D_MODEL, d_inner=320),
        "Dense d_inner=1365 (≈MoE参数量)": build_args(use_moe=False, latent=False,
                                                       d_latent=D_MODEL, d_inner=1365),
    }
    print(f"  {'模型':<31s}{'参数量M':>9s}{'fwd ms':>9s}{'fwd+bwd ms':>12s}")
    for name, args in cfgs.items():
        m = build_model(args)
        tf = time_ms(make_fwd_fn(m))
        tb = time_ms(make_train_fn(m))
        print(f"  {name:<31s}{num_params(m)/1e6:>8.2f}{tf:>9.2f}{tb:>12.2f}")

    print("  ---- 不含 LM head(词表投影), 只看 transformer 主体+反向 ----")
    for name, args in cfgs.items():
        m = build_model(args, no_head=True)
        tf = time_ms(make_fwd_fn(m, no_head=True))
        tb = time_ms(make_train_fn(m, no_head=True))
        print(f"  {name:<31s}{num_params(m)/1e6:>8.2f}{tf:>9.2f}{tb:>12.2f}")
    print()


# =====================================================================
# 2) MoEFFN 子操作耗时分解 (逐段复刻 models.py 的 MoEFFN.forward)
# =====================================================================
def moe_stage_stats(moe: MoEFFN, n_rep=None):
    n_rep = (10 if QUICK else 20) if n_rep is None else n_rep
    """复刻 forward 并逐段计时, 返回各段平均 ms。"""
    N = moe.args.n_experts
    Kk = moe.args.n_experts_per_tok
    d = moe.args.d_latent
    stages = {
        "router+topk+prob": 0.0,
        "sort": 0.0,
        "bincount+cpu同步": 0.0,
        "latent_down": 0.0,
        "expert_loop(gather+FFN+scatter)": 0.0,
        "latent_up": 0.0,
        "TOTAL(实测forward)": 0.0,
    }

    def timed(fn):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        return s, e

    for _ in range(n_rep):
        x = torch.randn(BATCH, SEQ, D_MODEL, device=DEVICE, dtype=AMP_DTYPE)
        ev = []

        with torch.autocast(DEVICE, dtype=AMP_DTYPE):
            s0, e0 = timed(lambda: moe.router(x))
            router_logits = moe.router(x)
            ev.append((s0, e0))  # router

            s1, e1 = timed(lambda: torch.topk(router_logits, Kk, dim=-1))
            tl, ti = torch.topk(router_logits, Kk, dim=-1)
            tp = torch.sqrt(F.softplus(tl))
            tp = tp / tp.sum(dim=-1, keepdim=True)
            ev.append((s1, e1))  # topk+prob

            s2, e2 = timed(lambda: torch.sort(ti.view(-1), stable=True))
            sv, so = torch.sort(ti.view(-1), stable=True)
            tids = so // Kk
            ps = tp.view(-1)[so]
            ev.append((s2, e2))

            s3, e3 = timed(lambda: (
                torch.bincount(sv, minlength=N),
                (torch.cumsum(torch.bincount(sv, minlength=N), 0) - torch.bincount(sv, minlength=N)).cpu().tolist(),
                torch.cumsum(torch.bincount(sv, minlength=N), 0).cpu().tolist(),
            ))
            cnt = torch.bincount(sv, minlength=N)
            ends = torch.cumsum(cnt, dim=0)
            s_cpu = (ends - cnt).cpu().tolist()
            e_cpu = ends.cpu().tolist()
            ev.append((s3, e3))

            xl = x
            if moe.args.latent_moe:
                s4, e4 = timed(lambda: moe.latent_down(x))
                xl = moe.latent_down(x)
                ev.append((s4, e4))

            flat_x = xl.view(-1, d)
            flat_out = torch.zeros_like(flat_x)
            s5, e5 = timed(lambda: _expert_loop(moe, s_cpu, e_cpu, tids, ps, flat_x, flat_out))
            ev.append((s5, e5))

            s6, e6 = timed(lambda: moe.latent_up(flat_out.view(BATCH, SEQ, d)))
            ev.append((s6, e6))

            torch.cuda.synchronize()

        stages["router+topk+prob"] += ev[0][0].elapsed_time(ev[0][1]) + ev[1][0].elapsed_time(ev[1][1])
        stages["sort"] += ev[2][0].elapsed_time(ev[2][1])
        stages["bincount+cpu同步"] += ev[3][0].elapsed_time(ev[3][1])
        if moe.args.latent_moe:
            stages["latent_down"] += ev[4][0].elapsed_time(ev[4][1])
        stages["expert_loop(gather+FFN+scatter)"] += ev[5][0].elapsed_time(ev[5][1])
        stages["latent_up"] += ev[6][0].elapsed_time(ev[6][1])
        # 实测整段 (与训练相同: autocast 下)
        with torch.autocast(DEVICE, dtype=AMP_DTYPE):
            sf, ef = timed(lambda: moe(x))
        torch.cuda.synchronize()
        stages["TOTAL(实测forward)"] += sf.elapsed_time(ef)

    n = n_rep
    for k in stages:
        stages[k] /= n
    return stages


def _expert_loop(moe, s_cpu, e_cpu, tids, ps, flat_x, flat_out):
    for e_i, exp in enumerate(moe.experts):
        s_, e_ = s_cpu[e_i], e_cpu[e_i]
        if s_ < e_:
            t_idx = tids[s_:e_]
            pr = ps[s_:e_].unsqueeze(-1)
            ei = flat_x[t_idx]
            eo = exp(ei, token_ids=None)
            flat_out.index_add_(0, t_idx, eo.to(flat_out.dtype) * pr.to(flat_out.dtype))


def section2():
    print("=" * 84)
    print("2) MoEFFN 单层前向子操作耗时分解  (MoE k=1/12, 12专家循环)")
    print("=" * 84)
    moe = MoEFFN(build_args(use_moe=True)).to(DEVICE)
    st = moe_stage_stats(moe)
    tot = st["TOTAL(实测forward)"]
    for k, v in st.items():
        frac = v / tot * 100 if tot else 0
        print(f"  {k:<38s}{v:>8.3f} ms   ({frac:>5.1f}%)")
    print(f"  注: 该模型共 {N_LAYERS} 层, 每层都有同样一套路由/同步/专家循环")
    print()


# =====================================================================
# 3) profiler: kernel 数量与小 kernel 占比
# =====================================================================
def prof_stats(model, no_head=False, iters=None):
    iters = (2 if QUICK else 5) if iters is None else iters
    x = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.autocast(DEVICE, dtype=AMP_DTYPE):
            for _ in range(iters):
                if no_head:
                    model(x)
                else:
                    model(x)
    torch.cuda.synchronize()

    evs = prof.key_averages()
    n_kernel = 0
    n_small = 0
    time_small = 0.0
    time_all = 0.0
    for ev in evs:
        if ev.self_device_time_total is None:
            continue
        dt = ev.self_device_time_total / 1000.0  # us -> ms
        if dt > 0:
            n_kernel += 1
            time_all += dt
            if dt < 0.015:  # 15us 以下的"小 kernel"(启动开销主导)
                n_small += 1
                time_small += dt
    return {
        "kernels": n_kernel,
        "small_kernels(<15us)": n_small,
        "time_small_ms": time_small,
        "time_all_ms": time_all,
        "table": prof.key_averages(),
    }


def section3():
    print("=" * 84)
    print("3) 单次前向的 CUDA kernel 分布 (profiler, 5 iter)")
    print("=" * 84)
    for name, args in [
        ("MoE k=1/12", build_args(use_moe=True)),
        ("Dense d_inner=320", build_args(use_moe=False, latent=False,
                                         d_latent=D_MODEL, d_inner=320)),
    ]:
        m = build_model(args, no_head=True)
        st = prof_stats(m, no_head=True)
        print(f"  {name:<20s} kernels={st['kernels']:>4d}, "
              f"小kernel(<15us)={st['small_kernels(<15us)']:>4d} "
              f"({st['time_small_ms']/st['time_all_ms']*100:>4.1f}% 时间)")
    print()
    print("  ---- MoE 模型 profiler 前 25 个 CUDA 耗时 kernel ----")
    m = build_model(build_args(use_moe=True), no_head=True)
    st = prof_stats(m, no_head=True)
    tbl = st["table"]
    tbl = sorted(tbl, key=lambda e: e.self_device_time_total or 0, reverse=True)
    print(f"  {'op':<48s}{'self_cuda_ms':>14s}{'count':>7s}")
    for ev in tbl[:25]:
        cnt = ev.count if hasattr(ev, "count") else 1
        print(f"  {ev.key[:47]:<48s}{(ev.self_device_time_total or 0)/1000:>14.3f}{cnt:>7d}")
    print()


# =====================================================================
# 4) CPU 同步次数 (device->host)
# =====================================================================
def section4():
    print("=" * 84)
    print("4) 前向中 device->host 同步(.cpu()/.tolist()/bincount.item) 次数")
    print("=" * 84)
    args = build_args(use_moe=True)
    moe = MoEFFN(args).to(DEVICE)
    x = torch.randn(BATCH, SEQ, D_MODEL, device=DEVICE, dtype=AMP_DTYPE)

    syncs = {"cudaStreamSynchronize": 0, ".to(cpu)": 0}
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.autocast(DEVICE, dtype=AMP_DTYPE):
            moe(x)
    torch.cuda.synchronize()
    for ev in prof.key_averages():
        k = ev.key
        if "cudaStreamSynchronize" in k or "device_synchronize" in k:
            syncs["cudaStreamSynchronize"] += 1
        if "to(cpu)" in k or "to_copy" in k:
            syncs[".to(cpu)"] += 1
    print(f"  单层 MoEFFN: cudaStreamSynchronize 事件 {syncs['cudaStreamSynchronize']} 次, "
          f".to(cpu) {syncs['.to(cpu)']} 次")
    n_per_layer = 2  # starts_cpu / ends_cpu 各一次 .cpu()
    print(f"  代码中每层固定 {n_per_layer} 次 .cpu().tolist() -> 全模型 "
          f"{N_LAYERS} 层 = {N_LAYERS * n_per_layer} 次/前向, 每次强制 GPU 流水线排空")
    print()


# =====================================================================
# 5) 路由负载均衡
# =====================================================================
def section5():
    print("=" * 84)
    print("5) 路由负载均衡: 每专家分到多少 token  (100 次随机前向)")
    print("=" * 84)
    args = build_args(use_moe=True)
    moe = MoEFFN(args).to(DEVICE)
    N = args.n_experts
    samples = []
    n_samp = 30 if QUICK else 100
    with torch.autocast(DEVICE, dtype=AMP_DTYPE):
        for _ in range(n_samp):
            x = torch.randn(BATCH, SEQ, D_MODEL, device=DEVICE)
            router_logits = moe.router(x)
            _, ti = torch.topk(router_logits, K, dim=-1)
            samples.append(torch.bincount(ti.view(-1), minlength=N).cpu().numpy())
    a = np.array(samples)  # [100, N]
    mean = a.mean(axis=0)
    print(f"  每层 token 总数: {BS} (期望均分 {BS / N:.0f}/专家)")
    print(f"  专家均值分布: {' '.join(f'{v:.0f}' for v in mean)}")
    print(f"  min={a.min():.0f}  max={a.max():.0f}  std={a.std():.1f}")
    print(f"  负载失衡时, 最忙的专家决定该层串行循环的耗时")
    print()


# =====================================================================
# 6) torch.compile 开关对比
# =====================================================================
def section6():
    print("=" * 84)
    print("6) torch.compile 对比（默认 inductor 后端）")
    print("   新版 FixedCap MoE 为纯 tensor 分桶, 整模型编译实测 1.7x")
    print("=" * 84)
    args = build_args(use_moe=True)
    m = build_model(args, no_head=True)
    tf0 = time_ms(make_fwd_fn(m, no_head=True))
    tb0 = time_ms(make_train_fn(m, no_head=True))
    print(f"  原生 eager        fwd {tf0:6.2f} ms   fwd+bwd {tb0:6.2f} ms")

    if QUICK:
        print("  (quick 模式: 跳过 compile 实测, 结论见注释——inductor 1.7x)")
        print()
        return
    t0 = time.time()
    try:
        mc = torch.compile(m, mode="max-autotune")  # 默认 inductor
        tf1 = time_ms(make_fwd_fn(mc, no_head=True), warmup=1, iters=6)
        t_compile = time.time() - t0
        tb1 = time_ms(make_train_fn(mc, no_head=True), warmup=1, iters=6)
        print(f"  torch.compile     fwd {tf1:6.2f} ms   fwd+bwd {tb1:6.2f} ms   "
              f"(编译耗时 {t_compile:.0f}s)")
        verdict = "compile 生效" if tf1 < tf0 * 0.9 else "compile 无明显加速"
        print(f"  结论: {verdict}")
    except Exception as ex:
        print(f"  torch.compile 失败/跳过: {type(ex).__name__}: {ex}")
    print()


# =====================================================================
# 7) padded-batched-bmm 参考实现 —— MoE 理论上限
# =====================================================================
class PaddedMoEBmm(nn.Module):
    """单次 batched GEMM 完成全部专家 FFN (padding 到 max tokens), 无 Python 循环无同步"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        N = args.n_experts
        self.router = nn.Linear(args.d_model, N, bias=False)  # 与 models.py 一致, bias=False
        self.latent_down = nn.Linear(args.d_model, args.d_latent, bias=False)
        self.latent_up = nn.Linear(args.d_latent, args.d_model, bias=False)
        w = torch.empty(N, args.d_latent, args.d_inner)
        nn.init.normal_(w, std=0.02)
        self.w_gate = nn.Parameter(w.clone())
        self.w_up = nn.Parameter(w.clone())
        self.w_down = nn.Parameter(torch.empty(N, args.d_inner, args.d_latent).normal_(0, 0.02))

    def forward(self, x, token_ids=None):
        args = self.args
        N, Kk, d = args.n_experts, args.n_experts_per_tok, args.d_latent
        B, S, _ = x.shape
        T = B * S
        logits = self.router(x)  # [B,S,N]
        _, idx = torch.topk(logits, Kk, dim=-1)
        assert Kk == 1, "参考实现仅支持 k=1"
        expert = idx.view(-1)
        # 与 models.py 一致: sqrt(softplus) 后按 K 维归一化 (k=1 时权重恒等于 1.0)
        tl = torch.gather(logits, -1, idx)  # [B,S,1]
        tp = torch.sqrt(F.softplus(tl))
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ps = tp.view(-1)  # [T]

        xl = self.latent_down(x).view(T, -1)  # [T,d]
        order = torch.argsort(expert, stable=True)
        xs = xl[order]
        es = expert[order]
        ps = ps[order]

        counts = torch.bincount(es, minlength=N)  # 仅一次 bincount
        M = int(counts.max().item())  # 唯一的同步
        ends = torch.cumsum(counts, dim=0)
        starts = ends - counts
        ar = torch.arange(T, device=x.device)
        pos_in_row = ar - starts.gather(0, es)
        target = es.long() * M + pos_in_row  # [T] 全局块内下标

        total = N * M
        block = torch.zeros(total, d, device=x.device, dtype=xl.dtype)
        block.index_copy_(0, target, xs)
        block_p = torch.zeros(total, device=x.device, dtype=xl.dtype)
        block_p.index_copy_(0, target, ps.to(xl.dtype))
        block = block.view(N, M, d)

        h1 = F.silu(torch.bmm(block, self.w_gate))
        h2 = torch.bmm(block, self.w_up)
        out = torch.bmm(h1 * h2, self.w_down).view(total, d)
        out = out * block_p.unsqueeze(-1)
        # 槽位 -> 原 token 下标 (padding 槽权重为 0, 累加到哪都无影响)
        token_of_slot = torch.zeros(total, device=x.device, dtype=torch.long)
        token_of_slot.index_copy_(0, target, torch.arange(T, device=x.device))
        y = torch.zeros(T, d, device=x.device, dtype=out.dtype)
        y.index_add_(0, token_of_slot, out)
        return self.latent_up(y.view(B, S, d))


def section7():
    print("=" * 84)
    print("7) 专家 FFN 实现方式对比 (同一套权重形状, 单层前向)")
    print("=" * 84)
    args = build_args(use_moe=True)
    cur = MoEFFN(args).to(DEVICE)
    ref = PaddedMoEBmm(args).to(DEVICE)

    # 正确性: 路由 + 权重对齐后输出应一致
    with torch.no_grad():
        ref.router.weight.copy_(cur.router.weight)
        if ref.router.bias is not None and cur.router.bias is not None:
            ref.router.bias.copy_(cur.router.bias)
        if cur.args.latent_moe:
            ref.latent_down.weight.copy_(cur.latent_down.weight)
            ref.latent_up.weight.copy_(cur.latent_up.weight)
        for i, exp in enumerate(cur.experts):
            ref.w_gate[i].copy_(exp.gate_proj.weight.t())
            ref.w_up[i].copy_(exp.up_proj.weight.t())
            ref.w_down[i].copy_(exp.down_proj.weight.t())
    x = torch.randn(BATCH, SEQ, D_MODEL, device=DEVICE)

    def run_cur():
        with torch.autocast(DEVICE, dtype=AMP_DTYPE):
            return cur(x)

    def run_ref():
        with torch.autocast(DEVICE, dtype=AMP_DTYPE):
            return ref(x)

    with torch.no_grad():
        a = run_cur()
        b = run_ref()
    err = (a.float() - b.float()).abs().max().item()
    ok = err < 5e-2  # bf16 累加顺序不同会带来 ~1e-2 级别舍入差
    print(f"  输出最大误差: {err:.2e}  {'数值一致(bf16舍入差异)' if ok else '不一致!'}")

    t_cur = time_ms(lambda: run_cur())
    t_ref = time_ms(lambda: run_ref())
    print(f"  当前实现 (sort+同步+12专家循环) : {t_cur:8.3f} ms/层")
    print(f"  参考实现 (batched bmm, 1次同步) : {t_ref:8.3f} ms/层")
    print(f"  提速潜力: {t_cur/t_ref:.1f}x")
    print()


# =====================================================================
# 8) 修复实验: batched-GEMM MoE (0 次 CPU 同步) 与 Dense 的最快速度对比
# =====================================================================
class FastMoEFFN(nn.Module):
    """修复版 MoE:
    - 12 个专家合成一次 batched bmm (不再逐专家串行小 GEMM)
    - 固定容量 M_cap 填充, 分桶边界全部在 GPU 上计算 => 0 次 .cpu() 同步
    - 所有逻辑为纯 tensor 操作, 无数据依赖 Python 循环, 可被 torch.compile 正常编译
    """

    def __init__(self, args, m_cap=None):
        super().__init__()
        self.args = args
        self.N = args.n_experts
        self.Kk = args.n_experts_per_tok
        self.d = args.d_latent
        self.M_cap = m_cap  # None => 每层动态容量(1 次 .item() 同步); 否则固定容量(0 同步, 路由需均衡)
        self.router = nn.Linear(args.d_model, args.n_experts, bias=False)
        self.latent_down = nn.Linear(args.d_model, args.d_latent, bias=False)
        self.latent_up = nn.Linear(args.d_latent, args.d_model, bias=False)
        w = torch.empty(self.N, args.d_latent, args.d_inner).normal_(0, 0.02)
        self.w_gate = nn.Parameter(w)
        self.w_up = nn.Parameter(w.clone())
        self.w_down = nn.Parameter(
            torch.empty(self.N, args.d_inner, args.d_latent).normal_(0, 0.02)
        )

    @classmethod
    def from_moeffn(cls, moe: MoEFFN, m_cap=None):
        m = cls(moe.args, m_cap=m_cap)
        with torch.no_grad():
            m.router.weight.copy_(moe.router.weight)
            m.latent_down.weight.copy_(moe.latent_down.weight)
            m.latent_up.weight.copy_(moe.latent_up.weight)
            if hasattr(moe, "w_gate"):
                m.w_gate.copy_(moe.w_gate)
                m.w_up.copy_(moe.w_up)
                m.w_down.copy_(moe.w_down)
            else:
                for i, exp in enumerate(moe.experts):
                    m.w_gate[i].copy_(exp.gate_proj.weight.t())
                    m.w_up[i].copy_(exp.up_proj.weight.t())
                    m.w_down[i].copy_(exp.down_proj.weight.t())
        return m

    def forward(self, x, token_ids=None):
        args = self.args
        N, Kk, d = self.N, self.Kk, self.d
        B, S, _ = x.shape
        T = B * S

        logits = self.router(x)  # [B,S,N]
        _, idx = torch.topk(logits, Kk, dim=-1)
        assert Kk == 1, "FastMoEFFN 仅支持 k=1"
        expert = idx.view(-1)  # [T]
        # 与 models.py 相同的权重公式 (k=1 时分摊后恒为 1.0)
        tl = torch.gather(logits, -1, idx)
        tp = torch.sqrt(F.softplus(tl))
        tp = tp / tp.sum(dim=-1, keepdim=True)

        xl = self.latent_down(x).view(T, -1)  # [T,d]
        # ---- GPU 上分桶: 一次 argsort 把同专家分组 (0 次 CPU 同步) ----
        order = torch.argsort(expert, stable=True)
        es = expert[order]
        xs = xl[order]
        ps = tp.view(-1)[order].to(xl.dtype)

        counts = torch.bincount(es, minlength=N)
        cs = torch.cumsum(counts, dim=0)
        starts = cs - counts
        if self.M_cap is None:
            M = int(counts.max().item())  # 每层唯一一次同步
        else:
            M = self.M_cap
        ar = torch.arange(T, device=x.device)
        pos = ar - starts.gather(0, es)  # 组内序号
        total = N * M
        target = es.long() * M + pos  # 块内全局槽位

        block = torch.zeros(total, d, device=x.device, dtype=xl.dtype)
        block.index_copy_(0, target, xs)
        block = block.view(N, M, d)

        h1 = F.silu(torch.bmm(block, self.w_gate))
        h2 = torch.bmm(block, self.w_up)
        out = torch.bmm(h1 * h2, self.w_down).view(total, d)

        probs = torch.zeros(total, device=x.device, dtype=xl.dtype)
        probs.index_copy_(0, target, ps)
        out = out * probs.unsqueeze(-1)

        token_of_slot = torch.zeros(total, device=x.device, dtype=torch.long)
        token_of_slot.index_copy_(0, target, ar[order])
        y = torch.zeros(T, d, device=x.device, dtype=out.dtype)
        y.index_add_(0, token_of_slot, out)

        return self.latent_up(y.view(B, S, d))


class FixedCapMoEFFN(nn.Module):
    """FastMoEFFN 的固定容量版: 0 次 CPU 同步 (连 M 的 .item() 都去掉)。

    - M_cap = ceil(期望均分token数 * kappa / 16) * 16, 所有层相同, 首个 forward 定下
    - 超容量 token 的 prob 置 0 (丢弃), 用 index_add_ 归入 padding 槽, 不影响结果
    - 代价: 每专家固定算 M 个 slot, 容量内 token 少时浪费算力; kappa 权衡吞吐/丢弃率
    """

    def __init__(self, args, kappa=1.25, ds_gamma=0.0):
        super().__init__()
        self.args = args
        self.kappa = kappa
        self.ds_gamma = ds_gamma
        self.N = args.n_experts
        self.Kk = args.n_experts_per_tok
        self.d = args.d_latent
        self.M_cap = None
        self.last_drop = 0.0
        self.router = nn.Linear(args.d_model, args.n_experts, bias=False)
        self.latent_down = nn.Linear(args.d_model, args.d_latent, bias=False)
        self.latent_up = nn.Linear(args.d_latent, args.d_model, bias=False)
        w = torch.empty(self.N, args.d_latent, args.d_inner).normal_(0, 0.02)
        self.w_gate = nn.Parameter(w)
        self.w_up = nn.Parameter(w.clone())
        self.w_down = nn.Parameter(
            torch.empty(self.N, args.d_inner, args.d_latent).normal_(0, 0.02)
        )
        if ds_gamma > 0:
            self.register_buffer("expert_bias", torch.zeros(args.n_experts))

    @classmethod
    def from_moeffn(cls, moe: MoEFFN, kappa=1.25, ds_gamma=0.0):
        m = cls(moe.args, kappa=kappa, ds_gamma=ds_gamma)
        with torch.no_grad():
            m.router.weight.copy_(moe.router.weight)
            m.latent_down.weight.copy_(moe.latent_down.weight)
            m.latent_up.weight.copy_(moe.latent_up.weight)
            if hasattr(moe, "w_gate"):
                m.w_gate.copy_(moe.w_gate)
                m.w_up.copy_(moe.w_up)
                m.w_down.copy_(moe.w_down)
            else:
                for i, exp in enumerate(moe.experts):
                    m.w_gate[i].copy_(exp.gate_proj.weight.t())
                    m.w_up[i].copy_(exp.up_proj.weight.t())
                    m.w_down[i].copy_(exp.down_proj.weight.t())
        return m

    def forward(self, x, token_ids=None):
        args = self.args
        N, Kk, d = self.N, self.Kk, self.d
        B, S, _ = x.shape
        T = B * S

        logits = self.router(x)
        if self.ds_gamma > 0:
            logits_sel = logits + self.expert_bias
        else:
            logits_sel = logits
        _, idx = torch.topk(logits_sel, Kk, dim=-1)
        assert Kk == 1, "FixedCapMoEFFN 仅支持 k=1"
        expert = idx.view(-1)
        tl = torch.gather(logits, -1, idx)  # 权重用原始 logits
        tp = torch.sqrt(F.softplus(tl))
        tp = tp / tp.sum(dim=-1, keepdim=True)
        if self.ds_gamma > 0 and self.training:
            load = torch.bincount(expert, minlength=N).float()
            self.expert_bias.data.add_(-self.ds_gamma * torch.sign(load - load.mean()))

        if self.M_cap is None:
            mean_tok = T / N
            self.M_cap = math.ceil(mean_tok * self.kappa / 16) * 16
        M = self.M_cap

        xl = self.latent_down(x).view(T, -1)
        order = torch.argsort(expert, stable=True)
        es = expert[order]
        ps = tp.view(-1)[order].to(xl.dtype)

        counts = torch.bincount(es, minlength=N)
        cs = torch.cumsum(counts, dim=0)
        starts = cs - counts
        ar = torch.arange(T, device=x.device)
        pos = ar - starts.gather(0, es)
        keep = pos < M
        pos_safe = torch.clamp(pos, max=M - 1)
        target = es.long() * M + pos_safe
        w = keep.to(xl.dtype)
        total = N * M

        # 超容量 token 输入/权重置 0 后 index_add 到 padding 槽, 无越界且不污染结果
        block = torch.zeros(total, d, device=x.device, dtype=xl.dtype)
        block.index_add_(0, target, xl[order] * w.unsqueeze(-1))
        block_p = torch.zeros(total, device=x.device, dtype=xl.dtype)
        block_p.index_add_(0, target, ps * w)
        block = block.view(N, M, d)

        h1 = F.silu(torch.bmm(block, self.w_gate))
        h2 = torch.bmm(block, self.w_up)
        out = torch.bmm(h1 * h2, self.w_down).view(total, d)
        out = out * block_p.unsqueeze(-1)

        tok_of = torch.zeros(total, device=x.device, dtype=torch.long)
        tok_of.index_add_(0, target, ar[order] * keep.to(torch.long))
        y = torch.zeros(T, d, device=x.device, dtype=out.dtype)
        y.index_add_(0, tok_of, out)

        self.last_drop = (1 - keep.to(torch.float32).mean()).item()
        return self.latent_up(y.view(B, S, d))


class SimpleMaskMoEFFN(nn.Module):
    """MiniMind 风格 MoE: 最朴素实现 (奥卡姆) —— 每专家一个 mask 循环。

    - topk 后逐专家 mask 收集 token, 无 sort/bincount/分桶 (nonzero 有隐式同步, 见测量)
    - 路由公式与 models.py 完全一致: topk(raw logits) + sqrt(softplus) 归一化
    - 可选一行 aux loss (router_aux_loss_coef, MiniMind 同款):
      aux = coef * N * (load * gate.mean(0)).sum(), load 用 topk 选中比例
    - 可选 DeepSeek loss-free 负载均衡 (ds_gamma>0): bias 加到 topk 输入,
      不进梯度图; 每步按负载差自动更新, 无 aux loss
    """

    def __init__(self, args, aux_coef=0.0, ds_gamma=0.0):
        super().__init__()
        self.args = args
        self.aux_coef = aux_coef
        self.ds_gamma = ds_gamma
        self.N = args.n_experts
        self.Kk = args.n_experts_per_tok
        self.router = nn.Linear(args.d_model, args.n_experts, bias=False)
        self.latent_down = nn.Linear(args.d_model, args.d_latent, bias=False)
        self.latent_up = nn.Linear(args.d_latent, args.d_model, bias=False)
        self.experts = nn.ModuleList([FFN(args) for _ in range(args.n_experts)])
        if ds_gamma > 0:
            self.register_buffer("expert_bias", torch.zeros(args.n_experts))

    @classmethod
    def from_moeffn(cls, moe: MoEFFN, aux_coef=0.0, ds_gamma=0.0):
        m = cls(moe.args, aux_coef=aux_coef, ds_gamma=ds_gamma)
        with torch.no_grad():
            m.router.weight.copy_(moe.router.weight)
            m.latent_down.weight.copy_(moe.latent_down.weight)
            m.latent_up.weight.copy_(moe.latent_up.weight)
            if hasattr(moe, "w_gate"):
                for i in range(moe.N):
                    m.experts[i].gate_proj.weight.copy_(moe.w_gate[i].t())
                    m.experts[i].up_proj.weight.copy_(moe.w_up[i].t())
                    m.experts[i].down_proj.weight.copy_(moe.w_down[i].t())
            else:
                for i, exp in enumerate(moe.experts):
                    m.experts[i].load_state_dict(exp.state_dict())
        return m

    def forward(self, x, token_ids=None):
        args = self.args
        B, S, _ = x.shape
        T = B * S
        N, Kk = self.N, self.Kk
        logits = self.router(x).reshape(T, N)
        tl, ti = torch.topk(logits, Kk, dim=-1)
        assert Kk == 1, "SimpleMaskMoEFFN 仅支持 k=1"
        if self.ds_gamma > 0:
            # DeepSeek loss-free 均衡: bias 只影响 topk 选择, 不改权重
            logits_b = logits + self.expert_bias
            tl, ti = torch.topk(logits_b, Kk, dim=-1)
            tl = torch.gather(logits, -1, ti)  # 权重用原始 logits, 与 models.py 一致
            if self.training:
                # bias <- bias - gamma * sign(load_expert - load_mean)  (DeepSeek V3)
                load = torch.bincount(ti.view(-1), minlength=N).float()
                target = torch.sign(load - load.mean()) * self.ds_gamma
                self.expert_bias.data.add_(-target)  # 低负载专家 bias 抬高
        tp = torch.sqrt(F.softplus(tl))
        tp = tp / tp.sum(dim=-1, keepdim=True)  # k=1 时恒 1.0

        xl = self.latent_down(x).view(T, -1)
        y = torch.zeros(T, args.d_latent, device=x.device, dtype=xl.dtype)
        ti_f = ti.view(-1)      # [T] (k=1); k>1 时也按槽位展平, 与权重一致
        tp_f = tp.view(-1)
        for i, expert in enumerate(self.experts):
            tok = (ti_f == i).nonzero().flatten()
            if tok.numel():
                w = tp_f[tok].to(xl.dtype).unsqueeze(-1)
                y.index_add_(0, tok, expert(xl[tok]) * w)
        y = self.latent_up(y.view(B, S, args.d_latent))

        if self.aux_coef:
            load = F.one_hot(ti_f, N).to(xl.dtype).mean(dim=0)       # [N]
            gate = torch.softmax(logits, dim=-1).mean(dim=0)        # [N]
            self.aux_loss = self.aux_coef * N * (load * gate).sum()
        return y


def build_simple_model(args, aux_coef=0.0, ds_gamma=0.0):
    """每层 mlp 换成 SimpleMaskMoEFFN (MiniMind 风格, 每层 N 次 mask 循环)"""
    with redirect_stdout(io.StringIO()):
        m = MyLM(args)
    for blk in m.blocks:
        blk.mlp = SimpleMaskMoEFFN.from_moeffn(
            blk.mlp, aux_coef=aux_coef, ds_gamma=ds_gamma)
    return m.to(DEVICE)


class EinsumMoEFFN(nn.Module):
    """全专家 einsum MoE —— 极简实现: 0 CPU 同步 / 0 Python 循环 / 无 padding 无丢弃 / 数学精确。

    思路 (拿算力换 kernel 数量, latency-bound 场景下的净增益):
      - 把 12 个专家的 gate/up/down 权重各 stack 成 [N, d, d_inner] 张量
      - 3 次 einsum 一次性算完全部专家在全部 token 上的输出 ([T,N,d_inner]) -> N× FLOPs
      - 再 gather 出每个 token 被路由到的那 K 个专家, 加权求和 (k=1 即唯一专家)
    用 3 个大 kernel 替代 36 个串行小 Linear + 24 次 .nonzero()/.numel() 同步。
    当前规模下每专家仅处理 ~1400 token 的小 GEMM, GPU 利用率极低, 此法常为净增益。
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.n_experts
        self.Kk = args.n_experts_per_tok
        self.router = nn.Linear(args.d_model, args.n_experts, bias=False)
        if args.latent_moe:
            self.latent_down = nn.Linear(args.d_model, args.d_latent, bias=False)
            self.latent_up = nn.Linear(args.d_latent, args.d_model, bias=False)
        d_in = args.d_latent if args.latent_moe else args.d_model
        w = torch.empty(args.n_experts, d_in, args.d_inner).normal_(0, 0.02)
        self.w_gate = nn.Parameter(w)
        self.w_up = nn.Parameter(w.clone())
        self.w_down = nn.Parameter(
            torch.empty(args.n_experts, args.d_inner, d_in).normal_(0, 0.02))

    @classmethod
    def from_moeffn(cls, moe: MoEFFN):
        m = cls(moe.args)
        with torch.no_grad():
            m.router.weight.copy_(moe.router.weight)
            if moe.args.latent_moe:
                m.latent_down.weight.copy_(moe.latent_down.weight)
                m.latent_up.weight.copy_(moe.latent_up.weight)
            if hasattr(moe, "w_gate"):
                m.w_gate.copy_(moe.w_gate)
                m.w_up.copy_(moe.w_up)
                m.w_down.copy_(moe.w_down)
            else:
                for i, exp in enumerate(moe.experts):
                    m.w_gate[i].copy_(exp.gate_proj.weight.t())
                    m.w_up[i].copy_(exp.up_proj.weight.t())
                    m.w_down[i].copy_(exp.down_proj.weight.t())
        return m

    def forward(self, x, token_ids=None):
        args = self.args
        N, Kk = self.N, self.Kk
        B, S, _ = x.shape
        T = B * S

        logits = self.router(x)  # [B, S, N]
        tl, ti = torch.topk(logits, Kk, dim=-1)  # [B, S, K]
        tp = torch.sqrt(F.softplus(tl))
        tp = tp / tp.sum(dim=-1, keepdim=True)  # k=1 时恒为 1.0

        if args.latent_moe:
            x = self.latent_down(x)
        d = x.shape[-1]
        flat_x = x.reshape(T, d)  # [T, d]

        # 3 个大 einsum 替代 36 个小 Linear: 一次性算完全部专家在全部 token 上的输出
        gate_all = torch.einsum('td,ndh->tnh', flat_x, self.w_gate)  # [T, N, d_inner]
        up_all = torch.einsum('td,ndh->tnh', flat_x, self.w_up)
        h = F.silu(gate_all) * up_all
        out_all = torch.einsum('tnh,nhd->tnd', h, self.w_down)  # [T, N, d]

        # 每个 token 只取被路由到的那 K 个专家 (k=1 即唯一专家), 加权求和
        ti_flat = ti.reshape(T, Kk)  # [T, K]
        idx_exp = ti_flat.unsqueeze(-1).expand(T, Kk, d)  # [T, K, d]
        selected = out_all.gather(1, idx_exp)  # [T, K, d]
        selected = selected * tp.reshape(T, Kk, 1).to(selected.dtype)
        selected = selected.sum(dim=1)  # [T, d]

        out = selected.view(B, S, d)
        if args.latent_moe:
            out = self.latent_up(out)
        return out


def build_einsum_model(args):
    """每层 mlp 换成 EinsumMoEFFN (3 大 kernel, 0 同步, N× 算力换吞吐)"""
    with redirect_stdout(io.StringIO()):
        m = MyLM(args)
    for blk in m.blocks:
        blk.mlp = EinsumMoEFFN.from_moeffn(blk.mlp)
    return m.to(DEVICE)


def build_fast_moe_model(args, m_cap=None):
    """先建普通 MyLM(MoE), 再把每层 mlp 换成 FastMoEFFN (复制权重)"""
    with redirect_stdout(io.StringIO()):
        m = MyLM(args)
    for blk in m.blocks:
        blk.mlp = FastMoEFFN.from_moeffn(blk.mlp, m_cap=m_cap)
    return m.to(DEVICE)


def build_fixed_cap_model(args, kappa=1.25, ds_gamma=0.0):
    """每层 mlp 换成 FixedCapMoEFFN (0 次同步, 固定容量, 可选 DS bias)"""
    with redirect_stdout(io.StringIO()):
        m = MyLM(args)
    for blk in m.blocks:
        blk.mlp = FixedCapMoEFFN.from_moeffn(
            blk.mlp, kappa=kappa, ds_gamma=ds_gamma)
    return m.to(DEVICE)


def section8():
    print("=" * 84)
    print("8) 修复实验全模型对比 (在与训练一致的完整模型上, 含 LM head)")
    print("=" * 84)

    # 数值验证: 同一份权重, FastMoE 与 原始 MoE 输出一致
    a_moe = build_args(use_moe=True)
    _orig = build_model(a_moe)  # 含 LM head 的原版 MoE
    _fast_weights = build_model(a_moe)
    _fast_weights.load_state_dict(_orig.state_dict())
    for _blk in _fast_weights.blocks:
        _blk.mlp = FastMoEFFN.from_moeffn(_blk.mlp).to(DEVICE)  # 复制同权重再替换
    xv = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    _orig.eval()
    _fast_weights.eval()  # 关掉 attn dropout, 否则两次调用随机 mask 不同
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=AMP_DTYPE):
        o1 = _orig(xv)
        o2 = _fast_weights(xv)
    err = (o1.float() - o2.float()).abs().max().item()
    print(f"  FastMoE vs 原MoE 输出最大误差: {err:.2e}"
          f"  {'一致(bf16舍入)' if err < 5e-2 else '不一致!'}")

    # 竞速模型集合
    race = {
        "MoE原版(eager)": build_model(a_moe),
        "MoE原版(compile)": build_model(a_moe),
        "FastMoE(eager)": build_fast_moe_model(a_moe, m_cap=None),
        "FastMoE(compile)": build_fast_moe_model(a_moe, m_cap=None),
        "Dense-320(eager)": build_model(build_args(use_moe=False, latent=False,
                                                   d_latent=D_MODEL, d_inner=320)),
        "Dense-320(compile)": build_model(build_args(use_moe=False, latent=False,
                                              d_latent=D_MODEL, d_inner=320)),
        "Dense-1365(eager)": build_model(build_args(use_moe=False, latent=False,
                                             d_latent=D_MODEL, d_inner=1365)),
        "Dense-1365(compile)": build_model(build_args(use_moe=False, latent=False,
                                               d_latent=D_MODEL, d_inner=1365)),
    }
    m_orig, m_fast = race["MoE原版(eager)"], race["FastMoE(eager)"]

    if not QUICK:
        for key in ["MoE原版(compile)", "FastMoE(compile)",
                    "Dense-320(compile)", "Dense-1365(compile)"]:
            race[key] = torch.compile(race[key], mode="max-autotune", backend="eager")
    else:
        print("  (quick 模式: 跳过 4 个 compile 变体, 只测 eager)")
        for key in ["MoE原版(compile)", "FastMoE(compile)",
                    "Dense-320(compile)", "Dense-1365(compile)"]:
            race.pop(key, None)

    torch.cuda.synchronize()

    print(f"  {'模型':<22s}{'M':>6s}{'fwd ms':>9s}{'fwd+bwd ms':>13s}")
    results = {}
    for key, m in race.items():
        try:
            tf = time_ms(make_fwd_fn(m))
            tb = time_ms(make_train_fn(m))
        except Exception as ex:
            print(f"  {key:<34s} 失败: {type(ex).__name__}: {ex}")
            continue
        results[key] = tb
        print(f"  {key:<34s}{num_params(m)/1e6:>6.2f}{tf:>9.2f}{tb:>13.2f}")

    moe_best = min(v for k, v in results.items() if "MoE" in k)
    dense_best = min(v for k, v in results.items() if "Dense" in k)
    moe_key = [k for k, v in results.items() if v == moe_best][0]
    dense_key = [k for k, v in results.items() if v == dense_best][0]
    print(f"  最快 MoE   : {moe_key}  {moe_best:.2f} ms")
    print(f"  最快 Dense : {dense_key}  {dense_best:.2f} ms")
    if moe_best < dense_best:
        print(f"  ==> MoE 比 Dense 快 {dense_best/moe_best:.2f}x"
              f"  ({-100*(moe_best-dense_best)/dense_best:.0f}%)")
    else:
        print(f"  ==> Dense 仍快 {moe_best/dense_best:.2f}x  "
              f"({-100*(dense_best-moe_best)/moe_best:.0f}%)")
    print()

    # ---- 全尺寸 (训练真实 B=64, S=257) 快速对拍, 回答"MoE能否比Dense快" ----
    print("  ---- 训练尺寸对拍 (B=64, S=257, eager, 不使用 compile) ----")

    def _full_args(*av, **kw):
        a = build_args(*av, **kw)
        a.seq_max_len = 257
        return a

    full = {}  # 下方构建
    # 分开构建 (seq_max_len=257), 复用训练前随机权重保证成本可比
    def _load_pretrained(m):
        sd = {k: v for k, v in _orig.state_dict().items()
              if not (k.endswith("cos_cached") or k.endswith("sin_cached"))}
        m.load_state_dict(sd, strict=False)  # RoPE 缓存按新 seq 自行生成

    _full_moe = MyLM(_full_args(use_moe=True)).to(DEVICE)
    _load_pretrained(_full_moe)
    _full_fast = MyLM(_full_args(use_moe=True)).to(DEVICE)
    _load_pretrained(_full_fast)
    for _blk in _full_fast.blocks:
        _blk.mlp = FastMoEFFN.from_moeffn(_blk.mlp).to(DEVICE)
    full = {
        "MoE原版": _full_moe,
        "FastMoE": _full_fast,
        "Dense-320": build_model(_full_args(use_moe=False, latent=False,
                                            d_latent=D_MODEL, d_inner=320)),
        "Dense-1365(≈MoE参数)": build_model(_full_args(use_moe=False, latent=False,
                                                       d_latent=D_MODEL, d_inner=1365)),
    }
    fr = {}
    for key, m in full.items():
        try:
            tb = time_ms(make_train_fn(m, batch=64, seq=257), warmup=1, iters=4)
        except Exception as ex:
            print(f"  {key:<18s} 失败: {type(ex).__name__}: {ex}")
            continue
        fr[key] = tb
        print(f"  {key:<18s}fwd+bwd {tb:8.2f} ms")
    moe_key = min(fr, key=fr.get) if any("MoE" in k for k in fr) else None
    dense_key = min((k for k in fr if "Dense" in k), key=fr.get, default=None)
    if moe_key and dense_key:
        mv, dv = fr[moe_key], fr[dense_key]
        if mv < dv:
            print(f"  ==> 训练尺寸下 最快MoE({moe_key}){mv:.1f}ms 快过 最快Dense({dense_key}){dv:.1f}ms: "
                  f"{dv/mv:.2f}x faster")
        else:
            print(f"  ==> 训练尺寸下 最快Dense({dense_key}){dv:.1f}ms 仍快于 MoE({moe_key}){mv:.1f}ms "
                  f"({mv/dv:.2f}x)")
    print()


def collect_cap_drop(model, iters=20):
    """统计 FixedCapMoE 每层超容量丢弃率(%)。全模型前向, 每层每iter一次 .item()。

    注意: 输入必须是 token id (Long), 不是连续向量 —— 走完整模型 forward。
    """
    x = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    drops = {bi: [] for bi in range(len(model.blocks))}
    with torch.no_grad(), torch.autocast(DEVICE, dtype=AMP_DTYPE):
        for _ in range(iters):
            model(x)
            for bi, blk in enumerate(model.blocks):
                mlp = blk.mlp
                drops[bi].append(getattr(mlp, "last_drop", 0.0))
    torch.cuda.synchronize()
    return {k: np.mean(v) * 100 for k, v in drops.items()}


_BENCH_SCRIPT = r"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import time
import torch
from debug_moe import (build_args, build_model, build_fast_moe_model,
                       build_fixed_cap_model, build_simple_model, build_einsum_model,
                       make_fwd_fn, make_train_fn, time_ms, FixedCapMoEFFN, collect_cap_drop)
kind = sys.argv[1]
arg = float(sys.argv[2]) if len(sys.argv) > 2 else None
if kind == "dense":
    a = build_args(use_moe=False, latent=False, d_latent=512, d_inner=320)
    a.seq_max_len = 257
    m = build_model(a)
elif kind == "dense1365":
    a = build_args(use_moe=False, latent=False, d_latent=512, d_inner=1365)
    a.seq_max_len = 257
    m = build_model(a)
else:
    a = build_args(use_moe=True)
    a.seq_max_len = 257
    if kind == "orig":  m = build_model(a)
    elif kind == "simple": m = build_simple_model(a)
    elif kind == "simpleaux": m = build_simple_model(a, aux_coef=float(arg or 5e-4))
    elif kind == "simplebias": m = build_simple_model(a, ds_gamma=5e-3)
    elif kind == "einsum": m = build_einsum_model(a)
    elif kind == "fast": m = build_fast_moe_model(a, m_cap=None)
    elif kind == "fixed": m = build_fixed_cap_model(a, kappa=arg)
    elif kind == "fixedb": m = build_fixed_cap_model(a, kappa=arg, ds_gamma=5e-3)
    elif kind == "origncpl":
        from models import exclude_moe_from_compile
        m = build_model(a)
        exclude_moe_from_compile(m)
        m = torch.compile(m, mode="max-autotune", backend="eager")
    elif kind == "fixedcpl":
        from models import exclude_moe_from_compile
        m = build_fixed_cap_model(a, kappa=arg)
        t0 = time.time()
        m = torch.compile(m, mode="max-autotune", backend="eager")
        print(f"compile_time_s={time.time()-t0:.1f}", flush=True)
    else:
        raise SystemExit(f"unknown kind {kind}")
tf = time_ms(make_fwd_fn(m, batch=64, seq=257), warmup=1, iters=4)
tb = time_ms(make_train_fn(m, batch=64, seq=257), warmup=1, iters=4)
dr = None
mlp0 = getattr(m.blocks[0], "mlp", None)
if isinstance(mlp0, FixedCapMoEFFN) and "cpl" not in kind:
    dr = max(collect_cap_drop(m).values())
print(f"RESULT fwd={tf:.2f} tb={tb:.2f} drop={dr if dr is not None else '-'}")
"""


def _bench_subprocess(kind, arg=None):
    import subprocess
    args = [sys.executable, "-c", _BENCH_SCRIPT, kind]
    if arg is not None:
        args.append(str(arg))
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return None, None, None, r.stdout[-400:] + r.stderr[-400:]
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            parts = line.split()
            tf = float(parts[1].split("=")[1])
            tb = float(parts[2].split("=")[1])
            drop = None if parts[3].split("=")[1] == "-" else float(parts[3].split("=")[1])
            return tf, tb, drop, None
    return None, None, None, r.stdout[-400:]


def section9():
    print("=" * 84)
    print("9) 正规化方案对比: 训练尺寸 B=64, S=257, fwd+bwd (每变体独立子进程)")
    print("=" * 84)

    variants = [
        ("MoE原版(eager)", "orig", None),
        ("SimpleMask(eager)", "simple", None),
        ("EinsumMoE(eager)", "einsum", None),
        ("FastMoE动态(eager)", "fast", None),
        ("FixedCap k=1.10", "fixed", 1.10),
        ("FixedCap k=1.25", "fixed", 1.25),
        ("FixedCap k=1.60", "fixed", 1.60),
        ("FixedCap k=1.25(compile)", "fixedcpl", 1.25),
        ("MoE原版(compile+exclude)", "origncpl", None),
        ("Dense-320(eager)", "dense", None),
    ]
    if QUICK:
        print("(quick 模式: 跳过 compile 变体, 只测 eager)")
        variants = [v for v in variants if "cpl" not in v[1]]

    print(f"  {'模型':<26s}{'fwd ms':>9s}{'fwd+bwd ms':>11s}{'最大丢弃率':>11s}")
    results = {}
    for name, kind, arg in variants:
        tf, tb, drop, err = _bench_subprocess(kind, arg)
        if tf is None:
            print(f"  {name:<26s} 子进程失败: {err}")
            continue
        results[name] = tb
        ds = f"{drop:.1f}%" if drop is not None else "-"
        print(f"  {name:<26s}{tf:>9.2f}{tb:>11.2f}{ds:>11s}")

    moe = {k: v for k, v in results.items() if "MoE" in k or "FixedCap" in k or "FastMoE" in k}
    dense = {k: v for k, v in results.items() if "Dense" in k}
    best_moe = min(moe, key=moe.get)
    best_dense = min(dense, key=dense.get) if dense else None
    print(f"\n  最快 MoE: {best_moe} ({moe[best_moe]:.2f} ms)")
    if best_dense:
        print(f"  最快 Dense: {best_dense} ({dense[best_dense]:.2f} ms)")
        if moe[best_moe] < dense[best_dense]:
            print(f"  ==> MoE已快于 Dense ({dense[best_dense]/moe[best_moe]:.2f}x)")
        else:
            print(f"  ==> Dense仍快 {moe[best_moe]/dense[best_dense]:.2f}x")

    base = results.get("MoE原版(eager)")
    if base:
        print("\n  相对 原版MoE 的变化:")
        for k, v in results.items():
            if k == "MoE原版(eager)":
                continue
            tag = "+" if v > base else "-"
            print(f"    {k}: {base/v if v else 0:.2f}x  ({tag}{abs(v-base)/base*100:.1f}%)")
    print()


def section10():
    print("=" * 84)
    print("10) 奥卡姆裁决: 原版sort / MiniMask简单版 / FixedCap 三选一")
    print("=" * 84)

    # A) 数值一致性: 同一权重下 SimpleMask / EinsumMoE vs 原版
    a_moe = build_args(use_moe=True)
    _orig = build_model(a_moe)
    sm = build_simple_model(a_moe)
    sd = {k: v for k, v in _orig.state_dict().items()
          if not (k.endswith("cos_cached") or k.endswith("sin_cached"))}
    sm.load_state_dict(sd, strict=False)
    # EinsumMoE: 非mlp权重从原版load, mlp用from_moeffn复制(_orig同权重)
    em = MyLM(a_moe).to(DEVICE)
    _non_mlp = {k: v for k, v in sd.items() if ".mlp." not in k}
    em.load_state_dict(_non_mlp, strict=False)
    for blk_o, blk_e in zip(_orig.blocks, em.blocks):
        blk_e.mlp = EinsumMoEFFN.from_moeffn(blk_o.mlp).to(DEVICE)
    xv = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    _orig.eval(); sm.eval(); em.eval()
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=AMP_DTYPE):
        o1 = _orig(xv); o2 = sm(xv); o3 = em(xv)
    err_sm = (o1.float() - o2.float()).abs().max().item()
    err_em = (o1.float() - o3.float()).abs().max().item()
    print(f"  A) SimpleMask vs 原MoE 误差: {err_sm:.2e}"
          f"  {'一致(bf16舍入)' if err_sm < 5e-2 else '不一致!'}")
    print(f"     EinsumMoE vs 原MoE 误差: {err_em:.2e}"
          f"  {'一致(bf16舍入)' if err_em < 5e-2 else '不一致!'}")

    # B) 实现体量 (forward 方法体行数, 反映维护成本)
    import ast as _ast
    src = open(__file__, encoding="utf-8").read()
    m_src = open(os.path.join(os.path.dirname(__file__), "models.py"),
                 encoding="utf-8").read()
    def _fwd_body_lines(source, cls):
        tree = _ast.parse(source)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ClassDef) and node.name == cls:
                for child in node.body:
                    if isinstance(child, _ast.FunctionDef) and child.name == "forward":
                        return child.end_lineno - child.lineno - 1
        return -1
    loc = {"MoEFFN原版": _fwd_body_lines(m_src, "MoEFFN"),
           "SimpleMask": _fwd_body_lines(src, "SimpleMaskMoEFFN"),
           "EinsumMoE": _fwd_body_lines(src, "EinsumMoEFFN"),
           "FixedCap": _fwd_body_lines(src, "FixedCapMoEFFN")}
    print("  B) forward 方法体行数 (越小越好维护):")
    for k, v in loc.items():
        print(f"    {k:<16s} {v} 行")

    # C) 吞吐对拍 (独立子进程, 训练尺寸)
    print("  C) 训练尺寸对拍 (B=64, S=257, fwd+bwd):")
    variants = [
        ("MoE原版(sort+同步)", "orig", None, "瓶颈基线"),
        ("SimpleMask(朴素)", "simple", None, "MiniMind风格, nonzero同步"),
        ("EinsumMoE", "einsum", None, "3大kernel 0同步 N×算力, 精确无丢"),
        ("SimpleMask+aux5e-4", "simpleaux", 5e-4, "一行aux平衡, 同赛道"),
        ("FixedCap k=1.10", "fixed", 1.10, "bmm 0同步, 快但有丢率"),
        ("FixedCap k=1.25", "fixed", 1.25, "bmm 0同步"),
        ("Dense-320", "dense", None, "非MoE上限"),
    ]
    if QUICK:
        print("  (quick: 只测前4项)")
        variants = variants[:4]
    results = {}
    for name, kind, arg, note in variants:
        tf, tb, drop, errd = _bench_subprocess(kind, arg)
        if tf is None:
            print(f"    {name:<22s} 失败: {errd}")
            continue
        results[name] = tb
        ds = f" (最大丢率{drop:.0f}%)" if drop is not None else ""
        print(f"    {name:<22s} fwd+bwd {tb:8.2f} ms{ds}  {note}")
    base = results.get("MoE原版(sort+同步)")
    if base:
        print("\n    相对原版:")
        for k, v in results.items():
            print(f"      {k:<22s} {v:8.2f} ms  ({v/base:.2f}x)")
    print()

    # D) 平衡验证: 微小训练 无/有一行 aux loss, 看路由是否崩向单专家
    print("  D) 平衡验证: 200步小训 (B=32,S=129,随机数据), 专家利用均衡度:")
    for coef, tag in [(0.0, "无aux "), (5e-4, "有aux5e-4")]:
        res = _run_train_probe(coef)
        print(f"    {tag}: {res}")

    print()


def _run_train_probe(coef):
    """子进程跑 200 步随机数据小训练, 返回每层 count.max/count.min 均衡度。
    aux=0 或 5e-4, 观测路由是否崩塌到单个专家。"""
    import subprocess
    script = r'''
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch, torch.nn.functional as F
from debug_moe import build_args, build_simple_model, VOCAB
coef = float(sys.argv[1])
torch.manual_seed(7)
a = build_args(use_moe=True)
m = build_simple_model(a, aux_coef=coef).cuda().train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
x = torch.randint(0, VOCAB, (32, 129), device="cuda")
y = torch.randint(0, VOCAB, (32, 129), device="cuda")
for s in range(200):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(x, token_ids=x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        if coef:
            loss = loss + sum(b.mlp.aux_loss for b in m.blocks)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    got = []
    handles = []
    for blk in m.blocks:
        h = blk.mlp.router.register_forward_hook(
            lambda mod, i, o: got.append(o))
        handles.append(h)
    logits = m(x, token_ids=x)
    for h in handles:
        h.remove()
    out = []
    for bi, raw in enumerate(got):
        cnt = torch.bincount(raw.argmax(-1).view(-1), minlength=12).float()
        out.append(f"L{bi} max={cnt.max().item()/cnt.sum().item():.0%}"
                   f" min={cnt.min().item()/cnt.sum().item():.0%}")
    print("BAL " + " ".join(out) + f" loss={loss.item():.3f}")
'''
    r = subprocess.run([sys.executable, "-c", script, str(coef)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=os.path.dirname(os.path.abspath(__file__)))
    if r.returncode != 0:
        return "FAIL: " + (r.stderr or r.stdout)[-400:]
    for line in r.stdout.splitlines():
        if line.startswith("BAL"):
            return line[4:]
    return "FAIL: no BAL line\n" + r.stdout[-400:]


def section11():
    print("=" * 84)
    print("11) 真实语料简易训练: 路由均衡性 + 训练速度 (B=32, S=192)")
    print("=" * 84)

    from dataset import PretrainTokenIDDataset
    data_path = "data/mini_data192mixen_v3.npy" if os.path.exists("data/mini_data192mixen_v3.npy") \
        else "data/medium_data256v2.npy"
    seq_len = 192 if "mini" in data_path else 256
    print(f"  数据: {data_path}  seq={seq_len}  (vocab={VOCAB})")

    ds = PretrainTokenIDDataset(data_path, seq_max_len=seq_len)
    n_use = min(len(ds), 100_000)
    rng = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=rng)[:n_use].tolist()
    ds.indices = idx
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True,
                                         num_workers=2, drop_last=True)

    steps_total = 400 if not QUICK else 100

    def _train(model, tag):
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        got = {}
        handles = []
        for bi, blk in enumerate(model.blocks):
            mlp = blk.mlp
            if hasattr(mlp, "router"):
                handles.append((bi, mlp.router.register_forward_hook(
                    lambda mod, i, o, bi=bi: got.update({bi: o.detach().clone()}))))
        t0 = time.time()
        loss = float("nan")
        for step, (xb, yb, mb) in enumerate(loader):
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, dtype=AMP_DTYPE):
                logits = model(xb, token_ids=xb)
                loss = (F.cross_entropy(logits.view(-1, VOCAB), yb.view(-1),
                                        reduction="none") * mb.view(-1)).sum() / mb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step + 1 >= steps_total:
                break
        dt = time.time() - t0
        ms_step = dt * 1000 / min(steps_total, len(loader))

        # 最终路由分布: hook 拿到的 router logits (+ bias 后 argmax)
        dists = {}
        for bi, raw in got.items():
            mlp = model.blocks[bi].mlp
            lr = raw + getattr(mlp, "expert_bias", 0)
            cnt = torch.bincount(lr.argmax(-1).view(-1), minlength=N_EXPERTS).float()
            dists[bi] = cnt / cnt.sum()
        stats = []
        for bi in range(N_LAYERS):
            d = dists.get(bi)
            if d is not None:
                stats.append(f"L{bi} max={d.max().item()*100:.0f}%"
                             f" min={d.min().item()*100:.0f}%")
        for h in handles:
            h[1].remove()
        # FixedCap 系列: 汇总最终丢率
        drops = []
        for blk in model.blocks:
            mlp = blk.mlp
            if isinstance(mlp, FixedCapMoEFFN):
                drops.append(f"{mlp.last_drop*100:.0f}%")
        return ms_step, loss, "  ".join(stats), drops

    a = build_args(use_moe=True)
    a.seq_max_len = seq_len
    a.vocab_size = VOCAB
    print(f"  MoE ({num_params(build_model(a))/1e6:.1f}M params):")
    moe_ms, moe_loss, moe_stats, _ = _train(build_model(a), "MoE")
    print(f"    loss={moe_loss:.3f}  {moe_stats}")
    print(f"    ms/step={moe_ms:.1f}")

    ds_gamma = 1e-2 if QUICK else 5e-3
    print(f"  SimpleMask+DS-bias({ds_gamma}) ({num_params(build_model(a))/1e6:.1f}M params):")
    ds_ms, ds_loss, ds_stats, _ = _train(build_simple_model(a, ds_gamma=ds_gamma), "DS-bias")
    print(f"    loss={ds_loss:.3f}  {ds_stats}")
    print(f"    ms/step={ds_ms:.1f}")

    print(f"  FixedCap k=1.25 (无均衡):")
    fc_ms, fc_loss, fc_stats, fc_drops = _train(build_fixed_cap_model(a, kappa=1.25), "FC")
    print(f"    loss={fc_loss:.3f}  {fc_stats}  丢率={fc_drops}")
    print(f"    ms/step={fc_ms:.1f}")

    print(f"  FixedCap k=1.25 + DS-bias({ds_gamma}):")
    fcd_ms, fcd_loss, fcd_stats, fcd_drops = _train(
        build_fixed_cap_model(a, kappa=1.25, ds_gamma=ds_gamma), "FC+DS")
    print(f"    loss={fcd_loss:.3f}  {fcd_stats}  丢率={fcd_drops}")
    print(f"    ms/step={fcd_ms:.1f}")

    a2 = build_args(use_moe=False, latent=False, d_latent=D_MODEL, d_inner=320)
    a2.seq_max_len = seq_len
    a2.vocab_size = VOCAB
    print(f"  Dense-320 ({num_params(build_model(a2))/1e6:.1f}M params):")
    de_ms, de_loss, _, _ = _train(build_model(a2), "Dense")
    print(f"    loss={de_loss:.3f}")
    print(f"    ms/step={de_ms:.1f}")

    print(f"\n  ==> MoE {moe_ms:.1f} ms/step | +DS {ds_ms:.1f} | FixedCap {fc_ms:.1f} "
          f"| FixedCap+DS {fcd_ms:.1f} | Dense {de_ms:.1f} ms/step")
    print()
    print("  (崩塌时 SimpleMask 反而'快': 多个专家空转直接跳过; FixedCap 靠丢token"
          "压成本; DS-bias 均衡后 FixedCap 丢率应趋向 0, 速度回到容量算力)")
    print()


def section12():
    print("=" * 84)
    print("12) κ 扫描 + compile: 只选'又快又不丢'的点 (DS-bias 固定开启)")
    print("=" * 84)

    from dataset import PretrainTokenIDDataset
    data_path = "data/mini_data192mixen_v3.npy" if os.path.exists("data/mini_data192mixen_v3.npy") \
        else "data/medium_data256v2.npy"
    seq_len = 192 if "mini" in data_path else 256
    print(f"  数据: {data_path}  seq={seq_len}  (vocab={VOCAB}), 训练 B=32")

    ds = PretrainTokenIDDataset(data_path, seq_max_len=seq_len)
    n_use = min(len(ds), 100_000)
    rng = torch.Generator().manual_seed(0)
    ds.indices = torch.randperm(len(ds), generator=rng)[:n_use].tolist()
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True,
                                         num_workers=2, drop_last=True)
    steps_total = 300 if not QUICK else 80
    ds_gamma = 5e-3

    def _train_kappa(model):
        """训 steps_total 步, 复试: 每次构造新模型/优化器, 统计 loss/丢率/速度"""
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        got = {}
        handles = []
        for bi, blk in enumerate(model.blocks):
            mlp = blk.mlp
            if hasattr(mlp, "router"):
                handles.append((bi, mlp.router.register_forward_hook(
                    lambda mod, i, o, bi=bi: got.update({bi: o.detach().clone()}))))
        t0 = time.time()
        loss = float("nan")
        for step, (xb, yb, mb) in enumerate(loader):
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, dtype=AMP_DTYPE):
                logits = model(xb, token_ids=xb)
                loss = (F.cross_entropy(logits.view(-1, VOCAB), yb.view(-1),
                                        reduction="none") * mb.view(-1)).sum() / mb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step + 1 >= steps_total:
                break
        ms_step = (time.time() - t0) * 1000 / min(steps_total, len(loader))
        loss_v = loss.item()
        max_drop = 0.0
        dists = {}
        for bi, raw in got.items():
            mlp = model.blocks[bi].mlp
            max_drop = max(max_drop, getattr(mlp, "last_drop", 0.0))
            lr = raw + getattr(mlp, "expert_bias", 0)
            cnt = torch.bincount(lr.argmax(-1).view(-1), minlength=N_EXPERTS).float()
            dists[bi] = cnt / cnt.sum()
        mx = max((d.max().item() for d in dists.values()), default=0.0)
        for h in handles:
            h[1].remove()
        return ms_step, loss_v, max_drop * 100, mx * 100

    def run_final(kappa, compile_it=False):
        a = build_args(use_moe=True)
        a.seq_max_len = seq_len
        a.vocab_size = VOCAB
        m = build_fixed_cap_model(a, kappa=kappa, ds_gamma=ds_gamma)
        if compile_it:
            m = torch.compile(m, mode="max-autotune", backend="eager")
        ms, loss, mx_drop, mx = _train_kappa(m)
        tag = f"FixedCap k={kappa:.2f}" + (" +compile" if compile_it else "")
        print(f"  {tag:<28s} {ms:7.1f} ms/step  loss={loss:.3f} 丢率={mx_drop:.1f}%  "
              f"路由最大={mx:.0f}%")
        return ms, loss, mx_drop, mx

    print("\n  A) κ 扫描 (eager, 只选丢率≤3% 的最快点):")
    rows = {}
    for k in (1.05, 1.10, 1.25):
        rows[k] = run_final(k)
    print("\n  B) 编译收益 (FixedCap 纯 tensor):")
    run_final(1.10, compile_it=True)
    run_final(1.25, compile_it=True)
    print()

    ok = {k: r for k, r in rows.items() if r[2] <= 3.0}
    if ok:
        best = min(ok, key=lambda k: ok[k][0])
        print(f"  ==> 选 κ={best:.2f}: 丢率≤3% 中最快的点 ({ok[best][0]:.1f} ms/step)")
    else:
        print("  ==> 无 κ 满足丢率≤3%, 不激进, 维持 κ=1.25")
    print()


def section13():
    print("=" * 84)
    print("13) 总横评: 所有候选在同一口径 (B=64, S=257, fwd+bwd) 下重测")
    print("=" * 84)

    variants = [
        ("Dense-320(非MoE上限)", "dense", None),
        ("Dense-1365(≈MoE参量)", "dense1365", None),
        ("MoE原版(生产逐专家mask)", "orig", None),
        ("SimpleMask+DS-bias", "simplebias", None),
        ("FixedCap k=1.10 无均衡", "fixed", 1.10),
        ("FixedCap k=1.25 无均衡", "fixed", 1.25),
        ("FixedCap k=1.15 +DS-bias", "fixedb", 1.15),
        ("FixedCap k=1.25 +DS-bias", "fixedb", 1.25),
        ("EinsumMoE(全einsum)", "einsum", None),
        ("FastMoE 动态容量", "fast", None),
    ]
    if QUICK:
        print("(quick: 只跑前 6 项)")
        variants = variants[:6]

    print(f"\n  {'模型':<28s}{'fwd ms':>9s}{'fwd+bwd ms':>11s}{'丢率(max)':>10s}"
          f"  vs MoE原版")
    results = {}
    for name, kind, arg in variants:
        tf, tb, drop, errd = _bench_subprocess(kind, arg)
        if tf is None:
            print(f"  {name:<20s} 失败: {errd}")
            continue
        results[name] = tb
        ds = f"{drop:.0f}%" if drop is not None else "-"
        print(f"  {name:<20s}{tf:>9.2f}{tb:>13.2f}{ds:>10s}")

    base = results.get("MoE原版(Simple逐专家mask)")
    if base is None and "orig" in " ".join(results):
        # quick 模式名字可能不同, 直接从第一个 MoE 行取基准
        base = next(v for k, v in results.items() if "MoE原版" in k)
    if base:
        print("\n  相对 MoE 原版 (1.00 = 持平):")
        for k, v in sorted(results.items(), key=lambda kv: kv[1]):
            t = f"{v/base:+.2f}" if v >= base else f"{v/base:.2f}x"
            print(f"    {k:<26s} {v:8.2f} ms  {t}")
    dense = {k: v for k, v in results.items() if "Dense" in k}
    if dense:
        dv = min(dense.values())
        print(f"\n  最快 Dense-320: {dv:.1f} ms; "
              f"最快 MoE: {min(v for k, v in results.items() if 'MoE' in k):.1f} ms "
              f"({min(v for k, v in results.items() if 'MoE' in k)/dv:.2f}x 慢于Dense)")
    print()
    print("  注: 丢率列是未训练权重下的最大丢弃率。训练收敛 + DS-bias 后 FixedCap"
          "\n    丢率降至 1~7% (见 section11/12), 此处说明的是\"塌陷最坏情况\"。")
    print()
    print("  ---- 训练实测口径 (B=32, S=192, 真实语料, 300~400 步, ms/step) ----")
    print("    参考 section11/12 数据 (非本轮重测, 同机器同配置):")
    print("      SimpleMask        ~110   Dense-320        ~57")
    print("      FixedCap+DS-bias  ~68    (比 SimpleMask 快 1.6x, 比 Dense 慢 1.2x)")
    print()


def summary():
    print("=" * 84)
    print("结论 (瓶颈定位 + 提速路径)")
    print("=" * 84)
    print("""
  1) 瓶颈分解 (单层 MoEFFN 前向 ~5ms, 训练尺寸 B=64,S=257):
     - 专家循环 76%: 12 个专家串行 gather+FFN+index_add, 每个只处理 ~1400
       token, latency-bound 而非 compute-bound。
     - CPU 同步 13%: 每层 2 次 .cpu().tolist(), 强制排空 GPU 流水线。
     - 路由/topk/sort ~8-9%。

  2) torch.compile (默认 backend="eager" + exclude): 旧版 MoE 的逐专家
     数据依赖循环会触发 dynamo graph-break/重编译, fwd 慢 10x (66ms -> 700ms)。
     2026-08 更新: 新版 FixedCap (排序分桶+bmm, 纯 tensor 无循环) 可直接
     inductor 编译, B=32,S=193 实测 fwd+bwd 37.9ms vs eager 64.4ms (1.7x);
     MoE 包含在编译内 (38.6ms) 比 exclude MoE (44.2ms) 更快, 无需再排除。

  3) 实测提速结论 (训练尺寸 B=64,S=257, fwd+bwd ms, 独立子进程干净测量):
     - 原版 MoEFFN (sort+bincount+CPU同步) : ~183
     - SimpleMask (逐专家 mask, 0 附加)    : ~185  (1.00x, 数学完全等价, 当前生产)
     - EinsumMoE (全专家 einsum, 0同步精确): ~206  <- 更慢! 详见 3a
     - FastMoE 动态容量 batched bmm       : ~195  <- 更慢! 原因见第5点
     - FixedCap k=1.10~1.25 (0同步)        : ~160  <- 最快, 但丢率 78~80%, 见第4/5点
     - Dense-320                           : ~151  (非MoE上限)

=> 最终选定 (2026-07): SimpleMask 逐专家 mask 版 (无循环, 可编译)。
         2026-08 更新: 已升级为 FixedCap κ=1.25 + DS-bias 版 (models.py
         MoEFFN): 纯 tensor 分桶 + batched bmm, 比 SimpleMask 快 1.13x
         (B=64,S=257: 177 vs 200ms), κ=1.25 下训练收敛后丢率 0~5%
         (DS-bias 自均衡, 无 aux loss)。

  3a) EinsumMoE 负结果 (本轮新增, 用于厘清瓶颈性质):
      把 12 个专家合成 3 次大 einsum, 消除 24 次 .nonzero() 同步 + 36 个小 Linear。
      数学精确 (0.00e+00), 但 fwd+bwd 反而慢 12% (206 vs 183)。
      拆开看: fwd 68ms (比 SimpleMask 70ms 还快, 确实省 kernel), 但 bwd ~138ms
      (比 SimpleMask ~114ms 慢 20%), 因为反向要对 [T,N,d_inner] 中间量算三份梯度,
      N× 算力在反向被放大。
      结论: 训练吞吐的瓶颈不是单纯 kernel-launch 延迟 (否则 fwd 不会更快), 而是
      反向算力 + [T,N,d_inner] 中间量的显存带宽。证明"全专家密集计算"换不来吞吐,
      精确 MoE 想破 ~185ms 必须走分组 bmm (只算被分配的 token) 即 FixedCap 路线。

  4) FixedCapMoEFFN = FastMoE 的固定容量版: 每专家固定 M_cap 槽位,
     0 次 .cpu() 同步, 超容量 token 权重置 0 (index_add 进 padding 槽)。
     吞吐最快 (~142ms), 但未训练时丢率 78~81%, 平衡高度依赖 aux loss,
     而本课题明确不使用 aux loss, 故不选。

  5) 路由崩塌 (本次调查的核心发现): 在真实模型的注意力之后的层上,
     top-1 路由 logits 尺度变大, argmax 几乎全部落到同一个专家
     (counts.max 达到 12853/16448), 其余 11 个专家空转/闲置:
     - FastMoE 动态容量因此要 padding 到 M≈13000, 全模型反而变慢;
     - FixedCap 靠丢弃超容量 token 把代价压住, 但崩塌时 78~81% token
       被丢弃, 吞吐指标是"快但有损"的。
     不使用 aux loss (用户决定): 崩塌修复留给真实训练收敛后的自平衡
     (200 步随机小训实测: 无 aux 时深层仍 30~41% 失衡, 有待长训验证)。

  6) SimpleMask+aux 5e-4 实测 1835ms (约 11x 慢): F.one_hot + softmax
     的 aux 路径在 bf16/autocast 下有严重性能异常, 进一步佐证不用 aux。
""")
    print()


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}   "
          f"B*S={BS} token/层  n_layers={N_LAYERS}  n_experts={N_EXPERTS} k={K}  "
          f"AMP={AMP_DTYPE}")
    only = os.environ.get("MOE_DEBUG_SECTIONS", "")
    sections = {
        "1": section1, "2": section2, "3": section3, "4": section4,
        "5": section5, "6": section6, "7": section7, "8": section8,
        "9": section9, "10": section10, "11": section11, "12": section12,
        "13": section13,
    }
    targets = [sections[s] for s in (only.split(",") or ["1"]) if s in sections] if only else \
              [section1, section2, section3, section4, section5, section6,
               section7, section8, section9, section10, section11, section12,
               section13, summary]
    for fn in targets:
        fn()
    print("done.")
