# -*- coding: utf-8 -*-
"""通用模块性能 benchmark 工具（从仓库根运行）。

用法:
    uv run python tools/checks/bench_module.py

这不是命令行工具: 所有配置都在本文件的「常量配置」区完成。
核心是 bench_module(spec) 函数，可 benchmark **任意 nn.Module**
（PyTorch 自带模块、第三方模块、models.py 组件一视同仁）。BENCH_TARGETS
列表里直接放模块对象即可（实例或 lambda 工厂），想跑谁写谁、注释掉即不跑；
需要自定义输入/名字的条目再写成 dict 形式。

每个模块输出三类指标:
  - FP   吞吐: no_grad 纯前向（推理口径），tokens/s(或 iters/s) + ms/iter + TFLOP/s
  - BP   吞吐: backward 单独事件计时（前向建图不计入）
  - STEP 吞吐: FP+BP(+optimizer.step) 的完整训练步口径，附 FP/BP/OPT 时间分解
以及显存: 参数+缓冲常驻、FP 峰值、FP+BP 峰值、STEP 峰值（差值≈激活/梯度）。

口径说明:
  - USE_AMP=True 与 pre_train.py 一致: 权重 fp32 + bf16 autocast。
  - TFLOP/s 用 torch.utils.flop_counter.FlopCounterMode 实测 GEMM 类算子
    （matmul/bmm/conv/sdpa），不含 elementwise/softmax，只用于横向比较。
  - 无参数模块想测带输入梯度的 BP，设 INPUT_REQUIRES_GRAD = True。
"""
import os
import sys
import gc
import time
import statistics
from pathlib import Path

# Windows + torch/openmp 双加载兜底（AGENTS.md 约定，其他脚本需自行设置）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tqdm import tqdm
from torch.utils.flop_counter import FlopCounterMode

# 示例中引用本仓库组件；测纯 torch 模块时不需要本段，随意删改
from models import (
    MyLMArgs,
    RMSNorm,
    Attention,
    FFN,
    MoEFFN,
    MyLMDecoderLayer,
    MyLM,
    CompressedAttention,
)

# ============================ 常量配置 ============================
# ---- 形状 / batch（决定 tokens/iter = BATCH x SEQ_LEN；须在 BENCH_TARGETS 之前）----
BATCH = 64
SEQ_LEN = 512
D_MODEL = 512
D_INNER = 1344
VOCAB_SIZE = 7160        # 仅 model 用（tokenizer 实际保存大小）

# ---- 训练口径 ----
USE_AMP = True            # 权重 fp32 + bf16 autocast（同 pre_train.py）
USE_COMPILE = True        # torch.compile
COMPILE_MODE = "max-autotune"  # "default" | "reduce-overhead" | "max-autotune"
TRAIN_MODE = True         # True: module.train()（含 dropout 等有状态分支）
INCLUDE_OPTIMIZER = True  # STEP 口径是否叠加 optimizer.step（AdamW）
INPUT_REQUIRES_GRAD = False  # 无参数模块需要回传到输入梯度时置 True
SEED = 42

# ---- 计时 ----
WARMUP_ITERS = 5
MEASURE_ITERS = 32

# ---- 待测模块列表: 直接放对象, 想跑谁写谁, 注释掉即不跑 ----
# 条目三种写法（自动识别）:
#   1. nn.Module 实例/零参工厂     -> 用默认输入(按类型自选)+自动命名
#   2. spec dict {"module", "inputs", "name"?, "params"?} -> 完全控制
#   3. {"module": 普通可调用, "params": [...]} -> 无状态函数带独立参数
def default_args() -> MyLMArgs:
    """models.py 组件的缺省结构参数（对齐 pre_train.py 当前训练配置）。"""
    return MyLMArgs(
        d_model=512, latent_moe=False, d_latent=256,
        d_inner=1344, d_head=128, n_heads=None, n_layers=6,
        vocab_size=VOCAB_SIZE, seq_max_len=SEQ_LEN,
        n_experts=8, n_experts_per_tok=2, moe_capacity=1.25,
        attn_bias=True, dropout=0.05, base_init_std=0.02,
    )


def rand_hidden(device):
    """(B,S,D) 激活输入，适合 Attention/FFN/MoE/DecoderLayer 等隐藏态模块。"""
    return torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device)


def rand_ids(device):
    """(B,S) token id，适合 MyLM 等以词表索引为输入的模块。"""
    return torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN), device=device)


def moe_layer(args):
    """把 decoder layer 的 mlp 换成 MoE 的小构造器（示例用）。"""
    m = MyLMDecoderLayer(args, layer_idx=1)
    m.mlp = MoEFFN(args)
    return m


BENCH_TARGETS = [
    # 直接传对象: 名字自动取类名, 输入按类型给默认(MyLM 用 ids, 其余用 hidden)
    # MoEFFN(default_args()),
    Attention(default_args(), use_gate=False),
    CompressedAttention(default_args(), compress_ratio=4),
    # RMSNorm(default_args().d_model),                     # 已弃用的旧手写实现（对照用）
    # torch.nn.RMSNorm(default_args().d_model, eps=1e-6),  # 现役：模型内所有 norm 均用它
    # torch.nn.LayerNorm(default_args().d_model),
    # MyLMDecoderLayer(default_args(), layer_idx=1),
    # 实例也可写成工厂(lambda: MyLM(default_args()))，延迟到计时前才构造
    # 需要自定义输入/名字时用 dict 形式:
    # {
    #     "name": "decoder_layer_moe",
    #     "module": lambda: moe_layer(default_args()),
    #     "inputs": rand_hidden,
    # },
    # --- 其他常用示例（注释的条目不参与本轮）---
    # Attention(default_args()),
    # Attention(default_args(), use_gate=True),
    # FFN(default_args()),
    # {"module": lambda: MyLM(moe_args), "inputs": rand_ids},   # MyLM 自动用 ids
    # {"module": lambda: torch.nn.Linear(D_MODEL, D_INNER).cuda(),
    #  "inputs": rand_hidden},
]
# ================================================================


# ----------------------------- 通用工具 -----------------------------

class EventTimer:
    """CUDA event 计时器: with 块配对，单位 ms。"""

    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        torch.cuda.synchronize()
        return False

    def ms(self) -> float:
        return self.start.elapsed_time(self.end)


def stats_of(times_ms):
    return {
        "mean": statistics.fmean(times_ms),
        "median": statistics.median(times_ms),
        "std": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def first_tensor(out):
    """forward 输出可能是 Tensor / tuple / dict，取第一个张量算 loss。"""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        out = list(out.values())
    if isinstance(out, (tuple, list)):
        for v in out:
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError(f"无法从输出 {type(out)} 中找到 loss 用张量")


def normalize_spec(spec):
    """条目归一化: nn.Module 实例/可调用对象 → spec dict。"""
    if isinstance(spec, dict):
        return dict(spec)
    return {"module": spec}


def build_module(spec):
    """实例化 module 并返回 (module, params)。

    module: nn.Module 实例 | 返回实例的零参 callable | 普通 callable(配 params)
    """
    module = spec["module"]
    if callable(module) and not isinstance(module, torch.nn.Module):
        built = module()
    else:
        built = module
    params = spec.get("params")
    if isinstance(built, torch.nn.Module):
        params = list(built.parameters())
    if params is None:
        raise TypeError("module 既不是 nn.Module 也未提供 params 列表")
    return built, params


def resolve_inputs(spec, module):
    """未显式给 inputs 时按模块类型猜默认: MyLM 用 ids, 其余用 hidden。"""
    inputs = spec.get("inputs", rand_ids if isinstance(module, MyLM)
                     else rand_hidden)
    if callable(inputs):
        inputs = inputs(torch.device("cuda"))
    return inputs


def count_flops(module, inputs, device):
    """FlopCounterMode 实测一次前向的 GEMM 类 FLOPs。"""
    try:
        with FlopCounterMode(display=False) as counter:
            with torch.no_grad(), torch.autocast(
                    device.type, enabled=USE_AMP,
                    dtype=torch.bfloat16 if USE_AMP else torch.float32):
                module(*inputs)
        return float(counter.get_total_flops())
    except Exception as exc:  # 个别自定义算子不被 counter 支持时不阻塞
        print(f"  [warn] FLOPs 计数失败: {exc}")
        return float("nan")


# ----------------------------- 单模块 benchmark -----------------------------

def bench_module(spec, device: torch.device) -> dict:
    """通用入口: 对一个条目(spec)跑完整 FP/BP/STEP 计时与显存统计。

    spec 可为 nn.Module 实例、零参工厂或 {"module","inputs","name","params"} dict。
    """
    spec = normalize_spec(spec)
    torch.manual_seed(SEED)
    module, params = build_module(spec)
    name = spec.get("name") or type(module).__name__
    if isinstance(module, torch.nn.Module):
        module.to(device)
        module.train(TRAIN_MODE)
    if USE_COMPILE and isinstance(module, torch.nn.Module):
        module = torch.compile(module, mode=COMPILE_MODE)

    inputs = resolve_inputs(spec, module)
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    if INPUT_REQUIRES_GRAD:
        for t in inputs:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                t.requires_grad_(True)

    amp_dtype = torch.bfloat16 if USE_AMP else torch.float32

    def forward():
        with torch.autocast(device.type, enabled=USE_AMP, dtype=amp_dtype):
            return module(*inputs)

    def backward(out):
        first_tensor(out).sum().backward()

    n_params = sum(p.numel() for p in params)
    param_mem = sum(p.numel() * p.element_size() for p in params)
    if isinstance(module, torch.nn.Module):
        param_mem += sum(b.numel() * b.element_size() for b in module.buffers())
    flops_fp = count_flops(module, inputs, device)

    optimizer = None
    if INCLUDE_OPTIMIZER and params:
        optimizer = torch.optim.AdamW(
            [p for p in params if p.requires_grad], lr=1e-3)

    t_compile = 0.0
    if USE_COMPILE:
        torch.cuda.synchronize()
        t_compile = -time.perf_counter()
    # warmup: 触发 compile / MoE MCap 定下 / 梯度缓冲首分配
    for _ in tqdm(range(WARMUP_ITERS), desc=f"[{name}] warmup-FP", leave=False):
        with torch.no_grad():
            forward()
    for _ in tqdm(range(WARMUP_ITERS), desc=f"[{name}] warmup-step", leave=False):
        if optimizer:
            optimizer.zero_grad(set_to_none=False)
        elif isinstance(module, torch.nn.Module):
            module.zero_grad(set_to_none=False)
        backward(forward())
        if optimizer:
            optimizer.step()
    if USE_COMPILE:
        t_compile += time.perf_counter()
    torch.cuda.synchronize()

    results = {"name": name, "n_params": n_params, "param_mem": param_mem}

    # ---- FP: no_grad 前向 ----
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fp_times = []
    for _ in tqdm(range(MEASURE_ITERS), desc=f"[{name}] FP(no_grad)", leave=False):
        with torch.no_grad(), EventTimer() as t:
            forward()
        fp_times.append(t.ms())
    results["fp"] = stats_of(fp_times)
    results["fp_peak_mem"] = torch.cuda.max_memory_allocated()

    # ---- 训练前向（建图, 不反传）----
    fwd_g_times = []
    for _ in tqdm(range(MEASURE_ITERS), desc=f"[{name}] FP(建图)", leave=False):
        with EventTimer() as t:
            out = forward()
        del out
        fwd_g_times.append(t.ms())
    results["fwd_grad"] = stats_of(fwd_g_times)

    # ---- BP: 图已建好, 只计 backward ----
    def clear_grads():
        if isinstance(module, torch.nn.Module):
            module.zero_grad(set_to_none=True)
        for p in params:
            if not isinstance(module, torch.nn.Module) and p.grad is not None:
                p.grad = None

    clear_grads()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    bp_times = []
    for _ in tqdm(range(MEASURE_ITERS), desc=f"[{name}] BP", leave=False):
        out = forward()
        with EventTimer() as t:
            backward(out)
        bp_times.append(t.ms())
        del out
    results["bp"] = stats_of(bp_times)
    results["bp_peak_mem"] = torch.cuda.max_memory_allocated()

    # ---- STEP: FP+BP(+opt) 完整训练步 ----
    clear_grads()
    for p in params:  # 预分配梯度缓冲, 口径同 pre_train.py 的 CUDAGraph 约定
        if p.requires_grad and p.grad is None:
            p.grad = torch.zeros_like(p)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step_times, s_fp, s_bp, s_opt = [], [], [], []

    def zero_keep_buf():  # 保留梯度缓冲置零, 口径同 pre_train.py
        if isinstance(module, torch.nn.Module):
            module.zero_grad(set_to_none=False)
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    for _ in tqdm(range(MEASURE_ITERS), desc=f"[{name}] STEP(FP+BP)", leave=False):
        if optimizer:
            optimizer.zero_grad(set_to_none=False)
        else:
            zero_keep_buf()
        with EventTimer() as t_step:
            with EventTimer() as t_fwd:
                out = forward()
            with EventTimer() as t_bwd:
                backward(out)
            del out
            if optimizer:
                with EventTimer() as t_opt:
                    optimizer.step()
                s_opt.append(t_opt.ms())
        step_times.append(t_step.ms())
        s_fp.append(t_fwd.ms())
        s_bp.append(t_bwd.ms())
    results["step"] = stats_of(step_times)
    results["step_fp"] = stats_of(s_fp)
    results["step_bp"] = stats_of(s_bp)
    results["step_opt"] = stats_of(s_opt) if s_opt else {"mean": 0.0, "std": 0.0}
    results["step_peak_mem"] = torch.cuda.max_memory_allocated()
    results["reserved"] = torch.cuda.max_memory_reserved()
    results["flops_fp"] = flops_fp
    if USE_COMPILE:
        results["compile_s"] = t_compile
    return results


# ----------------------------- 报告 -----------------------------

def report_one(r: dict):
    tokens = BATCH * SEQ_LEN
    fl = r["flops_fp"]
    print(f"\n{'=' * 78}")
    print(f"模块: {r['name']}   B={BATCH} S={SEQ_LEN} tokens/iter={tokens:,}")
    print(f"amp={'bf16' if USE_AMP else 'fp32'}  compile={USE_COMPILE}"
          f"  train_mode={TRAIN_MODE}  optimizer={'AdamW' if INCLUDE_OPTIMIZER else '无'}")
    print(f"参数量: {r['n_params'] / 1e6:.2f} M   "
          f"参数+缓冲显存: {r['param_mem'] / 2**20:.1f} MiB"
          + (f"   FP 理论算子: {fl / 1e9:.2f} GFLOP" if fl == fl else ""))
    if "compile_s" in r:
        print(f"编译+warmup 耗时: {r['compile_s']:.1f} s")
    print("-" * 78)
    print(f"{'指标':<18}{'ms/iter(mean±std)':>22}{'tokens/s':>14}{'TFLOP/s':>10}")
    for key, label, mult in (
        ("fp", "FP (no_grad)", 1.0),
        ("fwd_grad", "FP (建图)", 1.0),
        ("bp", "BP (backward)", 2.0),   # 经验值: backward GEMM ≈ 2x forward
        ("step", "STEP (FP+BP+opt)", 3.0),
    ):
        s = r[key]
        tok_s = tokens / (s["mean"] / 1000.0)
        tf = (fl * mult / (s["mean"] / 1000.0) / 1e12) if fl == fl else float("nan")
        print(f"{label:<18}{s['mean']:>13.3f}±{s['std']:<6.3f}{tok_s:>14,.0f}{tf:>10.2f}")
    print("-" * 78)
    fp_ms, bp_ms, opt_ms = (r["step_fp"]["mean"], r["step_bp"]["mean"],
                            r["step_opt"]["mean"])
    total = fp_ms + bp_ms + opt_ms
    if total > 0:
        print("STEP 分解: " + "  ".join(
            f"{n} {v:.3f} ms ({v / total * 100:.1f}%)"
            for n, v in (("FP", fp_ms), ("BP", bp_ms), ("OPT", opt_ms))))
    print(f"显存峰值  FP           : {r['fp_peak_mem'] / 2**20:>10.1f} MiB")
    print(f"显存峰值  FP+BP        : {r['bp_peak_mem'] / 2**20:>10.1f} MiB")
    print(f"显存峰值  STEP(+opt)   : {r['step_peak_mem'] / 2**20:>10.1f} MiB")
    print(f"≈激活+梯度 (STEP-FP)   : "
          f"{(r['step_peak_mem'] - r['fp_peak_mem']) / 2**20:>10.1f} MiB")
    print(f"显存 reserved(峰值)    : {r['reserved'] / 2**20:>10.1f} MiB")
    print("=" * 78)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 设备")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")  # 对齐 PreTrainer.__init__
    if not BENCH_TARGETS:
        raise SystemExit("BENCH_TARGETS 为空, 请直接在列表里放入待测模块对象")

    print(f"GPU: {torch.cuda.get_device_name(0)}   "
          f"torch {torch.__version__}   cuda {torch.version.cuda}")
    for entry in BENCH_TARGETS:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        report_one(bench_module(entry, device))


if __name__ == "__main__":
    main()
