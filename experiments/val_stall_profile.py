"""微实验:定位 pre_train.py 触发 validate 时 GPU 长时间空转的根因。

验证三个假设(全部用 profile 数据说话,不凭感觉):
  H1 val_loader persistent_workers=False -> 每次 validate 重新 spawn worker 进程池
     (Windows spawn:每个 worker 重新 import 主模块+torch),期间 GPU 空闲。
  H2 val worker 每轮重 spawn 还会重新 pickle「当前 epoch 的 base.perm」
     -> 验证集内容逐 epoch 漂移;persistent=True 则冻结在首 spawn 时的快照。
  H3 generate_test 的 padding_side="none" 使 decode 每步输入形状 +1,
     torch.compile 触发一连串重编译(max-autotune 下单次编译代价进一步放大);
     改 padding_side="left" 固定形状后消失。inference_mode 与 train 切换
     是否引入额外编译,由 exp2_ctx 单独验证。

运行方式(父进程自动编排全部 phase,子进程隔离 dynamo/编译缓存状态):
    $env:PYTHONUTF8='1'; uv run python experiments/val_stall_profile.py
产物:experiments/results/val_stall_<时间戳>/{summary.md, phases.jsonl, *_stderr.log}

注意(Windows spawn 坑):worker 会重新 import 本文件顶层,故顶层只允许
import/常量,一切执行逻辑包在 __main__ guard 里。
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# ---------------- 常量(与生产口径对应,规模缩小) ----------------
MINI_DATA = str(ROOT / "data/mini_data192mixen_v3.npy")   # 行宽 193,uint16
TOKENIZER = str(ROOT / "tokenizer/bbpe_tokenizer_7k_260723_xl.json")
SEQ_MAX_LEN = 128          # exp1 用(<=192 存储列)
SEQ_MAX_LEN_ML = 192       # exp2 模型用(对齐 mini 数据存储列)
SEED = 42

# exp1:验证轮数(每轮模拟一次 validate 触发)
EXP1_ROUNDS = 4
# exp2:generate 轮数(每轮 = 一次 val 里的 generate_test 调用)
EXP2_ROUNDS = 4
EXP2_GEN_LEN = 20


# ---------------- GPU 利用率采样(证明「空转」) ----------------
class GpuSampler:
    """后台线程按 ~200ms 采样 nvidia-smi 利用率,按时间窗切片统计 idle 占比。"""

    def __init__(self, enabled=True):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        if enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                util = int(out.splitlines()[0])
            except Exception:
                util = -1
            self.samples.append((t, util))
            self._stop.wait(0.2)

    def stop(self):
        self._stop.set()

    def stats(self, t0, t1):
        vals = [u for (t, u) in self.samples if t0 <= t <= t1 and u >= 0]
        if not vals:
            return None
        idle = sum(1 for u in vals if u <= 10) / len(vals)
        return {"n": len(vals), "idle_frac": round(idle, 3),
                "mean_util": round(sum(vals) / len(vals), 1)}


# ---------------- exp1:DataLoader 每次 validate 的重建成本 ----------------
def _exp1_build(variant):
    from dataset import PretrainTokenIDDataset

    base = PretrainTokenIDDataset(
        MINI_DATA, seq_max_len=SEQ_MAX_LEN, downsample=0.004,
        padding_side="right", shuffle_seed=SEED,
    )
    n_val = max(16, int(len(base) * 0.03))
    n_tr = len(base) - n_val
    train_ds, val_ds = torch.utils.data.random_split(base, [n_tr, n_val])

    # 生产同款 train loader(常驻 worker 与 val 抢 CPU/管道,保真)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=8, shuffle=False, drop_last=True,
        pin_memory=True, num_workers=4, prefetch_factor=4,
        persistent_workers=False,
    )
    common = dict(batch_size=8, drop_last=True, pin_memory=True)
    if variant == "old":       # 生产改前配置
        vl_kwargs = dict(shuffle=True, num_workers=4, prefetch_factor=3,
                         persistent_workers=False)
    elif variant == "new":     # 生产改后配置
        vl_kwargs = dict(shuffle=False, num_workers=2, prefetch_factor=2,
                         persistent_workers=True)
    else:                      # 无 worker 对照
        vl_kwargs = dict(shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, **common, **vl_kwargs)
    return base, train_loader, val_loader


def phase_exp1(variant):
    device = torch.device("cuda")
    base, train_loader, val_loader = _exp1_build(variant)
    # 把 train worker 池先跑起来(模拟训练进行中,val 是在 epoch 中途触发的)
    tr_it = iter(train_loader)
    for _ in range(2):
        x, y, m = next(tr_it)
        _ = x.to(device, non_blocking=True)
    # 极简 GPU 负载:只为让利用率采样有意义,不承担计时对象
    w = torch.randn(SEQ_MAX_LEN, 64, device=device)
    sampler = GpuSampler()
    records = []
    for r in range(EXP1_ROUNDS):
        # epoch 开头生产会重排 base.perm(train 侧 M4 机制),此处同样调用
        base.set_permute_seed(SEED + r)
        t0 = time.perf_counter()
        it = iter(val_loader)          # old:此处付 spawn 代价
        t_enter = time.perf_counter()
        first_batch_t = None
        wait_total = 0.0
        row_hash = []
        n_b = 0
        while True:
            tw = time.perf_counter()
            try:
                inp, tgt, msk = next(it)
            except StopIteration:
                break
            wait_total += time.perf_counter() - tw
            if first_batch_t is None:
                first_batch_t = tw + wait_total - t0
            row_hash.append(int(inp[0, :16].sum().item()))
            n_b += 1
            _ = (inp.to(device, non_blocking=True).float()
                 .view(-1, SEQ_MAX_LEN) @ w).sum()
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        # 释放本轮迭代器(old 的 worker 池拆卸/新建发生在下轮 iter())
        del it
        records.append({
            "variant": variant, "round": r,
            "t_first_batch": round(first_batch_t or 0.0, 3),
            "t_round_total": round(t_end - t0, 3),
            "t_iter_make": round(t_enter - t0, 3),
            "loader_wait": round(wait_total, 3),
            "n_batches": n_b,
            "row_sig": row_hash[0] if row_hash else None,
            "gpu_win": sampler.stats(t0, t_end),
        })
    sampler.stop()
    return {"phase": f"exp1_{variant}", "records": records}


# ---------------- exp2:torch.compile + generate 形状漂移 ----------------
def _build_small_lm_args(vocab_size):
    from models import MyLMArgs
    return MyLMArgs(
        d_model=256, d_inner=768, n_layers=4, latent_moe=0, d_latent=256,
        vocab_size=vocab_size, seq_max_len=SEQ_MAX_LEN_ML,
        d_conv=4, compress_ratio=8, attn_bias=True, dropout=0.05, d_head=64,
    )


def _build_compiled_model(mode="default"):
    from models import MyLM
    import tokenizers
    tok = tokenizers.Tokenizer.from_file(TOKENIZER)
    args = _build_small_lm_args(tok.get_vocab_size())
    model = MyLM(args).to(torch.device("cuda"))
    if mode == "max-autotune":
        model = torch.compile(model, mode="max-autotune")
    else:
        model = torch.compile(model)
    return model, tok


def phase_exp2_generate(padding_side):
    from utils import TextGenerator
    model, tok = _build_compiled_model("default")
    device = torch.device("cuda")
    gen = TextGenerator(model, tok, device, padding_side="none")
    sampler = GpuSampler()
    rounds = []
    for r in range(EXP2_ROUNDS):
        t0 = time.perf_counter()
        gen.generate(
            start_token="人工智能", gen_seq_len=EXP2_GEN_LEN,
            print_out=False, temperature=0.7, top_k=20,
            frequency_penalty=1.5, padding_side=padding_side,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        rounds.append({"round": r, "wall_s": round(t1 - t0, 2),
                       "gpu_win": sampler.stats(t0, t1)})
    sampler.stop()
    return {"phase": f"exp2_generate_{padding_side}", "rounds": rounds}


def phase_exp2_eager():
    """方案 C:generate 用未 compile 的裸模块(_orig_mod,与 compiled 共享参数),
    训练/验证仍走 compiled。验证:decode 不触发任何 dynamo 编译,且首个
    validate 的 eval 图编译不受影响。"""
    from utils import TextGenerator
    from models import MyLM
    import tokenizers
    device = torch.device("cuda")
    tok = tokenizers.Tokenizer.from_file(TOKENIZER)
    args = _build_small_lm_args(tok.get_vocab_size())
    base_model = MyLM(args).to(device)
    compiled = torch.compile(base_model)
    gen = TextGenerator(base_model, tok, device, padding_side="none")
    rounds = []
    sampler = GpuSampler()
    for r in range(EXP2_ROUNDS):
        t0 = time.perf_counter()
        gen.generate(start_token="人工智能", gen_seq_len=EXP2_GEN_LEN,
                     print_out=False, temperature=0.7, top_k=20,
                     frequency_penalty=1.5)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        rounds.append({"round": r, "wall_s": round(t1 - t0, 3),
                       "gpu_win": sampler.stats(t0, t1)})
    sampler.stop()
    ids = torch.randint(1, tok.get_vocab_size(), (4, SEQ_MAX_LEN_ML),
                        device=device, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = compiled(ids, padding_mask=(ids != 0))
        torch.cuda.synchronize()
    rounds.append({"round": "val_eval_graph(对照,应仍有首编译)",
                   "wall_s": round(time.perf_counter() - t0, 2)})
    return {"phase": "exp2_eager", "rounds": rounds}


def phase_exp2_ctx():
    """train 步 <-> validate(inference_mode) 切换是否引入额外编译。"""
    device = torch.device("cuda")
    model, tok = _build_compiled_model("default")
    args = model.args if hasattr(model, "args") else model._orig_mod.args
    pad_id = args.pad_id
    crit = nn.CrossEntropyLoss(reduction="none")
    ids = torch.randint(1, tok.get_vocab_size(), (4, SEQ_MAX_LEN_ML),
                        device=device, dtype=torch.long)
    tgt = torch.randint(1, tok.get_vocab_size(), (4, SEQ_MAX_LEN_ML),
                        device=device, dtype=torch.long)
    msk = torch.ones_like(tgt, dtype=torch.float)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    sampler = GpuSampler()
    segs = []
    for r in range(3):
        for seg in ("train_step", "val_inference_mode", "val_no_grad"):
            t0 = time.perf_counter()
            if seg == "train_step":
                model.train()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(ids, padding_mask=(ids != pad_id))
                    loss = (crit(out.view(-1, out.size(-1)), tgt.view(-1))
                            * msk.view(-1)).sum()
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            else:
                model.eval()
                ctx = (torch.inference_mode() if seg == "val_inference_mode"
                       else torch.no_grad())
                with ctx:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(ids, padding_mask=(ids != pad_id))
                        _ = out.float().sum()
            torch.cuda.synchronize()
            segs.append({"round": r, "seg": seg,
                         "wall_s": round(time.perf_counter() - t0, 2)})
    sampler.stop()
    return {"phase": "exp2_ctx", "segments": segs}


# ---------------- exp3:单次编译代价(default vs max-autotune) ----------------
def phase_exp3_compile(mode):
    device = torch.device("cuda")
    model, tok = _build_compiled_model(mode)
    ids = torch.randint(
        1, tok.get_vocab_size(), (4, SEQ_MAX_LEN_ML),
        device=device, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(ids, padding_mask=(ids != 0))
        torch.cuda.synchronize()
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(20):
            model(ids, padding_mask=(ids != 0))
        torch.cuda.synchronize()
    t_warm = (time.perf_counter() - t0) / 20
    return {"phase": f"exp3_compile_{mode}",
            "first_call_compile_s": round(t_first, 1),
            "warm_call_s": round(t_warm, 4)}


# ---------------- 父进程编排 ----------------
PHASES = {
    "exp1_old": lambda: phase_exp1("old"),
    "exp1_new": lambda: phase_exp1("new"),
    "exp1_w0": lambda: phase_exp1("w0"),
    "exp2_none": lambda: phase_exp2_generate("none"),
    "exp2_left": lambda: phase_exp2_generate("left"),
    "exp2_eager": phase_exp2_eager,
    "exp2_ctx": phase_exp2_ctx,
    "exp3_default": lambda: phase_exp3_compile("default"),
    "exp3_autotune": lambda: phase_exp3_compile("max-autotune"),
}


def run_child(phase, out_dir):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--phase", phase]
    env = dict(os.environ)
    if phase.startswith("exp2") or phase.startswith("exp3"):
        env["TORCH_LOGS"] = "recompiles"
    p = subprocess.run(cmd, capture_output=True, text=True,
                      encoding="utf-8", errors="replace", env=env,
                      cwd=str(ROOT))
    (out_dir / f"{phase}_stderr.log").write_text(
        p.stderr or "", encoding="utf-8", errors="replace")
    n_recomp = sum(
        1 for ln in (p.stderr or "").splitlines()
        if "recompil" in ln.lower())
    n_limit = sum(
        1 for ln in (p.stderr or "").splitlines()
        if "cache_size_limit" in ln.lower())
    rec = None
    for ln in (p.stdout or "").splitlines():
        if ln.startswith("RESULT_JSON "):
            rec = json.loads(ln[len("RESULT_JSON "):])
    if rec is None:
        rec = {"phase": phase, "error": (p.stderr or "")[-800:],
               "returncode": p.returncode}
    rec["stderr_recompile_lines"] = n_recomp
    rec["stderr_cache_size_limit_lines"] = n_limit
    return rec


def summarize(results, out_dir):
    lines = ["# validate 卡点微实验结果", ""]
    with open(out_dir / "phases.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def table(rs, header, rowfmt):
        lines.append("## " + rs[0]["phase"])
        lines.append("")
        lines.append(header)
        lines.append("|" + "---|" * len(header.split("|")) )
        for r in rs:
            lines.append(rowfmt(r))
        if rs and "stderr_recompile_lines" in rs[0]:
            lines.append(f"- recompiles(stderr 行数): "
                         f"{rs[0].get('stderr_recompile_lines')}, "
                         f"cache_size_limit: "
                         f"{rs[0].get('stderr_cache_size_limit_lines')}")
        lines.append("")

    if any(r.get("error") for r in results):
        lines.append("!! 有 phase 失败,详见 phases.jsonl / *_stderr.log\n")
    ok = [r for r in results if "records" in r]
    if ok:
        header = "| phase | round | iter创建(s) | 首批等待(s) | loader等待累计(s) | 整轮(s) | 首批行签名 | GPU空闲占比 |"
        def rowf(r, rec):
            g = rec.get("gpu_win") or {}
            return (f"| {r['phase']} | {rec['round']} | "
                    f"{rec['t_iter_make']} | {rec['t_first_batch']} | "
                    f"{rec['loader_wait']} | {rec['t_round_total']} | "
                    f"{rec['row_sig']} | {g.get('idle_frac')} |")
        for r in ok:
            lines.append("## " + r["phase"])
            lines.append("")
            lines.append(header)
            lines.append("|---|---|---|---|---|---|---|---|")
            for rec in r["records"]:
                lines.append(rowf(r, rec))
            lines.append("")
    gen = [r for r in results if "rounds" in r]
    for r in gen:
        lines.append("## " + r["phase"])
        lines.append("")
        lines.append("| round | wall(s) | GPU空闲占比 |")
        lines.append("|---|---|---|")
        for rec in r["rounds"]:
            g = rec.get("gpu_win") or {}
            lines.append(f"| {rec['round']} | {rec['wall_s']} "
                         f"| {g.get('idle_frac')} |")
        lines.append(f"- recompiles(stderr): "
                     f"{r.get('stderr_recompile_lines')}, cache_size_limit: "
                     f"{r.get('stderr_cache_size_limit_lines')}")
        lines.append("")
    ctx = [r for r in results if "segments" in r]
    for r in ctx:
        lines.append("## " + r["phase"])
        lines.append("")
        lines.append("| round | seg | wall(s) |")
        lines.append("|---|---|---|")
        for rec in r["segments"]:
            lines.append(f"| {rec['round']} | {rec['seg']} | "
                         f"{rec['wall_s']} |")
        lines.append(f"- recompiles(stderr): "
                     f"{r.get('stderr_recompile_lines')}")
        lines.append("")
    c3 = [r for r in results if "first_call_compile_s" in r]
    for r in c3:
        lines.append(f"## {r['phase']}: 首次调用(含编译)={r['first_call_compile_s']}s,"
                     f" 稳态={r['warm_call_s']}s/次,"
                     f" recompiles(stderr)={r.get('stderr_recompile_lines')}")
        lines.append("")
    txt = "\n".join(lines)
    (out_dir / "summary.md").write_text(txt, encoding="utf-8")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.phase:  # 子进程:只跑一个 phase,结果走 stdout
        t0 = time.perf_counter()
        rec = PHASES[a.phase]()
        rec["phase_wall_s"] = round(time.perf_counter() - t0, 1)
        print("RESULT_JSON " + json.dumps(rec, ensure_ascii=False))
        return

    ts = time.strftime("%y%m%d-%H%M%S")
    out_dir = ROOT / "experiments/results" / f"val_stall_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ph in PHASES:
        print(f"== phase {ph} ==", flush=True)
        try:
            r = run_child(ph, out_dir)
        except Exception as e:
            r = {"phase": ph, "error": repr(e)}
        results.append(r)
        key = r.get("phase_wall_s", r.get("error", "")[:80])
        print(f"   done: wall={key} recompiles={r.get('stderr_recompile_lines')}",
              flush=True)
    summarize(results, out_dir)


if __name__ == "__main__":
    main()
