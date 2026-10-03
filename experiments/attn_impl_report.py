# -*- coding: utf-8 -*-
"""汇总 experiments/attn_impl_bench_*.csv：绝对值表 + sdpa 相对 std 的比值表与图。

用法:
  uv run python experiments/attn_impl_report.py                # 自动取 results 下全部
  uv run python experiments/attn_impl_report.py --csv a.csv --png
  uv run python experiments/attn_impl_report.py --impls std,sdpa,sdpa-cudnn
"""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
BASE_IMPL = "std"
MEM_KEY = "step_dev_mib"        # 设备级（含 CUDA Graph 私有池）
MEM_FALLBACK = "step_peak_mib"  # 老 CSV 回退到分配器口径


def num(r, k):
    v = (r.get(k) or "").strip() if r.get(k) is not None else ""
    if v in ("", "OOM"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def mem_of(r):
    v = num(r, MEM_KEY)
    return v if v is not None else num(r, MEM_FALLBACK)


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def ratio(a, b):
    if a is None or b is None or b == 0:
        return "-"
    return f"{a / b:.2f}x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*", default=None)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--impls", default="std,sdpa,sdpa-cudnn",
                    help="绝对值表里保留的实现（比值表始终含全部）")
    ap.add_argument("--out", default=str(RESULTS / "attn_impl_report.md"))
    args = ap.parse_args()

    paths = args.csv or sorted(glob.glob(str(RESULTS / "attn_impl_bench_*.csv")))
    print("CSV: " + ", ".join(Path(p).name for p in paths))
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    keep_abs = {s.strip() for s in args.impls.split(",") if s.strip()}

    # ---- 索引：core/module/model 用同一 batch 可直接配对；
    #      capacity 的 std/sdpa 各自 OOM 前最大 batch 不同，key 里不能带 batch ----
    idx = defaultdict(dict)
    for r in rows:
        batch = r.get("batch", "")
        key = (r["stage"], r.get("mask", ""), r.get("branch", "") or "-",
               int(float(r["seq"])),
               -1 if r["stage"] == "capacity" else int(float(batch)),
               r["compile"] == "True")
        impl = r["impl"]
        idx[key].setdefault(impl, []).append(r)

    def one(d, impl):
        lst = d.get(impl)
        if not lst:
            return None
        # capacity 里同一 impl 可能有多个 batch 记录（重试），取最大 batch 的那条
        return max(lst, key=lambda r: num(r, "batch") or 0)

    lines = []

    # ---------------- 绝对值表 ----------------
    lines.append("# 一、绝对值（fwd+bwd 训练态，bf16 autocast）\n")
    lines.append("| stage | mask | branch | seq | batch | compile | impl "
                 "| ms_fp | ms_step | ktok/s | TFLOP/s(满算) "
                 "| 设备峰值MiB | 分配器峰值MiB |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, d in sorted(idx.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                     str(kv[0][2]), kv[0][3],
                                                     str(kv[0][5]))):
        stage, mask, branch, seq, batch, compiled = key
        for impl in sorted(keep_abs):
            r = one(d, impl)
            if r is None:
                continue
            lines.append(
                f"| {stage} | {mask} | {branch} | {seq} "
                f"| {r.get('batch', batch)} | {compiled} | {impl} "
                f"| {fmt(num(r, 'ms_fp'))} | {fmt(num(r, 'ms_step'))} "
                f"| {fmt(num(r, 'ktok_s'), 1)} | {fmt(num(r, 'tflops_full'), 2)} "
                f"| {fmt(mem_of(r), 1)} | {fmt(num(r, MEM_FALLBACK), 1)} |")

    # ---------------- capacity 单独成表 ----------------
    cap_keys = [k for k in idx if k[0] == "capacity"]
    if cap_keys:
        lines.append("\n# 二、capacity：同显存预算下能开多大 batch\n")
        lines.append("| mask | seq | compile | impl | 最大 batch | tokens/step "
                     "| ms_step | ktok/s | 设备峰值MiB |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        cap_rows = {}
        for key in sorted(cap_keys, key=lambda k: (k[1], k[3], str(k[5]))):
            stage, mask, branch, seq, _b, compiled = key
            d = idx[key]
            for impl in ("std", "sdpa"):
                r = one(d, impl)
                if r is None:
                    continue
                lines.append(f"| {mask} | {seq} | {compiled} | {impl} "
                             f"| {r.get('batch')} | {int(float(r['batch']) * seq)} "
                             f"| {fmt(num(r, 'ms_step'), 2)} "
                             f"| {fmt(num(r, 'ktok_s'), 1)} "
                             f"| {fmt(mem_of(r), 1)} |")
                cap_rows[(mask, seq, compiled, impl)] = r
        for (mask, seq, compiled), _ in {(k[0], k[1], k[2]): 0 for k in cap_rows}.items():
            s = cap_rows.get((mask, seq, compiled, "std"))
            r = cap_rows.get((mask, seq, compiled, "sdpa"))
            if s is None or r is None:
                continue
            lines.append(
                f"| {mask} | {seq} | {compiled} | **sdpa/std** "
                f"| {float(r['batch']) / float(s['batch']):.2f}x "
                f"| {int(float(r['batch']) * seq) / int(float(s['batch']) * seq):.2f}x "
                f"| - | {ratio(num(r, 'ktok_s'), num(s, 'ktok_s'))} "
                f"| {ratio(mem_of(s), mem_of(r))} |")

    # ---------------- 比值表 ----------------
    lines.append("\n# 三、比值（一律 std/sdpa：耗时、吞吐 >1 表示 sdpa 更快；"
                 "显存列 >1 表示 sdpa 更省）\n")
    lines.append("| stage | mask | branch | seq | batch | compile | 对比 "
                 "| 前向加速 | 单步加速 | 吞吐提升 | 显存节省 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    chart = defaultdict(dict)
    for key, d in sorted(idx.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                     str(kv[0][2]), kv[0][3],
                                                     str(kv[0][5]))):
        stage, mask, branch, seq, batch, compiled = key
        if stage == "capacity":
            continue
        base = one(d, BASE_IMPL)
        if base is None:
            continue
        for impl in sorted(d):
            if impl == BASE_IMPL:
                continue
            r = one(d, impl)
            a, b = num(base, "ms_step"), num(r, "ms_step")
            if a is None or b is None:
                lines.append(f"| {stage} | {mask} | {branch} | {seq} "
                             f"| {base.get('batch')} | {compiled} | {impl}/{BASE_IMPL} "
                             f"| OOM | OOM | - | - |")
                continue
            am, bm = mem_of(base), mem_of(r)
            lines.append(
                f"| {stage} | {mask} | {branch} | {seq} | {r.get('batch')} | {compiled} "
                f"| {impl}/{BASE_IMPL} "
                f"| {ratio(num(base, 'ms_fp'), num(r, 'ms_fp'))} "
                f"| {ratio(a, b)} "
                f"| {ratio(num(r, 'ktok_s'), num(base, 'ktok_s'))} "
                f"| {ratio(am, bm) if am and bm else '-'} |")
            if stage in ("core", "module", "model"):
                chart[(stage, mask, compiled)][(impl, seq)] = a / b

    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"已写出 {args.out}")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = sorted(chart)
        fig, axes = plt.subplots(1, len(groups),
                                 figsize=(4.3 * len(groups), 3.4), squeeze=False)
        for ax, gk in zip(axes[0], groups):
            for impl in sorted({k[0] for k in chart[gk]}):
                pts = sorted((k[1], v) for k, v in chart[gk].items()
                             if k[0] == impl)
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                        label=impl)
            ax.axhline(1.0, color="k", lw=.8, ls="--")
            ax.set_title(f"{gk[0]} mask={gk[1]} "
                         f"{'compile' if gk[2] else 'eager'}", fontsize=9)
            ax.set_xlabel("seq_len")
            ax.set_ylabel("std/sdpa 单步耗时比")
            ax.set_xscale("log", base=2)
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        png = Path(args.out).with_suffix(".png")
        fig.savefig(png, dpi=140)
        print(f"图已写出 {png}")


if __name__ == "__main__":
    main()
