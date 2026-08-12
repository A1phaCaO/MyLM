"""
聚合 experiments/results/init_sweep_*.csv 中所有 (seed, method) 数据，
按初始化方案输出跨种子均值表（val_loss / train_last50 / top1）。

用法: uv run python experiments/report_merge.py [--min-val-rows 100]
"""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent / "results"


def short_name(method: str) -> str:
    """把 scheme 字符串变短，便于阅读"""
    return (method.replace("gpt2_002_rescale", "gpt2rs")
                 .replace("gpt2_002", "gpt2")
                 .replace(":std", ",s=")
                 .replace(":emb", ",emb=")
                 .replace(":embr", ",r=")
                 .replace(":", ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100,
                   help="steps 等于该值的运行才纳入统计（默认 100，剔除 60/150 等不齐的对比）")
    p.add_argument("--tags", nargs="*", default=None,
                   help="只统计指定 tag 的 CSV（默认全部；推荐指定本次配置的 tag 保持一致模型）")
    args = p.parse_args()

    data = defaultdict(list)   # method -> list of (val_loss, train_last50, top1, steps)
    for path in sorted(glob.glob(str(OUTDIR / "init_sweep_*.csv"))):
        tag = Path(path).stem.replace("init_sweep_", "")
        if args.tags and not any(t in tag for t in args.tags):
            continue
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    steps = int(row["steps"])
                except ValueError:
                    continue
                if steps != args.steps:
                    continue
                if not row.get("val_loss"):
                    continue
                data[row["method"]].append((
                    float(row["val_loss"]),
                    float(row["train_last50"]),
                    float(row["top1"]),
                    steps,
                ))

    rows = []
    for method, runs in data.items():
        v = [r[0] for r in runs]
        t = [r[1] for r in runs]
        t1 = [r[2] for r in runs]
        rows.append((method, len(runs),
                     sum(v) / len(v), min(v), (max(v) - min(v)) / 2,
                     sum(t) / len(t),
                     sum(t1) / len(t1)))
    rows.sort(key=lambda r: r[2])

    print(f"（仅统计 steps>={args.steps} 的运行）")
    print(f"{'scheme':<34}{'n':>3}{'val_loss_avg':>12}{'best':>8}{'±range':>9}"
          f"{'train_avg':>11}{'top1_avg':>9}")
    print("-" * 90)
    for method, n, va, vb, vr, ta, cc in rows:
        print(f"{short_name(method):<34}{n:>3}{va:>12.4f}{vb:>8.4f}{vr:>9.4f}"
              f"{ta:>11.4f}{cc:>9.4f}")
    print("-" * 90)
    print(f"共 {len(rows)} 个方案、{sum(len(v) for v in data.values())} 次运行")


if __name__ == "__main__":
    main()