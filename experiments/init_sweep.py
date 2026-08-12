"""
权重初始化对比实验

对每个初始化方案独立构建模型（同种子），用与 pre_train.py 对齐的精简管线
训练 target_steps 步，收集 train/val loss，对比谁收敛更快、更低。

用法:
    uv run python experiments/init_sweep.py                     # 跑全部方案
    uv run python experiments/init_sweep.py --methods gpt2_002 xavier_normal
    uv run python experiments/init_sweep.py --steps 120           # 更快的小规模
    uv run python experiments/init_sweep.py --quick             # smoke：仅 default，60 步
"""

import argparse
import csv
import gc
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端

import matplotlib.pyplot as plt
import torch

from common import ExperimentConfig, ExperimentTrainer, apply_init, set_seed

ALL_METHODS = ["default", "gpt2_002", "gpt2_002_rescale", "xavier_normal"]

OUTDIR = Path(__file__).resolve().parent / "results"


def parse_args():
    p = argparse.ArgumentParser(description="初始化对比实验")
    p.add_argument("--methods", nargs="*", default=ALL_METHODS,
                   help="要跑的初始化方案子集")
    p.add_argument("--steps", type=int, default=None, help="覆盖 target_steps")
    p.add_argument("--seeds", type=int, nargs="*", default=[42],
                   help="多个随机种子，跨种子聚合更可靠")
    p.add_argument("--quick", action="store_true", help="smoke：仅 default，steps=60")
    p.add_argument("--d_model", type=int, default=None,
                   help="覆盖模型宽度（跨尺寸可迁移性实验用）；d_inner 自动取 8/3·d")
    p.add_argument("--batch_size", type=int, default=None,
                   help="覆盖 batch_size（大模型防 OOM）")
    p.add_argument("--tag", type=str, default=time.strftime("%Y%m%d-%H%M%S"),
                   help="结果文件名后缀")
    return p.parse_args()


def main():
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    methods = ["default"] if args.quick else list(args.methods)
    steps = args.steps if args.steps is not None else (60 if args.quick else 150)
    seeds = args.seeds

    print(f"=== 初始化实验 tag={args.tag} methods={methods} "
          f"seeds={seeds} steps={steps} ===")

    summary_csv = OUTDIR / f"init_sweep_{args.tag}.csv"
    csv_file = open(summary_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["seed", "method", "d_model", "steps", "elapsed_s",
                         "train_final", "train_last50", "train_first",
                         "val_loss", "ppl", "top1"])

    per_seed = {}                  # (d_model, method) -> [(seed, val_loss), ...]
    d_model = args.d_model or ExperimentConfig().d_model

    for seed in seeds:
        print(f"\n===== seed = {seed} =====")
        for method in methods:
            print(f"\n### method = {method}")
            cfg = ExperimentConfig()
            cfg.seed = seed
            cfg.target_steps = steps
            if args.d_model is not None:
                cfg.d_model = args.d_model
                cfg.d_inner = max(256, int(round(args.d_model * 8 / 3)))
            if args.batch_size is not None:
                cfg.batch_size = args.batch_size
            set_seed(seed)
            trainer = ExperimentTrainer(cfg)          # 同种子，数据/优化器一致
            apply_init(trainer.model, method, cfg.n_layers)

            result = trainer.run()
            series = result["loss_series"]
            tail = [l for _, l in series[-50:]]
            row = [seed, method, cfg.d_model, result["steps"], round(result["elapsed"], 1),
                   round(series[-1][1], 4) if series else "",
                   round(sum(tail) / len(tail), 4) if tail else "",
                   round(series[0][1], 4) if series else "",
                   round(result["final_metrics"]["val_loss"], 4),
                   round(result["final_metrics"]["ppl"], 3),
                   round(result["final_metrics"]["top1"], 4)]
            csv_writer.writerow(row)
            csv_file.flush()
            per_seed.setdefault((cfg.d_model, method), []).append(
                (result["final_metrics"]["val_loss"], row[6], result["final_metrics"]["top1"]))
            print(
                f"  DONE method={method} "
                f"val_loss={result['final_metrics']['val_loss']:.4f} "
                f"train_last50={row[5]} ({result['elapsed']:.0f}s)"
            )
            del trainer, result, series, tail
            gc.collect()
            torch.cuda.empty_cache()

    csv_file.close()
    _plot_from_csv(summary_csv, args.tag, seeds, steps)
    _print_multi_seed_table(per_seed)
    print(f"\n结果已写: {summary_csv}")
    print(f"曲线图: {OUTDIR / f'init_sweep_{args.tag}.png'}")


def _plot_from_csv(csv_path, tag, seeds, steps):
    """读取 CSV，绘制每个 method 的 val_loss 均值（±seed 范围）柱状图"""
    data = {}   # method -> [val_loss,...]
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["method"], []).append(float(row["val_loss"]))

    png = OUTDIR / f"init_sweep_{tag}.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = sorted(data, key=lambda m: sum(data[m]) / len(data[m]))
    means = [sum(v) / len(v) for v in (data[m] for m in methods)]
    spreads = [max(v) - min(v) for v in (data[m] for m in methods)]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    ax.bar(range(len(methods)), means, yerr=spreads, capsize=5, width=0.6,
           color=colors[:len(methods)], alpha=0.85)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("avg val loss (lower=better)")
    ax.set_title(f"Weight init comparison ({steps} steps, {len(seeds)} seeds)")
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def _print_multi_seed_table(per_seed: dict):
    rows = []
    for (dmodel, method), v in per_seed.items():
        vloss = [x[0] for x in v]
        t50 = [x[1] for x in v]
        top1 = [x[2] for x in v]
        rows.append((method, dmodel,
                     sum(t50) / len(t50),
                     sum(vloss) / len(vloss),
                     (max(vloss) - min(vloss)) / 2,
                     sum(top1) / len(top1)))
    rows.sort(key=lambda r: (r[1], r[3]))
    print("\n" + "=" * 88)
    print(f"{'method':<26}{'d':>5}{'train_last50':>13}{'val_loss(avg)':>14}{'±range':>9}{'top1':>7}")
    print("-" * 88)
    for method, d, t50, vloss, spread, top1 in rows:
        print(f"{method:<26}{d:>5}{t50:>13.4f}{vloss:>14.4f}{spread:>9.4f}{top1:>7.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()