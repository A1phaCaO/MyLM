# -*- coding: utf-8 -*-
"""验证 batch64 x1(无累积) 与 batch32 x2(梯度累积) 是否等价的复现测试。

两个独立 run 仅 batch_size/batch_acceleration 不同，其余（seed、数据、
LR 调度、优化器、模型初始化）完全一致，且消费的数据序列按索引顺序相同：
  - A: batch_size=64, batch_acceleration=1
  - B: batch_size=32, batch_acceleration=2
对比方式：
  [公平对齐] A 的优化步 k  vs  B 的边界 loss（global_step=2k+1，同一批 64 条数据）
  [用户式对齐] 同一 global_step 横轴（B 只消费了 A 一半的数据量）
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根（tools/checks/ 上两级）

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pre_train import PreTrainer, TrainingConfig
from models import MyLMArgs


def make_cfg(batch_size, batch_acceleration, log_dir):
    cfg = TrainingConfig(
        data_dir=r"data/medium_data256v2.npy",
        tokenizer_dir=r"tokenizer/bbpe_tokenizer_7k_260723_xl.json",
        log_dir=log_dir,
        seed=42,
        epochs=1,
        batch_size=batch_size,
        batch_acceleration=batch_acceleration,
        dataset_downsample=0.03,
        valset_rate=0.005,
        val_interval_step=10 ** 9,
        ckpt_interval_step=10 ** 9,
        use_compile=False,
        use_amp=True,
        learning_rate=1.5e-3,
        min_learning_rate=1.5e-4,
        lr_decay_start_rate=0.75,
        warmup_steps=5,
        dataset_shuffle_seed=42,
    )
    cfg.seq_max_len = 128
    cfg.model_args = MyLMArgs(
            d_model=128,
            latent_moe=False,
            d_latent=64,
            d_inner=320,
            d_head=64,
            n_heads=None,
            n_layers=2,
            vocab_size=None,
            seq_max_len=128,
            use_moe=False,
            n_experts=8,
            n_experts_per_tok=2,
            d_conv=None,
            conv_bias=None,
            ffn_bias=False,
            attn_bias=True,
            dropout=0.0,
            base_init_std=0.02,
        )
    return cfg


def run(cfg):
    trainer = PreTrainer(cfg)
    trainer.train()
    return trainer.train_loss_log


def main():
    print("=== run A: batch_size=64, accum=1 ===", flush=True)
    log_a = run(make_cfg(64, 1, r"logs\test_accum_a"))
    print("=== run B: batch_size=32, accum=2 ===", flush=True)
    log_b = run(make_cfg(32, 2, r"logs\test_accum_b"))

    gs_a = np.array([s for s, _ in log_a])
    loss_a = np.array([l for _, l in log_a], dtype=np.float64)
    gs_b = np.array([s for s, _ in log_b])
    loss_b = np.array([l for _, l in log_b], dtype=np.float64)

    n = min(len(loss_a), (len(loss_b) - 1) // 2)
    b_boundary = loss_b[1::2][:n]
    a_fair = loss_a[:n]
    diff_fair = b_boundary - a_fair

    print("\n[公平对齐] B 边界 loss(global_step=2k+1) vs A loss(step=k)：同一批 64 条数据")
    for k in range(0, n, 50):
        print(f"  step {k:5d}: A={a_fair[k]:.6f}  B={b_boundary[k]:.6f}  d={diff_fair[k]:+.6f}")
    print(f"  MAE={np.abs(diff_fair).mean():.2e}  max|d|={np.abs(diff_fair).max():.2e}")

    print("\n[用户式对齐] 同一 global_step 横轴（B 只消费了 A 一半的数据量）")
    m = min(len(loss_a), len(loss_b))
    diff_user = loss_b[:m] - loss_a[:m]
    for k in range(0, m, 50):
        print(f"  step {k:5d}: A={loss_a[k]:.6f}  B={loss_b[k]:.6f}  d={diff_user[k]:+.6f}")
    print(f"  平均差={diff_user.mean():+.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gs_a, loss_a, label="A: batch64 x1 (step = optimizer step)")
    ax.plot(gs_b, loss_b, label="B: batch32 x2 (step = micro-batch)", alpha=0.6)
    ax.plot(gs_b[1::2][:n], b_boundary, ls="--", lw=2,
            label="B boundary loss (aligned to A)")
    ax.set_xlabel("global_step (micro-batch count)")
    ax.set_ylabel("train loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(r"logs\test_accum_equiv.png", dpi=120)
    print("\n图已保存 logs/test_accum_equiv.png")


if __name__ == "__main__":
    main()