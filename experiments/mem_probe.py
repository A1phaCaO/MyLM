"""显存探测：在目标模型尺寸(seq=192, d=512, MoE 6x3)下测量不同 batch 的峰值显存。
用法: uv run python experiments/mem_probe.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from common import ExperimentConfig, ExperimentTrainer, set_seed


def probe(cfg, batch_size, steps=3):
    set_seed(cfg.seed)
    cfg.batch_size = batch_size
    trainer = ExperimentTrainer(cfg)
    # 用全 1 输入 + 随机 target 做 forward/backward 测峰值
    inputs = torch.randint(1, trainer.vocab_size, (batch_size, cfg.seq_max_len)).to(trainer.device)
    targets = torch.randint(1, trainer.vocab_size, (batch_size, cfg.seq_max_len)).to(trainer.device)
    mask = torch.ones(batch_size, cfg.seq_max_len).to(trainer.device)
    for i in range(steps):
        trainer.model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            out = trainer.model(inputs)
            loss = F.cross_entropy(out.view(-1, trainer.vocab_size), targets.view(-1))
        loss.backward()
    alloc = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    print(f"batch={batch_size:4d}  peak_allocated={alloc:7.1f} MiB  reserved={reserved:7.1f} MiB")
    del trainer, inputs, targets, mask
    torch.cuda.empty_cache()
    return alloc


def bench_compute(cfg, batch_size, steps=10):
    """纯 forward+backward 计时（含梯度累积，不含 dataloader）"""
    set_seed(cfg.seed)
    cfg.batch_size = batch_size
    trainer = ExperimentTrainer(cfg)
    inputs = torch.randint(1, trainer.vocab_size, (batch_size, cfg.seq_max_len)).to(trainer.device)
    targets = torch.randint(1, trainer.vocab_size, (batch_size, cfg.seq_max_len)).to(trainer.device)
    mask = torch.ones(batch_size, cfg.seq_max_len).to(trainer.device)
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    # warmup
    for _ in range(2):
        trainer.model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            out = trainer.model(inputs)
            loss = F.cross_entropy(out.view(-1, trainer.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        trainer.model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            out = trainer.model(inputs)
            loss = F.cross_entropy(out.view(-1, trainer.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    n_tok = batch_size * cfg.seq_max_len
    print(f"batch={batch_size:4d}  fwd+bwd={dt*1000:6.1f} ms/step  {n_tok/dt/1e6:6.1f} k-tok/s")
    del trainer; torch.cuda.empty_cache()


if __name__ == "__main__":
    cfg = ExperimentConfig()
    print(f"模型: d={cfg.d_model} latent={cfg.d_latent} layers={cfg.n_layers} "
          f"MoE={cfg.n_experts}x{cfg.n_experts_per_tok} seq={cfg.seq_max_len} 设备: {torch.cuda.get_device_name(0)}")
    for bs in [32, 64, 96]:
        bench_compute(cfg, bs)
    for bs in [32, 48, 64, 80, 96]:
        probe(cfg, bs)
