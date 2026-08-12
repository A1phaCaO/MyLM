"""
实验用训练管线 —— 与仓库根目录 pre_train.py 对齐（精简版）

对齐点：
- 数据：PretrainTokenIDDataset 加载 .npy (uint16)，mask 加权交叉熵
- 模型：models.py 的 MyLM / MyLMArgs（同一架构，仅缩小规模）
- 优化器：Muon(2D 权重, match_rms_adamw) + AdamW8bit(其余) 双优化器
- 调度器：WarmUpStableDecayLR (WSD, linear decay)
- AMP：bfloat16 autocast + GradScaler
- 梯度累积：batch_acceleration，clip 1.0，loss 除以累积倍数

仅此文件夹内的文件会被修改；根目录代码只读导入。
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import contextlib
import io
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

from models import MyLM, MyLMArgs
from dataset import PretrainTokenIDDataset
from utils import WarmUpStableDecayLR


@dataclass
class ExperimentConfig:
    """小规模实验配置（对齐 TrainingConfig 的字段含义，仅规模缩小）"""

    # 数据
    data_dir: str = str(ROOT / "mini_data192mixen_v3.npy")
    tokenizer_dir: str = str(ROOT / "bbpe_tokenizer_7k_260723_xl.json")
    downsample: float = 0.02       # 数据集采样率（~21k 条，与模型尺寸匹配，避免快速过拟合）
    batch_size: int = 128          # dense 小模型，显存余量足，取大批量提速
    batch_acceleration: int = 4    # 梯度累积，有效 batch = 256
    seq_max_len: int = 128         # 需 <= 数据存储列(193)；缩小编短步长
    val_rate: float = 0.01
    padding_side: str = "right"

    # 优化
    seed: int = 42
    learning_rate: float = 4e-3
    min_learning_rate: float = 4e-4
    lr_decay_start_rate: float = 0.75
    warmup_steps: int = 20
    use_amp: bool = True

    # 模型（更小 dense，快速；d_inner ≈ 8/3·d_model）
    d_model: int = 256
    d_latent: int = 256
    d_inner: int = 768
    n_layers: int = 6
    d_head: int = 64               # n_heads = d_model // d_head = 4
    n_heads: int = None
    n_experts: int = 6             # 仅 MoE 用，dense 下无效
    n_experts_per_tok: int = 3
    use_moe: bool = False
    latent_moe: bool = False
    dropout: float = 0.0

    # 训练长度
    target_steps: int = 150        # 期望的优化器 step 数
    eval_interval: int = 50        # 每隔多少优化器 step 验证一次


def build_my_lm_args(cfg: ExperimentConfig, vocab_size: int) -> MyLMArgs:
    return MyLMArgs(
        d_model=cfg.d_model,
        d_inner=cfg.d_inner,
        n_layers=cfg.n_layers,
        latent_moe=cfg.latent_moe,
        d_latent=cfg.d_latent,
        vocab_size=vocab_size,
        seq_max_len=cfg.seq_max_len,
        use_moe=cfg.use_moe,
        n_heads=cfg.n_heads,
        n_experts=cfg.n_experts,
        n_experts_per_tok=cfg.n_experts_per_tok,
        d_conv=None,
        conv_bias=None,
        ffn_bias=False,
        attn_bias=True,
        dropout=cfg.dropout,
        base_init_std=0.02,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class MemDataset(torch.utils.data.Dataset):
    """内存版 token-ID 数据集：一次性把采样到的行拷入 RAM 并切片成 (input, target, mask)。
    避免 PretrainTokenIDDataset 每轮重复 mmap 读取 + 每行 Python pad 的 CPU 开销，
    大幅提升小模型实验的吞吐（瓶颈从 CPU 数据加载变为 GPU 计算）。
    约定：存储列数 = seq_max_len + 1（与 generate_dataset_v3 一致）。
    """

    def __init__(self, data_dir, seq_max_len, downsample, seed, pad_value=0):
        data = np.load(data_dir, mmap_mode="r")
        n = data.shape[0]
        if downsample < 1.0:
            rng = np.random.RandomState(seed)
            k = max(1, int(n * downsample))
            idx = rng.choice(n, size=k, replace=False)
        else:
            idx = np.arange(n)
        rows = np.array(data[idx], dtype=np.uint16)       # (k, storage)
        seq = rows[:, :seq_max_len]                        # (k, seq)
        self.inputs = torch.from_numpy(seq[:, :seq_max_len - 1].astype(np.int64)).clone()
        self.targets = torch.from_numpy(seq[:, 1:seq_max_len].astype(np.int64)).clone()
        self.mask = (self.targets != pad_value).float()

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i], self.mask[i]


class ExperimentTrainer:
    """与 PreTrainer 训练语义对齐的精简训练器"""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizers.Tokenizer.from_file(cfg.tokenizer_dir)
        self.vocab_size = int(len(self.tokenizer.get_vocab()))
        self._build_dataloader()
        # models.py FFN._reset_parameters 有调试 print，构建时静默掉
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = MyLM(build_my_lm_args(cfg, self.vocab_size)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self._build_optimizer()
        self.scaler = torch.GradScaler(self.device, enabled=cfg.use_amp)

    # ---------- 构建 ----------
    def _build_dataloader(self):
        dataset = MemDataset(
            self.cfg.data_dir,
            seq_max_len=self.cfg.seq_max_len,
            downsample=self.cfg.downsample,
            seed=self.cfg.seed,
        )
        n_val = max(1, int(len(dataset) * self.cfg.val_rate))
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [len(dataset) - n_val, n_val]
        )
        # 数据已在内存，用少量 worker 并行拼 batch 即可
        self.train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        # 优化器 step 数（对齐 pre_train 的 steps_per_epoch 算法）
        self.steps_per_epoch = (
            len(self.train_loader) + self.cfg.batch_acceleration - 1
        ) // self.cfg.batch_acceleration
        self.epochs = max(
            1, math.ceil(self.cfg.target_steps / self.steps_per_epoch))
        self.total_steps = self.epochs * self.steps_per_epoch + 1

    def _build_optimizer(self):
        muon_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 2:
                if ("embedding" in name.lower() or "embed" in name.lower()
                        or "head" in name.lower() or "classifier" in name.lower()):
                    other_params.append(param)
                elif "weight" in name and "bias" not in name:
                    muon_params.append(param)
                else:
                    other_params.append(param)
            else:
                other_params.append(param)

        self.optimizers = [
            torch.optim.Muon(
                muon_params,
                lr=self.cfg.learning_rate,
                adjust_lr_fn="match_rms_adamw",
                weight_decay=0.008,
            ),
            bnb.optim.adamw.AdamW8bit(
                other_params,
                lr=self.cfg.learning_rate,
                amsgrad=False,
                betas=(0.85, 0.999),
                eps=1e-6,
                weight_decay=0.008,
            ),
        ]
        self.schedulers = [
            WarmUpStableDecayLR(
                opt,
                total_steps=self.total_steps,
                warmup_steps=self.cfg.warmup_steps,
                stable_steps=int(
                    self.cfg.lr_decay_start_rate * self.total_steps
                    - self.cfg.warmup_steps
                ),
                min_lr=self.cfg.min_learning_rate,
                decay_mode="linear",
            )
            for opt in self.optimizers
        ]

    # ---------- 训练 ----------
    def _train_step(self, inputs, targets, mask):
        """单次微批次：前向+反向+累积；到达边界时更新。返回 (loss, 是否更新)"""
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        mask = mask.to(self.device)

        with torch.autocast(str(self.device), enabled=self.cfg.use_amp, dtype=torch.bfloat16):
            output = self.model(inputs)
            loss = self.criterion(
                output.view(-1, self.vocab_size), targets.view(-1))
            loss = (loss * mask.view(-1)).sum() / mask.sum()

        loss = loss / self.cfg.batch_acceleration
        self.scaler.scale(loss).backward()

        if self.micro_step % self.cfg.batch_acceleration == 0:
            for opt in self.optimizers:
                self.scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            for opt in self.optimizers:
                self.scaler.step(opt)
            self.scaler.update()
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)
            for sched in self.schedulers:
                sched.step()
            return loss.item() * self.cfg.batch_acceleration, True
        return loss.item() * self.cfg.batch_acceleration, False

    @torch.no_grad()
    def evaluate(self):
        """验证集 masked CE + PPL + top1（对齐 pre_train.validate 的核心指标）"""
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        n = 0
        for inputs, targets, mask in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)
            with torch.autocast(str(self.device), enabled=self.cfg.use_amp, dtype=torch.bfloat16):
                logits = self.model(inputs).view(-1, self.vocab_size)
            loss = self.criterion(logits, targets.view(-1))
            loss = (loss * mask.view(-1)).sum() / mask.sum()
            loss_sum += loss.item()
            valid = mask.view(-1) > 0
            pred = logits.argmax(-1)
            correct += (pred == targets.view(-1))[valid].sum().item()
            total += valid.sum().item()
            n += 1
        avg = loss_sum / max(n, 1)
        return {"val_loss": avg, "ppl": math.exp(avg), "top1": correct / max(total, 1)}

    def run(self, log_every=10, on_log=None):
        """主训练循环。on_log(step, loss) 每次优化器 step 后回调（用于收集曲线）。
        返回 {loss_series, final_metrics, steps, elapsed}"""
        self.micro_step = 0
        step = 0
        loss_series = []
        metrics_history = []
        t0 = time.perf_counter()
        last_log_t = t0
        last_log_step = 0

        for epoch in range(self.epochs):
            for inputs, targets, mask in self.train_loader:
                self.micro_step += 1
                loss, updated = self._train_step(inputs, targets, mask)
                if not updated:
                    continue
                step += 1
                loss_series.append((step, loss))
                if on_log is not None:
                    on_log(step, loss)
                if step % self.cfg.eval_interval == 0:
                    m = self.evaluate()
                    metrics_history.append((step, m))
                    now = time.perf_counter()
                    dw = now - last_log_t
                    sps = (step - last_log_step) / max(dw, 1e-9)
                    eta = (self.cfg.target_steps - step) / max(sps, 1e-9)
                    last_log_t, last_log_step = now, step
                    print(
                        f"  [step {step}/{self.total_steps}] train_loss: {loss:.4f} "
                        f"val_loss: {m['val_loss']:.4f} top1: {m['top1']:.3f} "
                        f"[{sps:.2f} step/s, ETA {eta:.0f}s] "
                        f"lr: {self.schedulers[0].get_last_lr()[0]:.2e}",
                        flush=True,
                    )
                if step >= self.cfg.target_steps:
                    break
            if step >= self.cfg.target_steps:
                break

        final_metrics = self.evaluate()
        return {
            "loss_series": loss_series,
            "metrics_history": metrics_history,
            "final_metrics": final_metrics,
            "steps": step,
            "elapsed": time.perf_counter() - t0,
        }


# ---------------------------------------------------#
#   初始化方案定义（奥卡姆：全部是几行 torch.nn.init 调用）
# ---------------------------------------------------#

# 残差输出投影（其输出直接累加到残差流）
RESID_KEYS = ("o_proj", "down_proj", "latent_up")


def apply_init(model: nn.Module, scheme: str, n_layers: int):
    """按 scheme 重置模型权重。scheme:
    - "default"        : 不做任何覆盖，即 models.py 当前行为
                         （层线性层 N(0,0.02)，embedding N(0,1/sqrt(d))，head kaiming，router N(0,0.01)）
    - "gpt2_002"       : 所有 Linear/Embedding 权重 N(0, 0.02)，bias 置零（GPT-2 配方）
    - "gpt2_002_rescale": 在 gpt2_002 基础上，残差输出投影再乘 1/sqrt(2*n_layers)
    - "xavier_normal"  : 所有 Linear 用 xavier_normal_(gain=1)，Embedding N(0,0.02)，bias 保持默认

    可缩放配置（冒号后逗号分隔）：
    - std{float}  : 所有 Linear 权重的基础标准差（默认 0.02）
    - emb{float}  : Embedding 的标准差（绝对）
    - embr{float} : Embedding 标准差 = 基础 std × 比值（相对）
    embr 与 emb 同时给出时后者生效。例："gpt2_002_rescale:std0.01,embr0.5"。
    """
    base, _, opt = scheme.partition(":")
    base_std = 0.02
    emb_std = None
    emb_ratio = None
    if opt:
        for part in opt.split(","):
            if part.startswith("std"):
                base_std = float(part[3:])
            elif part.startswith("embr"):
                emb_ratio = float(part[4:])
            elif part.startswith("emb"):
                emb_std = float(part[3:])

    if base == "default":
        return
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if base in ("gpt2_002", "gpt2_002_rescale"):
                nn.init.normal_(mod.weight, std=base_std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
                if base == "gpt2_002_rescale" and any(
                    k in name for k in RESID_KEYS
                ):
                    mod.weight.data.mul_(1.0 / math.sqrt(2 * n_layers))
            elif base == "xavier_normal":
                nn.init.xavier_normal_(mod.weight)
        elif isinstance(mod, nn.Embedding):
            if emb_std is None:
                emb_std = base_std * emb_ratio if emb_ratio is not None else 0.02
            nn.init.normal_(mod.weight, std=emb_std)
        elif isinstance(mod, nn.GRU):
            for pname, p in mod.named_parameters():
                if "weight" in pname:
                    nn.init.normal_(p, std=base_std)
                else:
                    nn.init.zeros_(p)
