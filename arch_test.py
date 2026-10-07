# -*- coding: utf-8 -*-
"""架构测试管线：pre_train.py 的精简版，用于快速验证新模型结构。

与 pre_train.py 保持一致的机制:
  - dataclass 单一配置源；vocab_size 由 tokenizer 注入
  - PretrainTokenIDDataset(mmap) + random_split + 固定排列 + drop_last
  - 双优化器: torch.optim.Muon(2D 权重, match_rms_adamw) + bnb AdamW8bit(其余)
  - WSD 调度器(WarmUpStableDecayLR)，梯度累积边界处 step
  - bf16 autocast + GradScaler；token 加权累积（边界统一除 Σden）；clip 1.0
  - zero_grad(set_to_none=False) + 预分配 .grad 缓冲；多卡 DataParallel
  - 详细的训练分析统计（本脚本的重点，全部进 TensorBoard）:
      * 训练效果: Loss/train(+EMA)、Loss/val、Val/ppl、Val/entropy、top1/top5
      * 稳定性:   GradNorm/raw、Grad/{max_abs,mean_abs,zero_ratio}、
                  GradNorm/per_layer 直方图、Weight/{norm,max_abs,std,sparse_ratio}、
                  WeightNorm/per_layer 直方图、Stability/loss_vs_ema（尖峰检测）
      * MoE 负载: MoE/drop_rate、router_entropy、expert_load 直方图（自动挂载）
      * 速度:     Perf/tok_s、Perf/step_ms
      * 定性:     每 N 次验证做一次文本生成探针，写入 GeneratedText
为架构测试砍掉的部分:
  - 断点续训整套机制（resume_from / ckpt 保存清理 / RNG 状态存取 / 数据定位）
    架构测试规模小、一跑到底，不需要中途存档
  - MoE 热力图/3D 图（matplotlib 全套）、plot_losses、结尾交互式 REPL
换模型方式: 改本文件顶部「换模型」常量区的 build_model，
唯一契约是 forward(x_ids, padding_mask=...) -> (B,S,V) logits。

运行:
    uv run python arch_test.py
    # 开 compile 前: $env:PYTHONUTF8='1'  (GBK 下 inductor 模板读取崩溃)
日志: logs/exp/<exp_name>_<时间戳>/   多次实验并排看:
    tensorboard --logdir logs/exp
调试小样本: 追加 JSON 覆盖配置（PowerShell 下建议写成文件传路径），如
    uv run python arch_test.py override.json
    # override.json: {"batch_size":8,"max_steps":100,"val_interval_step":50}
"""
import os
import sys
import gc
import json
import math
import time
import random
from dataclasses import dataclass, asdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import tokenizers
import bitsandbytes as bnb
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import MyLMArgs, MyLM
from dataset import PretrainTokenIDDataset
from utils import WarmUpStableDecayLR, MoEStatsCollector, TextGenerator

# ============================================================
#   换模型：改这一处即可
# ============================================================
# 契约: build_model(args: MyLMArgs) -> nn.Module
#       model.forward(x_ids:(B,S) long, padding_mask:(B,S) bool|None) -> (B,S,V) logits
# args 里 vocab_size / seq_max_len 已由管线注入，其余来自 MODEL_ARGS。
MODEL_NAME = r"mylm_moe_half_baseline(wo compile)"

MODEL_ARGS = MyLMArgs(
    d_model=384,
    latent_moe=True,
    d_latent=192,
    d_inner=int(((192 * (8 / 3)) // 64) * 64),
    d_head=128,
    n_heads=None,
    n_layers=6,
    vocab_size=None,          # 由 tokenizer 自动注入，勿硬编码
    seq_max_len=256,
    use_moe=True,
    n_experts=8,
    n_experts_per_tok=2,
    moe_capacity=1.25,
    d_conv=4,
    compress_ratio=8,
    ffn_bias=False,
    attn_bias=True,
    dropout=0.05,
    base_init_std=0.02,
)


def build_model(args: MyLMArgs) -> nn.Module:
    """架构测试的模型入口。示例：
      - 换 MoE:      from dataclasses import replace
                     return MyLM(replace(args, use_moe=True))
      - 全新结构:    return MyNewArch(d_model=args.d_model, ...)
    """
    return MyLM(args)


# ============================================================
#   训练/实验配置
# ============================================================

@dataclass
class ArchTestConfig:
    # 数据
    data_dir: str = r"data/medium_data256v3.npy"
    tokenizer_dir: str = r"tokenizer/bbpe_tokenizer_7k_260723_xl.json"
    padding_side: str = "right"
    dataset_downsample: float = 0.04        # 架构测试默认全量 + max_steps 早停
    valset_rate: float = 0.008        # 验证集比例（配合 max_steps 控制 val 时长）
    batch_size: int = 64
    seq_max_len: int = 256

    # 实验组织
    exp_name: str = MODEL_NAME         # logs/exp/<exp_name>_<时间戳>
    max_steps: int = 0                 # 优化器步数上限，0=跑满 epochs（早停用）
    seed: int = 42

    # 训练
    epochs: int = 1
    batch_acceleration: int = 2        # 梯度累积 micro-batch 数
    warmup_steps: int = 5
    learning_rate: float = 5e-3
    min_learning_rate: float = 5e-4    # WSD 衰减到峰值 LR 的 10%
    lr_decay_start_rate: float = 0.75  # 前 75% 步恒定，之后线性衰减
    use_amp: bool = True

    # compile 默认关：架构测试换模型频繁，免去 autotune 等待
    use_compile: bool = False
    compile_mode: str = "max-autotune"      # "default"|"reduce-overhead"|"max-autotune"

    # 日志与评测节奏（均以优化器步计）
    log_interval_step: int = 1
    val_interval_step: int = 100       # 0=不验证
    gen_every_n_val: int = 4           # 每 N 次验证做一次文本生成探针，0=关
    gen_prompts: tuple = ("人工智能", "他是")
    gen_len: int = 40


class ArchTestTrainer:
    """pre_train.py PreTrainer 的架构测试版：机制一致，分析统计齐全，无存档。"""

    def __init__(self, config: ArchTestConfig):
        torch.set_float32_matmul_precision("high")  # TF32，同 PreTrainer
        self.config = config
        self._set_seed()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizers.Tokenizer.from_file(config.tokenizer_dir)
        self.vocab_size = int(len(self.tokenizer.get_vocab()))
        self.pad_id = MODEL_ARGS.pad_id

        self.run_tag = f"{config.exp_name}_{time.strftime('%y%m%d-%H%M%S')}"
        self.log_dir = os.path.join("logs", "exp", self.run_tag)
        os.makedirs(self.log_dir, exist_ok=True)
        # writer 先建：_build_model 里就要写 MoE/Config 等文本
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.train_loader, self.val_loader = self._build_dataloader()
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizers, self.schedulers = self._build_optimizer()
        self.scaler = torch.GradScaler(
            str(self.device), enabled=config.use_amp)
        # 文本生成探针：TextGenerator 需要 model.args（seq_max_len/pad_id），
        # 自定义模型不满足时降级关闭，不影响训练主流程
        self.generator = None
        if hasattr(self.model, "args"):
            self.generator = TextGenerator(
                self.model, self.tokenizer, self.device, padding_side="none")
        else:
            print("[gen] 模型无 .args，文本生成探针已关闭")
        # MoE 负载收集（无 .blocks 或没有 MoEFFN 层时是零开销的空收集器）
        self.moe_collector = MoEStatsCollector(
            n_experts=MODEL_ARGS.n_experts,
            n_experts_per_tok=MODEL_ARGS.n_experts_per_tok,
        )
        if hasattr(self._naked_model, "blocks"):
            self.moe_collector.register(self._naked_model)

        self.current_epoch = 0
        self.global_step = 0        # micro-batch 计数
        self.opt_step = 0           # 优化器步计数（调度/评测节奏用）
        self.loss_ema = None        # 训练损失 EMA（平滑曲线 + 尖峰检测）
        self.grad_norm_ema = None   # 梯度范数 EMA（loss 未爆、梯度先炸的预警）
        self.val_count = 0
        self._dump_config()
        # 注意: 别写成 print(self._build_model())——重建模型会把
        # self._naked_model 换成未参与训练的孤儿模型，梯度统计全空、权重
        # 统计全是初始化值（打印用现成对象）
        print(self._naked_model)

    # ------------------------------------------------ 构建

    def _set_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _build_model(self):
        args = MyLMArgs(**{**asdict(MODEL_ARGS),
                           "vocab_size": self.vocab_size,
                           "seq_max_len": self.config.seq_max_len})
        self.model_args = args
        model = build_model(args)
        # 分析统计/钩子都作用在裸模型上（compile/DP 包裹后属性键会变）
        self._naked_model = model
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {type(model).__name__} 参数量: {n_params/1e6:.3f}M")
        if args.use_moe:
            active = sum(
                p.numel() * (args.n_experts_per_tok / args.n_experts)
                if "mlp." in n and n.endswith(("w_gate", "w_up", "w_down"))
                else p.numel()
                for n, p in model.named_parameters())
            print(f"[model] MoE 激活参数: {active/1e6:.3f}M "
                  f"({active/n_params*100:.1f}%)")
            self.writer.add_text(
                "MoE/Config",
                f"n_experts={args.n_experts}, "
                f"n_experts_per_tok={args.n_experts_per_tok}, "
                f"latent_moe={args.latent_moe}, kappa={args.moe_capacity}", 0)
        if self.config.use_compile:
            model = torch.compile(model, mode=self.config.compile_mode)
        if torch.cuda.device_count() > 1:
            print(f"多卡训练: {torch.cuda.device_count()} 张GPU")
            model = nn.DataParallel(model)
        return model

    def _build_dataloader(self):
        dataset = PretrainTokenIDDataset(
            self.config.data_dir,
            seq_max_len=self.config.seq_max_len,
            downsample=self.config.dataset_downsample,
            padding_side=self.config.padding_side,
            shuffle_seed=self.config.seed,
        )
        val_len = int(len(dataset) * self.config.valset_rate)
        train_len = len(dataset) - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_len, val_len])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,           # 排列已由 shuffle_seed 固定（同 seed 可复现）
            drop_last=True,          # batch 形状恒定（compile 友好）
            pin_memory=True,
            num_workers=6,
            prefetch_factor=4,
            persistent_workers=False,  # 每 epoch 换排列需重 spawn worker
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,           # 固定验证集，run 之间指标可直接对比
            drop_last=True,
            pin_memory=True,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=False,
        )
        return train_loader, val_loader

    def _build_optimizer(self):
        """与 pre_train.py 一致的双优化器切分:
        2D 权重(embedding/lm_head 除外) -> Muon；其余 -> AdamW8bit。
        """
        muon_params, other_params = [], []
        for name, param in self._naked_model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 2:
                lname = name.lower()
                if "embedding" in lname or "embed" in lname \
                        or "head" in lname or "classifier" in lname \
                        or "lm_head" in lname:
                    other_params.append(param)
                elif "weight" in lname and "bias" not in lname:
                    muon_params.append(param)
                else:
                    other_params.append(param)
            else:
                other_params.append(param)
        print(f"[optim] Muon {len(muon_params)} 组 / "
              f"AdamW8bit {len(other_params)} 组")
        if not muon_params:
            raise SystemExit("没有参数进入 Muon 组：模型可能没有命名含 weight 的"
                             "2D 权重，请检查参数命名约定")

        optimizers = [
            torch.optim.Muon(
                muon_params,
                lr=self.config.learning_rate,
                adjust_lr_fn="match_rms_adamw",
                weight_decay=0.01,
            ),
            bnb.optim.adamw.AdamW8bit(
                other_params,
                lr=self.config.learning_rate,
                amsgrad=False,
                betas=(0.85, 0.999),
                eps=1e-7,
                weight_decay=0.01,
            ),
        ]
        steps_per_epoch = (
            len(self.train_loader) + self.config.batch_acceleration - 1
        ) // self.config.batch_acceleration
        total_opt_steps = self.config.epochs * steps_per_epoch + 1
        if self.config.max_steps > 0:
            total_opt_steps = min(total_opt_steps, self.config.max_steps + 1)
        schedulers = [
            WarmUpStableDecayLR(
                optimizer,
                total_steps=total_opt_steps,
                warmup_steps=self.config.warmup_steps,
                stable_steps=max(
                    0,
                    int(self.config.lr_decay_start_rate * total_opt_steps
                        - self.config.warmup_steps),
                ),
                min_lr=self.config.min_learning_rate,
                decay_mode="linear",
            )
            for optimizer in optimizers
        ]
        return optimizers, schedulers

    def _dump_config(self):
        cfg = asdict(self.config)
        cfg["model_args"] = asdict(self.model_args)
        cfg["model_class"] = type(self._naked_model).__name__
        cfg["vocab_size"] = self.vocab_size
        with open(os.path.join(self.log_dir, "config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
        self.writer.add_text(
            "config", f"```\n{json.dumps(cfg, indent=2, ensure_ascii=False)}\n```", 0)

    # ------------------------------------------------ 分析统计

    def _compute_grad_stats(self):
        """unscale 后、zero_grad 前调用。
        返回 max_abs / mean_abs / zero_ratio / per_layer_norms(1D tensor)。"""
        max_abs = 0.0
        sum_abs = 0.0
        n_zero = 0
        n_total = 0
        layer_norms = []
        for _, param in self._naked_model.named_parameters():
            if param.grad is None or param.numel() == 0:
                continue
            g_flat = param.grad.detach().float().flatten()
            layer_norms.append(g_flat.norm().item())
            max_abs = max(max_abs, g_flat.abs().max().item())
            sum_abs += g_flat.abs().sum().item()
            n_zero += (g_flat.abs() < 1e-9).sum().item()
            n_total += g_flat.numel()
        return {
            "max_abs": max_abs,
            "mean_abs": sum_abs / max(n_total, 1),
            "zero_ratio": n_zero / max(n_total, 1),
            "per_layer_norms": torch.tensor(layer_norms, dtype=torch.float32),
        }

    def _compute_weight_stats(self):
        """权重动力学: L2 总范数 / 最大绝对值 / 整体 std / 死亡权重占比 /
        逐参数张量 norm（直方图）。发散、塌缩、梯度消失的第一信号。"""
        weight_norm_sq = 0.0
        weight_max = 0.0
        all_sum = 0.0
        all_sq_sum = 0.0
        n_zero = 0
        n_total = 0
        layer_norms = []
        for _, param in self._naked_model.named_parameters():
            if not param.requires_grad or param.numel() == 0:
                continue
            w_flat = param.detach().float().flatten()
            layer_norm = w_flat.norm().item()
            layer_norms.append(layer_norm)
            weight_norm_sq += layer_norm ** 2
            w_abs = w_flat.abs()
            weight_max = max(weight_max, w_abs.max().item())
            all_sum += w_flat.sum().item()
            all_sq_sum += (w_flat * w_flat).sum().item()
            n_zero += (w_abs < 1e-2).sum().item()
            n_total += w_flat.numel()
        mean = all_sum / max(n_total, 1)
        mean_sq = all_sq_sum / max(n_total, 1)
        return {
            "weight_norm": math.sqrt(weight_norm_sq),
            "weight_max": weight_max,
            "weight_std": math.sqrt(max(mean_sq - mean * mean, 0.0)),
            "sparse_ratio": n_zero / max(n_total, 1),
            "per_layer_norms": torch.tensor(layer_norms, dtype=torch.float32),
        }

    def _log_moe_stats(self, moe_stats):
        """MoE 逐层 drop_rate / router_entropy 标量 + 全局 expert_load 直方图。"""
        if not moe_stats:
            return
        n = len(moe_stats)
        drop = torch.zeros(n)
        ent = torch.zeros(n)
        loads = []
        for i, (layer_idx, s) in enumerate(moe_stats):
            drop[i] = s["drop_rate"]
            ent[i] = s["router_entropy"]
            loads.append(s["expert_load"])
            self.writer.add_scalar(
                f"MoE/drop_rate_L{layer_idx}", s["drop_rate"], self.opt_step)
            self.writer.add_scalar(
                f"MoE/router_entropy_L{layer_idx}", s["router_entropy"],
                self.opt_step)
        self.writer.add_scalar(
            "MoE/drop_rate_mean", drop.mean().item(), self.opt_step)
        self.writer.add_scalar(
            "MoE/router_entropy_mean", ent.mean().item(), self.opt_step)
        self._add_hist("MoE/expert_load", torch.cat(loads))

    def _add_hist(self, tag, values):
        """直方图防崩护栏: 空张量/全 NaN 时跳过并告警（一次评测工具的
        日志写入不该毁掉整次 run）；NaN 信号由 scalar 通道承担。"""
        values = torch.as_tensor(values, dtype=torch.float32).flatten()
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            print(f"[warn] {tag} 直方图无有效数据，跳过 (opt_step={self.opt_step})")
            return
        self.writer.add_histogram(tag, values, self.opt_step)

    # ------------------------------------------------ 训练

    def _train_step(self, inputs, targets, mask):
        """单 micro-batch 前向 -> (num, den)，token 加权 loss，同 pre_train.py。"""
        self.model.train()
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        with torch.autocast(str(self.device), enabled=self.config.use_amp,
                            dtype=torch.bfloat16):
            output = self.model(inputs, padding_mask=(inputs != self.pad_id))
            loss_per_token = self.criterion(
                output.view(-1, self.vocab_size), targets.view(-1))
            mask_f = mask.view(-1)
            num = (loss_per_token * mask_f).sum()
            den = mask_f.sum()
        return num, den

    def _optimizer_step(self, capture_grad_stats: bool):
        """边界处: unscale + clip(+梯度统计) + 双 step + zero_grad(保留缓冲)
        + 双调度 step。返回 (grad_norm, grad_stats)。"""
        for optimizer in self.optimizers:
            self.scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1.0).item()   # clip 前范数
        grad_stats = self._compute_grad_stats() if capture_grad_stats else None
        for optimizer in self.optimizers:
            self.scaler.step(optimizer)
        self.scaler.update()
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=False)
        for scheduler in self.schedulers:
            scheduler.step()
        return grad_norm, grad_stats

    @torch.inference_mode()
    def validate(self):
        """评测: token 加权 loss / ppl / entropy / top1 / top5。
        GPU 张量聚合、循环外一次同步（大验证集下明显提速）。"""
        self.model.eval()
        loss_acc = torch.zeros((), device=self.device)
        entropy_acc = torch.zeros((), device=self.device)
        top1_acc = torch.zeros((), device=self.device, dtype=torch.long)
        top5_acc = torch.zeros((), device=self.device, dtype=torch.long)
        total_tokens = torch.zeros((), device=self.device, dtype=torch.long)

        with torch.autocast(str(self.device), enabled=self.config.use_amp,
                            dtype=torch.bfloat16):
            for val_inputs, val_targets, val_mask in self.val_loader:
                val_inputs = val_inputs.to(self.device, non_blocking=True)
                val_targets = val_targets.to(self.device, non_blocking=True)
                val_mask = val_mask.to(self.device, non_blocking=True)
                out = self.model(
                    val_inputs, padding_mask=(val_inputs != self.pad_id))
                logits = out.view(-1, self.vocab_size)
                targets_flat = val_targets.view(-1)
                mask_flat = val_mask.view(-1)
                loss = self.criterion(logits, targets_flat)
                loss_acc += (loss * mask_flat).sum().float()

                probs = torch.softmax(logits.float(), dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                entropy_acc += (entropy * mask_flat).sum()
                valid = mask_flat > 0
                top1_acc += ((probs.argmax(dim=-1) == targets_flat)
                             & valid).sum()
                if self.vocab_size >= 5:
                    _, topk = probs.topk(5, dim=-1)
                    top5_acc += ((topk == targets_flat.unsqueeze(-1))
                                 .any(dim=-1) & valid).sum()
                else:
                    top5_acc = top1_acc
                total_tokens += valid.sum()

        if len(self.val_loader) == 0:
            print("警告: 验证集为空（0 个 batch），跳过本轮验证")
            return None
        total = max(total_tokens.item(), 1)
        avg_loss = loss_acc.item() / total
        return {
            "loss": avg_loss,
            "ppl": math.exp(min(avg_loss, 20)),   # 防溢出，训练塌了看 loss 即可
            "entropy": entropy_acc.item() / total,
            "top1": top1_acc.item() / total,
            "top5": top5_acc.item() / total,
        }

    def generate_probe(self):
        """文本生成探针： loss 曲线看不出「模型是不是在学人话」，生成可以。
        自定义模型不满足生成接口时静默跳过，不影响评测主流程。"""
        if self.config.gen_every_n_val <= 0 or self.generator is None:
            return
        if self.val_count % self.config.gen_every_n_val != 0:
            return
        self.model.eval()
        lines = []
        for prompt in self.config.gen_prompts:
            try:
                ans = self.generator.generate(
                    start_token=prompt, gen_seq_len=self.config.gen_len,
                    print_out=False, temperature=0.7, top_k=20,
                    frequency_penalty=1.5)
                text = "".join(ans)[len(prompt):]
            except Exception as exc:
                print(f"[gen] 生成探针失败（已关闭）: {exc}")
                self.config.gen_every_n_val = 0
                return
            lines.append(f"{prompt}-> {text}")
        joined = "\n\n".join(lines)
        print(f"\n[gen] opt_step={self.opt_step}\n{joined}")
        self.writer.add_text("GeneratedText", joined, self.opt_step)

    def train(self):
        gc.collect()
        print(f"~~~ 架构测试: {self.config.exp_name} -> {self.log_dir} ~~~")
        # 预分配 .grad 缓冲，保 CUDAGraph 稳定（同 pre_train.py）
        for p in self.model.parameters():
            if p.requires_grad and p.grad is None:
                p.grad = torch.zeros_like(p)

        stop = False
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            # 每 epoch 换排列（seed+epoch），穿透 random_split 的 Subset
            base_ds = self.train_loader.dataset
            while isinstance(base_ds, torch.utils.data.Subset):
                base_ds = base_ds.dataset
            base_ds.set_permute_seed(self.config.seed + epoch)

            bar = tqdm(self.train_loader, unit="batch",
                       desc=f"epoch {epoch}")
            train_loss_sum, train_n = 0.0, 0
            acc_num, acc_den = 0.0, 0
            t_window = time.perf_counter()
            tok_window = 0

            for i, (inputs, targets, mask) in enumerate(bar):
                # 提前判断本 micro-batch 是否处于「验证边界」：
                # MoE 统计按需开启（仅该 micro-batch 的 forward 收集，零常态开销）
                is_boundary = ((i + 1) % self.config.batch_acceleration == 0) \
                    or (i + 1 == len(self.train_loader))
                need_eval = is_boundary and self.config.val_interval_step \
                    and (self.opt_step + 1) % self.config.val_interval_step == 0
                self.moe_collector.set_enabled(need_eval)

                num, den = self._train_step(inputs, targets, mask)
                self.moe_collector.set_enabled(False)
                den_int = int(den.item())
                acc_num += float(num.item())
                acc_den += den_int
                tok_window += den_int
                self.scaler.scale(num).backward()
                self.global_step += 1

                if not is_boundary:
                    loss = float(num.item()) / max(den_int, 1)
                    bar.postfix = f"loss {loss:.3f}"
                    continue

                if acc_den > 0:    # 全局 token 加权：边界统一除 Σden
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.div_(acc_den)
                grad_norm, grad_stats = self._optimizer_step(need_eval)
                loss = acc_num / max(acc_den, 1)
                acc_num, acc_den = 0.0, 0
                train_loss_sum += loss
                train_n += 1
                self.opt_step += 1

                # EMA 平滑（0.99 窗口）+ 尖峰比：发散/塌缩/loss spike 一眼可见
                # loss 与 grad norm 各一条：梯度尖峰通常先于 loss 尖峰出现
                if self.loss_ema is None:
                    self.loss_ema = loss
                self.loss_ema = 0.99 * self.loss_ema + 0.01 * loss
                if self.grad_norm_ema is None:
                    self.grad_norm_ema = grad_norm
                self.grad_norm_ema = (0.99 * self.grad_norm_ema
                                      + 0.01 * grad_norm)
                # 梯度范数每步都记（单标量近零开销），避免被 log_interval 采样漏掉尖峰
                self.writer.add_scalar(
                    "GradNorm/raw", grad_norm, self.opt_step)
                self.writer.add_scalar(
                    "GradNorm/ema", self.grad_norm_ema, self.opt_step)
                self.writer.add_scalar(
                    "Stability/gradnorm_vs_ema",
                    grad_norm / max(self.grad_norm_ema, 1e-8), self.opt_step)

                if self.opt_step % self.config.log_interval_step == 0:
                    dt = time.perf_counter() - t_window
                    lr = float(self.schedulers[0].get_last_lr()[0])
                    self.writer.add_scalar("Loss/train", loss, self.opt_step)
                    self.writer.add_scalar(
                        "Loss/train_ema", self.loss_ema, self.opt_step)
                    self.writer.add_scalar(
                        "Stability/loss_vs_ema",
                        loss / max(self.loss_ema, 1e-8), self.opt_step)
                    self.writer.add_scalar("LR", lr, self.opt_step)
                    for opt_name, opt in zip(
                            ["Muon", "AdamW"], self.optimizers):
                        self.writer.add_scalar(
                            f"LearningRate/{opt_name}",
                            float(opt.param_groups[0]["lr"]), self.opt_step)
                    if dt > 0:  # tok/s = 窗口内有效 token / 墙钟；step_ms 平均
                        self.writer.add_scalar(
                            "Perf/tok_s", tok_window / dt, self.opt_step)
                        self.writer.add_scalar(
                            "Perf/step_ms",
                            dt / self.config.log_interval_step * 1000,
                            self.opt_step)
                    t_window = time.perf_counter()
                    tok_window = 0
                    bar.postfix = (f"loss {loss:.3f} lr {lr:.1e}")

                if need_eval:
                    moe_stats = self.moe_collector.stats()
                    if moe_stats:
                        self._log_moe_stats(moe_stats)
                    val_metrics = self.validate()
                    self.val_count += 1
                    ws = self._compute_weight_stats()
                    self.writer.add_scalar(
                        "Weight/norm", ws["weight_norm"], self.opt_step)
                    self.writer.add_scalar(
                        "Weight/max_abs", ws["weight_max"], self.opt_step)
                    self.writer.add_scalar(
                        "Weight/std", ws["weight_std"], self.opt_step)
                    self.writer.add_scalar(
                        "Weight/sparse_ratio", ws["sparse_ratio"],
                        self.opt_step)
                    self._add_hist(
                        "WeightNorm/per_param", ws["per_layer_norms"])
                    if grad_stats is not None:   # 边界 unscale 后已捕获
                        self.writer.add_scalar(
                            "Grad/max_abs", grad_stats["max_abs"],
                            self.opt_step)
                        self.writer.add_scalar(
                            "Grad/mean_abs", grad_stats["mean_abs"],
                            self.opt_step)
                        self.writer.add_scalar(
                            "Grad/zero_ratio", grad_stats["zero_ratio"],
                            self.opt_step)
                        self._add_hist(
                            "GradNorm/per_param",
                            grad_stats["per_layer_norms"])
                        self.writer.add_scalar(
                            "Stability/update_ratio",
                            grad_norm / (ws["weight_norm"] + 1e-8),
                            self.opt_step)
                    if val_metrics is not None:
                        self.writer.add_scalar(
                            "Loss/val", val_metrics["loss"], self.opt_step)
                        self.writer.add_scalar(
                            "Val/ppl", val_metrics["ppl"], self.opt_step)
                        self.writer.add_scalar(
                            "Val/entropy", val_metrics["entropy"],
                            self.opt_step)
                        self.writer.add_scalar(
                            "Val/top1_acc", val_metrics["top1"],
                            self.opt_step)
                        self.writer.add_scalar(
                            "Val/top5_acc", val_metrics["top5"], self.opt_step)
                        print(f"\n[val] opt_step={self.opt_step} "
                              f"val_loss={val_metrics['loss']:.4f} "
                              f"ppl={val_metrics['ppl']:.2f} "
                              f"top1={val_metrics['top1']:.4f} "
                              f"top5={val_metrics['top5']:.4f}")
                    self.generate_probe()

                if self.config.max_steps and self.opt_step >= self.config.max_steps:
                    print(f"[stop] 达到 max_steps={self.config.max_steps}")
                    stop = True
                    break

            bar.close()
            if train_n:
                print(f"[epoch {epoch}] avg_train_loss: "
                      f"{train_loss_sum/train_n:.4f}")
            if stop:
                break

        self.moe_collector.close()
        self.writer.close()
        print("完成。tensorboard --logdir logs/exp")


def _apply_cli_overrides(config: ArchTestConfig) -> ArchTestConfig:
    """调试便捷: 可选追加一个 JSON（内联或文件路径）覆盖配置字段。

    PowerShell 下内联双引号易被吞，推荐写成 override.json 传路径:
        uv run python arch_test.py override.json
    """
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        raw = open(arg, encoding="utf-8").read() if os.path.exists(arg) \
            else arg
        overrides = json.loads(raw)
        for k, v in overrides.items():
            if not hasattr(config, k):
                raise SystemExit(f"未知配置字段: {k}")
            setattr(config, k, v)
    return config


if __name__ == "__main__":
    config = _apply_cli_overrides(ArchTestConfig())
    trainer = ArchTestTrainer(config)
    trainer.train()
