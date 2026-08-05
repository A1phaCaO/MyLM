from models import MyLMArgs, MyLM, exclude_moe_from_compile
from dataset import TextDatasetV4, RuntimeTextDatasetV4, PretrainTextDataset, PretrainTokenIDDataset
from utils import (
    model_structure,
    TextGenerator,
    WarmUpCosineLR,
    WarmUpStableDecayLR,
    DebugTimer,
    MoEStatsCollector,
)
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import time
import math
import json
import gc
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import tokenizers
import matplotlib.pyplot as plt
import numpy as np
import bitsandbytes as bnb
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch
import os
# 设置环境变量以解决OpenMP库重复初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# ---------------------------------------------------#
#   工具组件
# ---------------------------------------------------#

t = DebugTimer()

# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.get_device_capability(0))
# print(torch.cuda.get_arch_list())


@dataclass
class TrainingConfig:
    """训练配置参数"""

    # 数据配置
    data_dir: str = r"medium_data256v2.npy"
    tokenizer_dir: str = r"bbpe_tokenizer_7k_260723_xl.json"
    model_save_dir: str = r"model\model_xl.pth"
    ckpt_save_dir: str = r"ckpt\ckpt.pth"
    config_save_dir: str = r"model\config_xl.json"
    log_dir: str = r"logs/" + time.strftime("%Y%m%d-%H%M%S")
    # log_dir: str = r"logs/20260102-143753"
    padding_side = "right"

    # 训练参数
    seed: int = 37
    epochs: int = 1
    batch_size: int = 64
    batch_acceleration: int = 3
    dataset_downsample: int = 1
    valset_rate: float = 0.0018
    val_interval_step: int = 1000
    seq_max_len = 256   # 对齐 v3 存储长度 193 (=192+1)，loader 零 pad
    use_compile: bool = False
    # "max-autotune" or "default" or "reduce-overhead"
    compile_mode: str = "max-autotune"
    # compile 时是否将 MoE 层排除在外（eager 执行）：
    # MoE 专家循环含 CPU 同步与数据依赖循环，被 compile 追踪反而更慢
    exclude_moe_from_compile: bool = True

    # 优化参数
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4  # WSD LRS衰减到1%
    lr_decay_start_rate: int = 0.8  # 最后衰减
    warmup_steps: int = 150
    use_amp: bool = True

    model_args = MyLMArgs(
        d_model=512,
        latent_moe=True,
        d_latent=256,
        d_inner=int(((256 * (8 / 3)) // 64) * 64),
        d_head=128,
        n_heads=None,
        n_layers=6,
        vocab_size=None,
        seq_max_len=seq_max_len,
        use_moe=True,
        n_experts=6,
        n_experts_per_tok=2,
        d_conv=None,
        conv_bias=None,
        ffn_bias=False,
        attn_bias=True,
        dropout=0.05,
    )

    # 新增参数：checkpoint保存间隔步数
    ckpt_interval_step: int = 1000
    # 保留最近N个checkpoint，超过自动清理（0表示不清理）
    max_ckpts_to_keep: int = 3
    # 新增参数：断点续训的checkpoint路径
    # resume_from: Optional[str] = r"ckpt\ckpt_epoch_0_step_6000.pth"
    resume_from: Optional[str] = None

class PreTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._set_seed()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizers.Tokenizer.from_file(config.tokenizer_dir)
        self.config.model_args.vocab_size = int(
            len(self.tokenizer.get_vocab()))
        self.train_loader, self.val_loader = self._build_dataloader()
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizers, self.schedulers = self._build_optimizer()
        self.scaler = torch.GradScaler(self.device, enabled=config.use_amp)
        self.generator = TextGenerator(
            self.model, self.tokenizer, self.device, padding_side="none"
        )

        # 用于扩展的属性

        self.current_epoch = 0
        self.global_step = 0
        self.start_epoch = 0
        self.current_step = 0
        self.start_step = 0
        self.train_loss_log = []  # 将改为存储(step, loss)格式
        self.val_loss_log = []
        self.lr_log = []
        self.ckpt_paths = []  # 跟踪已保存的checkpoint路径，用于自动清理
        self.writer = None  # TensorBoard SummaryWriter，在 train() 中赋值

        # 如果指定了resume_from路径，加载checkpoint
        if config.resume_from is not None:
            self.load_checkpoint(config.resume_from)

    def _set_seed(self):
        """设置随机种子"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _build_model(self):
        """构建模型"""
        model = MyLM(self.config.model_args)
        if self.config.use_compile:
            if self.config.exclude_moe_from_compile:
                n_excluded = exclude_moe_from_compile(model)
                if n_excluded:
                    print(
                        f"[compile] 已将 {n_excluded} 个 MoE 层排除在 torch.compile 之外（eager 执行）"
                    )
            model = torch.compile(
                model, mode=self.config.compile_mode, backend="eager"
            )
        return model

    def _build_dataloader(self):
        """构建数据加载器"""
        dataset = PretrainTokenIDDataset(
            self.config.data_dir,
            seq_max_len=self.config.seq_max_len,
            downsample=self.config.dataset_downsample,
            padding_side=self.config.padding_side,
            # dtype 默认 uint16，与 generate_dataset_v3 一致；若 v3 改 int32 这里也改
        )
        val_dataset_len = int(len(dataset) * self.config.valset_rate)
        train_dataset_len = len(dataset) - val_dataset_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_dataset_len, val_dataset_len]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=5,
            prefetch_factor=3,
            persistent_workers=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

        return train_loader, val_loader

    def _build_optimizer(self):
        """构建优化器"""
        muon_params = []
        other_params = []

        # 遍历所有命名参数
        for name, param in self.model.named_parameters():
            # 跳过不需要优化的参数
            if not param.requires_grad:
                continue

            # 根据参数维度和层类型判断
            if len(param.shape) == 2:
                # 进一步过滤：排除 Embedding 层和 LM Head 等
                if "embedding" in name.lower() or "embed" in name.lower():
                    other_params.append(param)
                elif (
                    "head" in name.lower()
                    or "classifier" in name.lower()
                    or "lm_head" in name.lower()
                ):
                    other_params.append(param)
                else:
                    # 检查是否为线性层权重（通常 bias 是 1D，权重是 2D）
                    if "weight" in name and "bias" not in name:
                        muon_params.append(param)
                    else:
                        other_params.append(param)
                # muon_params.append(param)
            else:
                # 1D、3D 及更高维参数都不适合 Muon
                other_params.append(param)

        # Muon With Aux AdamW
        optimizers = [
            torch.optim.Muon(
                muon_params,
                lr=self.config.learning_rate,
                adjust_lr_fn="match_rms_adamw",
                weight_decay=0.008,
            ),
            bnb.optim.adamw.AdamW8bit(
                other_params,
                lr=self.config.learning_rate,
                amsgrad=False,
                betas=(0.85, 0.999),
                eps=1e-6,
                weight_decay=0.008,
            ),
        ]

        # Full AdamW
        # optimizers = [
        # torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.config.learning_rate,
        #     amsgrad=False,
        # )
        # ]

        total_steps = (
            self.config.epochs
            * (len(self.train_loader) // self.config.batch_acceleration + 1)
        ) + 1

        # WarmUpCosineLR
        # schedulers = [
        #     WarmUpCosineLR(
        #         optimizer,
        #         total_steps=(
        #     self.config.epochs
        #     * (len(self.train_loader) // self.config.batch_acceleration + 1)
        # ) + 1,
        #         warmup_steps=self.config.warmup_steps,
        #         min_lr=self.config.min_learning_rate,
        #     ) for optimizer in optimizers]

        # WSD学习率调度器
        schedulers = [
            WarmUpStableDecayLR(
                optimizer,
                total_steps=total_steps,
                warmup_steps=self.config.warmup_steps,
                stable_steps=int(
                    self.config.lr_decay_start_rate * total_steps
                    - self.config.warmup_steps
                ),
                min_lr=self.config.min_learning_rate,
                decay_mode="linear",
            )
            for optimizer in optimizers
        ]

        return optimizers, schedulers

    def save_checkpoint(self, path: str, is_final: bool = False):
        """保存checkpoint
        - 完整训练状态（模型+优化器+调度器+RNG）保存到 path，用于断点续训
        - is_final=True 时额外保存纯模型权重到 model_save_dir
        - 非 final 保存会自动清理旧 checkpoint，仅保留最近 max_ckpts_to_keep 个
        """
        # 检测是否是DataParallel模式
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        # 为多个优化器分别保存状态
        optimizer_states = []
        scheduler_states = []
        for opt, sched in zip(self.optimizers, self.schedulers):
            optimizer_states.append(opt.state_dict())
            scheduler_states.append(sched.state_dict())

        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "current_step": self.current_step,
            "model_state_dict": model_state_dict,
            "optimizer_states": optimizer_states,
            "scheduler_states": scheduler_states,
            "train_loss": self.train_loss_log[-1] if self.train_loss_log else None,
            "rng_states": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "random": random.getstate(),
                "numpy": np.random.get_state(),
            },
        }
        torch.save(state, path)
        print(f"[ckpt] 保存checkpoint: {path}")

        if not is_final:
            # 跟踪并自动清理旧 checkpoint
            self.ckpt_paths.append(path)
            keep = self.config.max_ckpts_to_keep
            if keep > 0:
                while len(self.ckpt_paths) > keep:
                    oldest = self.ckpt_paths.pop(0)
                    if os.path.exists(oldest):
                        os.remove(oldest)
                        print(f"[ckpt] 已清理旧checkpoint: {oldest}")
        else:
            # final: 额外保存纯模型权重
            torch.save(model_state_dict, self.config.model_save_dir)
            print(f"[ckpt] 保存模型权重: {self.config.model_save_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # 恢复模型状态
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # 加载多个调度器的状态（按照PyTorch文档建议，在优化器之前加载）
        if "scheduler_states" in checkpoint and checkpoint["scheduler_states"]:
            for i, sched_state in enumerate(checkpoint["scheduler_states"]):
                if i < len(self.schedulers):
                    self.schedulers[i].load_state_dict(sched_state)
        else:
            # 兼容旧版本checkpoint
            self.schedulers.load_state_dict(checkpoint["scheduler_state_dict"])

        # 加载多个优化器的状态
        if "optimizer_states" in checkpoint and checkpoint["optimizer_states"]:
            for i, opt_state in enumerate(checkpoint["optimizer_states"]):
                if i < len(self.optimizers):
                    self.optimizers[i].load_state_dict(opt_state)
        else:
            # 兼容旧版本checkpoint
            self.optimizers.load_state_dict(checkpoint["optimizer_state_dict"])

        # 恢复训练状态
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.start_epoch = checkpoint["epoch"]
        self.start_step = checkpoint["current_step"]

        # 恢复随机状态（防止数据shuffle混乱）
        rng_states = checkpoint["rng_states"]
        torch.set_rng_state(rng_states["torch"])
        if rng_states["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_states["cuda"])
        random.setstate(rng_states["random"])
        np.random.set_state(rng_states["numpy"])

    def _train_step(self, inputs, targets, mask, capture_grad_stats=False):
        """单步训练（含梯度累加），返回 (loss, grad_norm, grad_stats)
        - grad_norm: clip 前的梯度总范数，仅在累积完成 step 时计算，否则为 None
        - grad_stats: dict {max_abs, mean_abs, zero_ratio, per_layer_norms(tensor)}，仅在
                      capture_grad_stats=True 且 grad_norm 已计算时返回（此时梯度已 unscale），否则为 None
        """
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        mask = mask.to(self.device)

        with torch.autocast(str(self.device), enabled=self.config.use_amp, dtype=torch.bfloat16):
            output = self.model(inputs)
            # 计算交叉熵损失（不使用ignore_index，因为我们手动应用mask）
            loss = self.criterion(
                output.view(-1, self.config.model_args.vocab_size), targets.view(-1)
            )
            # 应用mask：将mask展平并与损失相乘
            loss = (loss * mask.view(-1)).sum() / mask.sum()

        loss = loss / self.config.batch_acceleration

        self.scaler.scale(loss).backward()

        grad_norm = None
        grad_stats = None
        is_step_boundary = (
            (self.current_step + 1) % self.config.batch_acceleration == 0
        ) or (self.current_step + 1 == len(self.train_loader))
        if is_step_boundary:
            # 对每个优化器进行梯度缩放和更新
            for optimizer in self.optimizers:
                self.scaler.unscale_(optimizer)
            # clip_grad_norm_ 返回 clip 前的梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            ).item()

            # 捕获梯度统计（在 unscale 后、clip 前；此时为原始梯度）
            if capture_grad_stats:
                grad_stats = self._compute_grad_stats()

            # 对每个优化器单独执行step
            for optimizer in self.optimizers:
                self.scaler.step(optimizer)
            self.scaler.update()
            # 对每个优化器单独清零梯度
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            # 对每个调度器单独执行step
            for scheduler in self.schedulers:
                scheduler.step()

        return loss.item() * self.config.batch_acceleration, grad_norm, grad_stats

    def _compute_grad_stats(self):
        """计算当前梯度的统计量（需在 unscale 后、zero_grad 前调用）
        返回 dict: max_abs, mean_abs, zero_ratio, per_layer_norms(1D tensor)
        per_layer_norms: 每层梯度的 L2 范数，按 named_parameters() 顺序排列
        """
        max_abs = 0.0
        sum_abs = 0.0
        n_zero = 0
        n_total = 0
        layer_norms = []
        for name, param in self.model.named_parameters():
            if param.grad is None or param.numel() == 0:
                continue
            g = param.grad.detach().float()
            g_flat = g.flatten()
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

    def _log_moe_stats(self, writer, moe_stats, step):
        """将MoE负载均衡统计写入TensorBoard
        逐层标量合并为直方图（每步一帧、横轴为层），另保留跨层平均值曲线
        """
        n = len(moe_stats)
        metrics = {
            "aux_loss": torch.zeros(n),
            "router_entropy": torch.zeros(n),
            "top1_conf": torch.zeros(n),
            "balance_ratio": torch.zeros(n),
        }
        load_all = []
        for i, (layer_idx, s) in enumerate(moe_stats):
            metrics["aux_loss"][i] = s["aux_loss"]
            metrics["router_entropy"][i] = s["router_entropy"]
            metrics["top1_conf"][i] = s["top1_conf"]
            metrics["balance_ratio"][i] = s["balance_ratio"]
            load_all.append(s["expert_load"])
        # 逐层标量 → 直方图
        for name, vals in metrics.items():
            writer.add_histogram(f"MoE/{name}", vals, step)
            writer.add_scalar(f"MoE/avg_{name}", vals.mean().item(), step)
        # 所有层所有专家的负载占比合并为一张直方图
        writer.add_histogram("MoE/expert_load", torch.cat(load_all), step)

    def log(self):
        total_params = model_structure(self.model)
        print(f"本次训练参数：")
        print(f"词数: {self.config.model_args.vocab_size}")
        val_dataset_len, train_dataset_len = len(self.val_loader.dataset), len(
            self.train_loader.dataset
        )
        print(f"上下文长度：{self.config.model_args.seq_max_len}")
        print(f"数据集数量：{val_dataset_len+train_dataset_len}")
        print(f"训练集数量：{train_dataset_len}")
        print(f"测试集数量：{val_dataset_len}")
        nums_token = self.config.model_args.seq_max_len * train_dataset_len
        print(f"Token数约：{nums_token/1e6:.3f}M")
        print(f"模型参数：{total_params/1e6:.3f}M")
        print(
            f"计算量：{(nums_token * total_params * 6)/1e12:.2f}TFLOPs * {self.config.epochs} = {(nums_token * total_params * 6 * self.config.epochs)/1e12:.2f}TFLOPs"
        )

    def train(self):
        """训练主循环"""
        gc.collect()
        print("~~~训练咯~~~")

        # 多卡训练
        if torch.cuda.device_count() > 1:
            print(f"多卡训练: {torch.cuda.device_count()} 张GPU")
            self.model = nn.DataParallel(self.model)

        # MoE 负载均衡统计收集（forward hook 方式，对模型代码零侵入）
        self.moe_collector = None
        if self.config.model_args.use_moe:
            self.moe_collector = MoEStatsCollector(
                n_experts=self.config.model_args.n_experts,
                n_experts_per_tok=self.config.model_args.n_experts_per_tok,
            ).register(
                self.model.module
                if isinstance(self.model, nn.DataParallel)
                else self.model
            )

        for epoch in range(self.config.epochs):
            bar = tqdm(self.train_loader, unit="step")
            # 跳过已训练的epoch
            if epoch < self.start_epoch:
                print(f"跳过已训练的epoch: {epoch}")
                continue
            elif epoch == self.start_epoch:
                bar.update(self.start_step)

            self.current_epoch = epoch
            train_loss_sum = 0
            last_val_step = -1  # 记录上次验证的 global_step，避免 epoch 末尾重复验证
            last_val_metrics = None  # 缓存上次 val 指标，epoch 末复用
            writer = SummaryWriter(log_dir=self.config.log_dir)
            self.writer = writer  # 供 _train_step 写入直方图
            if self.config.model_args.use_moe:
                writer.add_text(
                    "MoE/Config",
                    f"n_experts={self.config.model_args.n_experts}, "
                    f"n_experts_per_tok={self.config.model_args.n_experts_per_tok}, "
                    f"latent_moe={self.config.model_args.latent_moe}",
                    0,
                )

            for i, (train_inputs, train_targets, train_mask) in enumerate(
                self.train_loader
            ):
                # 跳过已训练的step
                if i <= self.start_step and epoch == self.start_epoch:
                    continue
                self.current_step = i
                # 提前计算 need_val：基于 i 和 step 后的 global_step（即 global_step+1）
                need_val = (i % self.config.val_interval_step == 0) or (
                    (self.global_step + 1) % self.config.ckpt_interval_step == 0
                )
                # 仅在需要记录时开启MoE统计，避免每步额外开销
                if self.moe_collector is not None:
                    self.moe_collector.set_enabled(need_val)
                loss, grad_norm, grad_stats = self._train_step(
                    train_inputs, train_targets, train_mask,
                    capture_grad_stats=need_val,
                )
                if self.moe_collector is not None:
                    self.moe_collector.set_enabled(False)

                train_loss_sum += loss
                self.train_loss_log.append((self.global_step, loss))
                self.lr_log.append(
                    (self.global_step, float(
                        self.schedulers[0].get_last_lr()[0]))
                )
                self.global_step += 1

                # 多优化器分组学习率
                for opt_name, opt in zip(["Muon", "AdamW"], self.optimizers):
                    writer.add_scalar(
                        f"LearningRate/{opt_name}",
                        float(opt.param_groups[0]["lr"]),
                        self.global_step,
                    )
                writer.add_scalar("Loss/train", loss, self.global_step)
                if grad_norm is not None:
                    writer.add_scalar(
                        "GradNorm/raw", grad_norm, self.global_step)
                # 梯度统计（A 类：动力学 + B 类：分层 grad norm 分布）
                if grad_stats is not None:
                    writer.add_scalar(
                        "Grad/max_abs", grad_stats["max_abs"], self.global_step
                    )
                    writer.add_scalar(
                        "Grad/mean_abs", grad_stats["mean_abs"], self.global_step
                    )
                    writer.add_scalar(
                        "Grad/zero_ratio", grad_stats["zero_ratio"], self.global_step
                    )
                    # 分层 grad norm 分布（直方图叠加显示）
                    writer.add_histogram(
                        "GradNorm/per_layer",
                        grad_stats["per_layer_norms"],
                        self.global_step,
                    )

                if need_val:
                    # MoE 负载均衡统计（在 validate 前读取，避免被验证前向覆盖）
                    if self.moe_collector is not None:
                        moe_stats = self.moe_collector.stats()
                        if moe_stats:
                            self._log_moe_stats(
                                writer, moe_stats, self.global_step)
                    val_loss, val_ppl, val_metrics = self.validate()
                    last_val_step = self.global_step
                    last_val_metrics = val_metrics
                    self.val_loss_log.append((self.global_step, val_loss))
                    writer.add_scalar("Loss/val", val_loss, self.global_step)
                    writer.add_scalar("Val/ppl", val_ppl, self.global_step)
                    writer.add_scalar(
                        "Val/entropy", val_metrics["entropy"], self.global_step
                    )
                    writer.add_scalar(
                        "Val/top1_acc", val_metrics["top1_acc"], self.global_step
                    )
                    writer.add_scalar(
                        "Val/top5_acc", val_metrics["top5_acc"], self.global_step
                    )
                    # A 类：权重动力学
                    writer.add_scalar(
                        "Weight/norm", val_metrics["weight_norm"], self.global_step
                    )
                    writer.add_scalar(
                        "Weight/max_abs", val_metrics["weight_max"], self.global_step
                    )
                    writer.add_scalar(
                        "Weight/std", val_metrics["weight_std"], self.global_step
                    )
                    writer.add_scalar(
                        "Weight/sparse_ratio",
                        val_metrics["sparse_ratio"],
                        self.global_step,
                    )
                    # A 类：训练动力学（梯度/权重比）
                    if grad_stats is not None:
                        update_ratio = (
                            grad_norm / (val_metrics["weight_norm"] + 1e-8)
                            if grad_norm is not None
                            else 0.0
                        )
                        writer.add_scalar(
                            "Train/update_ratio", update_ratio, self.global_step
                        )
                    # B 类：分层 weight norm 分布（直方图叠加显示）
                    writer.add_histogram(
                        "WeightNorm/per_layer",
                        val_metrics["per_layer_weight_norms"],
                        self.global_step,
                    )
                    # 文本生成测试（打印到控制台 + 写入 TensorBoard）
                    test_text = self.generate_test("人工智能是")
                    writer.add_text(
                        "GeneratedText",
                        f"epoch_{epoch}_step_{i}: {test_text}",
                        self.global_step,
                    )
                    # checkpoint 保存（若是 ckpt 触发点）
                    is_ckpt_step = (
                        self.global_step % self.config.ckpt_interval_step == 0
                    )
                    if is_ckpt_step:
                        ckpt_path = f"{self.config.ckpt_save_dir.rsplit('.', 1)[0]}_epoch_{self.current_epoch}_step_{self.global_step}.pth"
                        self.save_checkpoint(ckpt_path, is_final=False)
                    print(
                        f"[step {self.global_step}] val_loss: {val_loss:.4f}, ppl: {val_ppl:.4f}, "
                        f"acc@1: {val_metrics['top1_acc']:.4f}, acc@5: {val_metrics['top5_acc']:.4f}, "
                        f"ent: {val_metrics['entropy']:.4f}"
                        + (" [ckpt]" if is_ckpt_step else "")
                    )

                # 进度显示
                bar.update(1)
                bar.postfix = f"train_loss: {loss:.2f} lr: {self.schedulers[0].get_last_lr()[0]:.2e}"

            bar.close()

            # epoch 结束：若最后一步刚验证过则复用，避免重复 validate
            if last_val_step == self.global_step and last_val_metrics is not None:
                val_loss = self.val_loss_log[-1][1]
                val_ppl = math.exp(val_loss)
                val_metrics = last_val_metrics
            else:
                val_loss, val_ppl, val_metrics = self.validate()
                self.val_loss_log.append((self.global_step, val_loss))
                writer.add_scalar("Loss/val", val_loss, self.global_step)
                writer.add_scalar("Val/ppl", val_ppl, self.global_step)
                writer.add_scalar(
                    "Val/entropy", val_metrics["entropy"], self.global_step
                )
                writer.add_scalar(
                    "Val/top1_acc", val_metrics["top1_acc"], self.global_step
                )
                writer.add_scalar(
                    "Val/top5_acc", val_metrics["top5_acc"], self.global_step
                )
                # A 类：权重动力学
                writer.add_scalar(
                    "Weight/norm", val_metrics["weight_norm"], self.global_step
                )
                writer.add_scalar(
                    "Weight/max_abs", val_metrics["weight_max"], self.global_step
                )
                writer.add_scalar(
                    "Weight/std", val_metrics["weight_std"], self.global_step
                )
                writer.add_scalar(
                    "Weight/sparse_ratio",
                    val_metrics["sparse_ratio"],
                    self.global_step,
                )
                # B 类：分层 weight norm 分布（直方图叠加显示）
                writer.add_histogram(
                    "WeightNorm/per_layer",
                    val_metrics["per_layer_weight_norms"],
                    self.global_step,
                )

            # epoch 级 checkpoint（is_final=True 同步保存模型权重）
            self.save_checkpoint(
                f"{self.config.ckpt_save_dir.rsplit('.', 1)[0]}_epoch_{epoch}.pth",
                is_final=True,
            )

            # epoch 末文本生成测试（打印到控制台）
            test_text = self.generate_test(gen_len=100, start="我是")
            writer.add_text(
                "GeneratedText", f"epoch_{epoch}: {test_text}", self.global_step
            )
            print(
                f"[epoch {epoch+1}/{self.config.epochs}] "
                f"avg_train_loss: {train_loss_sum/len(self.train_loader):.4f}, "
                f"val_loss: {val_loss:.4f}, ppl: {val_ppl:.4f}, "
                f"lr: {self.schedulers[0].get_last_lr()[0]:.2e}"
            )

        # 保存最终模型
        self.save_checkpoint(self.config.model_save_dir, is_final=True)

    def validate(self):
        """验证过程，返回 (avg_loss, ppl, metrics_dict)
        metrics_dict 包含: entropy, top1_acc, top5_acc,
                            weight_norm, weight_max, weight_std, sparse_ratio,
                            per_layer_weight_norms (1D tensor)
        """
        self.model.eval()
        val_loss_sum = 0
        entropy_sum = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_tokens = 0
        vocab_size = self.config.model_args.vocab_size

        with torch.no_grad():
            for val_inputs, val_targets, val_mask in self.val_loader:
                val_inputs = val_inputs.to(self.device)
                val_targets = val_targets.to(self.device)
                val_mask = val_mask.to(self.device)
                with torch.autocast(str(self.device), enabled=self.config.use_amp):
                    val_output = self.model(val_inputs)
                    logits = val_output.view(-1, vocab_size)
                    targets_flat = val_targets.view(-1)
                    mask_flat = val_mask.view(-1)
                    loss = self.criterion(logits, targets_flat)
                    loss = (loss * mask_flat).sum() / mask_flat.sum()
                val_loss_sum += loss.item()

                # 计算 entropy / top-k accuracy（用 float32 精度，仅对有效 token）
                probs = torch.softmax(logits.float(), dim=-1)
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum(dim=-1)  # (B*L,)
                valid_mask = mask_flat > 0
                entropy_sum += (entropy * mask_flat).sum().item()

                pred_top1 = probs.argmax(dim=-1)
                correct_top1 += (
                    (pred_top1 == targets_flat) & valid_mask
                ).sum().item()

                if vocab_size >= 5:
                    _, pred_top5 = probs.topk(5, dim=-1)
                    correct_top5 += (
                        (
                            pred_top5 == targets_flat.unsqueeze(-1)
                        ).any(dim=-1) & valid_mask
                    ).sum().item()
                else:
                    correct_top5 = correct_top1  # vocab 不足时退化为 top1

                total_tokens += valid_mask.sum().item()

        n = len(self.val_loader)
        avg_loss = val_loss_sum / n
        ppl = math.exp(avg_loss)
        avg_entropy = entropy_sum / max(total_tokens, 1)
        top1_acc = correct_top1 / max(total_tokens, 1)
        top5_acc = correct_top5 / max(total_tokens, 1)

        # 参数统计量（仅算一次，模型参数在 val 期间不变）
        weight_stats = self._compute_weight_stats()

        return avg_loss, ppl, {
            "entropy": avg_entropy,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            **weight_stats,  # weight_norm, weight_max, weight_std, sparse_ratio, per_layer_weight_norms
        }

    def _compute_weight_stats(self):
        """计算模型权重的统计量，返回 dict
        - weight_norm: 整体 L2 范数
        - weight_max: 最大绝对值
        - weight_std: 整体标准差（基于 mean 与二阶矩，正确公式）
        - sparse_ratio: |w|<1e-2 的比例（死亡神经元检测）
        - per_layer_weight_norms: 1D tensor，每层 L2 范数，按 named_parameters() 顺序
        """
        weight_norm_sq = 0.0
        weight_max = 0.0
        all_sum = 0.0       # 累加 x（用于求 mean）
        all_sq_sum = 0.0    # 累加 x^2（用于求 E[x^2]）
        n_zero = 0
        n_total = 0
        layer_norms = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.numel() == 0:
                continue
            w = param.detach().float()
            w_flat = w.flatten()
            # 分层 L2 norm
            layer_norm = w_flat.norm()
            layer_norms.append(layer_norm.item())
            # 全局统计
            weight_norm_sq += layer_norm.item() ** 2
            w_abs = w_flat.abs()
            weight_max = max(weight_max, w_abs.max().item())
            all_sum += w_flat.sum().item()
            all_sq_sum += (w_flat * w_flat).sum().item()
            n_zero += (w_abs < 1e-2).sum().item()
            n_total += w_flat.numel()
        weight_norm = math.sqrt(weight_norm_sq)
        mean = all_sum / max(n_total, 1)
        mean_sq = all_sq_sum / max(n_total, 1)
        # 方差 = E[x^2] - (E[x])^2
        weight_std = math.sqrt(max(mean_sq - mean * mean, 0.0))
        return {
            "weight_norm": weight_norm,
            "weight_max": weight_max,
            "weight_std": weight_std,
            "sparse_ratio": n_zero / max(n_total, 1),
            "per_layer_weight_norms": torch.tensor(layer_norms, dtype=torch.float32),
        }

    def generate_test(self, start: str = "我", gen_len: int = 25, verbose: bool = True):
        """文本生成测试（返回文本；verbose=True 时打印到控制台）"""
        self.model.eval()
        ans = self.generator.generate(
            start_token=start, gen_seq_len=gen_len, print_out=False
        )
        ans = ans[len(start):]  # 截掉start_token
        result = "".join(ans)
        if verbose:
            print(f"(input){start}-> {result}")
        return result

    def plot_losses(self):
        """绘制损失曲线"""
        fig, ax1 = plt.subplots(figsize=(16, 10))

        # 提取训练步骤和损失
        train_steps = [step for step, loss in self.train_loss_log]
        train_losses = [loss for step, loss in self.train_loss_log]

        # 提取验证步骤和损失
        val_steps = [step for step, loss in self.val_loss_log]
        val_losses = [loss for step, loss in self.val_loss_log]

        # 绘制左侧的训练损失和验证损失
        ax1.plot(train_steps, train_losses,
                 label="Train Loss")  # 修改为使用step作为x轴
        ax1.plot(val_steps, val_losses, "o-", label="Test Loss")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper left")

        # 创建右侧y轴
        ax2 = ax1.twinx()
        ax2.plot(
            [step for step, _ in self.lr_log],  # 使用学习率日志中的step作为x轴
            [float(value) for _, value in self.lr_log],  # 使用value作为y轴
            label="Learning Rate",
            color="c",
            linestyle="--",
        )
        ax2.set_ylabel("Learning Rate")
        ax2.tick_params(axis="y")
        ax2.legend(loc="upper right")

        plt.title("Train and Test Loss Curves with Learning Rate")
        plt.show()


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = PreTrainer(config)
    config_dict = asdict(config.model_args)
    with open(config.config_save_dir, "w") as f:
        json.dump(config_dict, f, indent=4)
    trainer.log()
    trainer.train()
    trainer.plot_losses()

    # 交互式测试
    MAX_LEN = 100
    T = 0.8
    while True:
        start = input("In>>")
        if start[:2] == "T=":
            T = float(start[2:])
            print(f"T={T}")
        else:
            print(
                f"T={T}\n"
                + "".join(
                    trainer.generator.generate(
                        start_token=start,
                        gen_seq_len=MAX_LEN,
                        temperature=T,
                        frequency_penalty=1.5,
                        print_out=False,
                    )
                )
            )
