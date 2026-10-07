import os
import subprocess
import sys
import glob
import re
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
#   matplotlib 中文字体 (图表标签使用中文)
# ---------------------------------------------------#
from matplotlib import font_manager as _fm  # noqa: E402

for _f in [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
]:
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
        plt.rcParams["font.sans-serif"] = [
            _fm.FontProperties(fname=_f).get_name(), "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False


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
    data_dir: str = r"data/large_data384v1.npy"
    tokenizer_dir: str = r"tokenizer/bbpe_tokenizer_7k_260723_xl.json"
    model_save_dir: str = r"model\model_dense_ca8conv4_261005.pth"
    ckpt_save_dir: str = r"ckpt\ckpt.pth"
    config_save_dir: str = r"model\config_dense_ca8conv4_261005.json"
    log_dir: str = r"logs/" +"dense_ca8conv4_261005_"+ time.strftime("%y%m%d-%H%M")
    # log_dir: str = r"logs\xl2_20260809-190755"
    padding_side = "right"

    # 训练参数
    seed: int = 42
    epochs: int = 1
    batch_size: int = 32
    batch_acceleration: int = 4
    dataset_downsample: int = 1
    valset_rate: float = 0.0016
    val_interval_step: int = 2000
    seq_max_len = 384   # 对齐 v3 存储长度 257 (=256+1)，loader 零 pad
    use_compile: bool = True
    # "max-autotune" or "default" or "reduce-overhead"
    compile_mode: str = "max-autotune"
    # compile 时是否将 MoE 层排除在外（eager 执行）：
    # 【已过时】旧逐专家循环才需要排除；新 FixedCap 纯 tensor 分桶可编译，
    # 实测排除 MoE 反而慢 14%（44.2ms vs 38.6ms），保持默认 False
    exclude_moe_from_compile: bool = False

    # 优化参数
    learning_rate: float = 5e-3
    min_learning_rate: float = 5e-4  # WSD LRS衰减到峰值LR的10%
    lr_decay_start_rate: int = 0.75  # 最后衰减
    warmup_steps: int = 5
    use_amp: bool = True

    model_args = MyLMArgs(
        # 终版架构：取自 arch_test 实验 mylm_dense_half_ca8_conv4_gate_mix_fixed
        # (logs/exp/..._260927-171958)，dense 小模型 + 奇数层窗口压缩注意力
        d_model=512,
        latent_moe=False,
        d_latent=256,
        d_inner=int(((512 * (8 / 3)) // 64) * 64),  # =1024
        d_head=128,
        n_heads=None,
        n_layers=8,
        vocab_size=None,
        seq_max_len=seq_max_len,
        use_moe=False,
        n_experts=8,
        n_experts_per_tok=2,
        d_conv=4,            # conv 局部分支 kernel（实验 conv4）；勿传 None——新架构 Conv1d 会崩
        compress_ratio=8,    # 窗口压缩比（实验 ca8），随 config json 入库
        ffn_bias=False,
        attn_bias=True,
        dropout=0.05,
        base_init_std=0.02
    )

    # 新增参数：checkpoint保存间隔步数
    ckpt_interval_step: int = 2000
    # checkpoint 自动清理：保留最近 ckpt_keep_recent 个，
    # 更早的每隔 ckpt_keep_stride 个保留 1 个（0 表示不清理）
    ckpt_keep_recent: int = 3
    ckpt_keep_stride: int = 3
    # 新增参数：断点续训的checkpoint路径
    # resume_from: Optional[str] = r"ckpt\ckpt_epoch_0_step_24000.pth"
    resume_from: Optional[str] = r"ckpt\ckpt_epoch_0_step_30001.pth"  # 261005 large 数据训练，17:45 进程被外部清理，从 step 30001 续训
    # 数据集固定种子 shuffle（在 PretrainTokenIDDataset 内部实现，替代 DataLoader
    # 原版 RandomSampler：续训时排列完全由 seed 决定，从已消费位置继续无重复。
    # 置 None 则退回原版 DataLoader shuffle 行为）
    dataset_shuffle_seed: Optional[int] = 42


class PreTrainer:
    def __init__(self, config: TrainingConfig):
        # 允许 float32 矩阵乘使用 TF32 tensor cores（消除 inductor 警告并提速）
        torch.set_float32_matmul_precision("high")
        self.config = config
        self._set_seed()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizers.Tokenizer.from_file(config.tokenizer_dir)
        self.config.model_args.vocab_size = int(
            len(self.tokenizer.get_vocab()))
        self.train_loader, self.val_loader = self._build_dataloader()
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
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
        # 跟踪已保存的checkpoint路径，用于自动清理。
        # 初始扫描磁盘已有文件（按时间顺序），保证恢复训练后
        # "最近 a 个"以磁盘真实文件为口径，而不是只算本次进程新保存的
        self.ckpt_paths = self._scan_existing_ckpts()
        # 曾保存过的所有 step 序号（升序，含已被清理的文件），
        # 清理时以这里的绝对序号判定"每隔 b 个保留 1 个"，避免锚点漂移
        self.ckpt_seq = [self._step_of(p) for p in self.ckpt_paths]
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
            # 默认后端 inductor: 真融合才有加速 (eager 后端只捕获图不融合, 实测无加速)
            # 注意: Windows 上需 PYTHONUTF8=1 环境变量, 否则 inductor 读模板时 GBK 解码崩溃
            model = torch.compile(
                model, mode=self.config.compile_mode
            )
        return model

    def _build_dataloader(self):
        """构建数据加载器"""
        dataset = PretrainTokenIDDataset(
            self.config.data_dir,
            seq_max_len=self.config.seq_max_len,
            downsample=self.config.dataset_downsample,
            padding_side=self.config.padding_side,
            shuffle_seed=self.config.dataset_shuffle_seed,
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
            # 已由 PretrainTokenIDDataset(shuffle_seed=...) 固定排列，
            # 关掉原版 RandomSampler：续训时按已消费 batch 位置继续，数据不重复
            shuffle=False,
            # drop_last：保证 batch 形状恒定，避免 CUDAGraph 为不齐的最后一个 batch
            # 反复录制动态形状图（torch.compile reduce-overhead）
            drop_last=True,
            pin_memory=True,
            num_workers=6,
            prefetch_factor=4,
            # persistent_workers=False：每个 epoch 重新 spawn worker，
            # 使 set_permute_seed(seed+epoch) 的新排列能传入 worker（M4）；
            # 数据集 pickle 已由 PretrainTokenIDDataset.__getstate__ 瘦身，spawn 代价 ~1-2s
            persistent_workers=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=3,
            persistent_workers=False
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

        # Full AdamW
        # optimizers = [
        # torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.config.learning_rate,
        #     amsgrad=False,
        # )
        # ]

        # 每个 epoch 的实际优化步数 = ceil(len / batch_acceleration)
        # （is_step_boundary 在 batch_acceleration 整数倍及每 epoch 最后一批时 step）
        steps_per_epoch = (
            len(self.train_loader) + self.config.batch_acceleration - 1
        ) // self.config.batch_acceleration
        total_steps = self.config.epochs * steps_per_epoch + 1

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
                stable_steps=max(
                    0,
                    int(
                        self.config.lr_decay_start_rate * total_steps
                        - self.config.warmup_steps
                    ),
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
        - 非 final 保存会自动清理旧 checkpoint：保留最近 ckpt_keep_recent 个，
          更早的每隔 ckpt_keep_stride 个保留 1 个
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
            step = self._step_of(path)
            if step >= 0 and step not in self.ckpt_seq:
                self.ckpt_seq.append(step)
            self._prune_ckpts()
        else:
            # final: 额外保存纯模型权重
            torch.save(model_state_dict, self.config.model_save_dir)
            print(f"[ckpt] 保存模型权重: {self.config.model_save_dir}")

    @staticmethod
    def _step_of(path: str) -> int:
        """从 checkpoint 文件名解析 step 序号（如 ckpt_epoch_0_step_30000.pth -> 30000）"""
        m = re.search(r"_step_(\d+)\.pth$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    def _scan_existing_ckpts(self):
        """扫描 ckpt 目录中已有的 step checkpoint，按修改时间排序返回。
        - 只识别 ckpt_*_step_*.pth（不含 epoch 末的 is_final 文件）
        - 非递归，backup 子目录不会被扫到
        """
        ckpt_dir = os.path.dirname(self.config.ckpt_save_dir) or "."
        paths = [
            p for p in glob.glob(os.path.join(ckpt_dir, "ckpt_*_step_*.pth"))
            if os.path.isfile(p)
        ]
        paths.sort(key=os.path.getmtime)
        return paths

    def _prune_ckpts(self):
        """自动清理旧checkpoint：
        - 最近 ckpt_keep_recent 个全部保留（ckpt_paths 列表尾部）
        - 更早的按 ckpt_seq 绝对保存序，每隔 ckpt_keep_stride 个保留 1 个
          （用绝对序号而非存活列表位置，保证删除后锚点不漂移）
        """
        recent = self.config.ckpt_keep_recent
        stride = self.config.ckpt_keep_stride
        if recent <= 0 or len(self.ckpt_paths) <= recent:
            return
        keep_paths = set(self.ckpt_paths[-recent:])  # 最近 a 个全保留
        for p in self.ckpt_paths[:-recent]:
            step = self._step_of(p)
            if step in self.ckpt_seq:
                idx = self.ckpt_seq.index(step)  # 绝对保存序号
                if idx % stride == 0:
                    keep_paths.add(p)
        removed = 0
        for p in self.ckpt_paths[:-recent]:
            if p not in keep_paths and os.path.exists(p):
                os.remove(p)
                removed += 1
        if removed:
            print(f"[ckpt] 已清理 {removed} 个旧checkpoint")
        # 重建存活路径列表（保持时间顺序，尾部即最近的）
        self.ckpt_paths = [p for p in self.ckpt_paths if os.path.exists(p)]

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

    def _train_step(self, inputs, targets, mask):
        """单步前向，返回 (num, den)：
        - num: 该 micro-batch 所有有效 token 的 loss 总和（带 autograd 图，由 train() 立即 backward 累加）
        - den: 该 micro-batch 有效 token 数
        梯度累加（per-micro-batch backward）+ 边界处整体除以 Σden，使跨 micro-batch 的 loss
        按「有效 token 数」加权，而非各 micro-batch 等权平均；同时保持显存只占单 micro-batch。
        """
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        mask = mask.to(self.device)

        with torch.autocast(str(self.device), enabled=self.config.use_amp, dtype=torch.bfloat16):
            # 显式 padding mask：由 token id 判定真实 token（取代「pad 行隐藏态全零」的
            # 隐式假设），左/右 padding 注意力屏蔽均正确；推理无 padding 时传 None。
            output = self.model(inputs, padding_mask=(inputs != self.config.model_args.pad_id))
            # 逐 token 交叉熵（reduction="none"），随后由 mask 加权
            loss_per_token = self.criterion(
                output.view(-1, self.config.model_args.vocab_size), targets.view(-1)
            )
            mask_f = mask.view(-1)
            num = (loss_per_token * mask_f).sum()   # 该 micro-batch 有效 token 的 loss 总和（带图）
            den = mask_f.sum()                        # 有效 token 数
        return num, den

    def _optimizer_step(self, capture_grad_stats: bool):
        """梯度累加边界：unscale + clip（可选梯度统计）+ step + update + zero_grad + 调度器 step。
        返回 (grad_norm, grad_stats)。"""
        for optimizer in self.optimizers:
            self.scaler.unscale_(optimizer)
        # clip_grad_norm_ 返回 clip 前的梯度范数
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1.0
        ).item()
        # 捕获梯度统计（在 unscale 后、clip 前；此时为原始梯度）
        grad_stats = self._compute_grad_stats() if capture_grad_stats else None
        # 对每个优化器单独执行 step
        for optimizer in self.optimizers:
            self.scaler.step(optimizer)
        self.scaler.update()
        # 对每个优化器单独清零梯度
        # 注意: set_to_none=False 保持 .grad 缓冲稳定（配合 train() 中的预分配），
        # 避免 CUDAGraph(torch.compile reduce-overhead) 梯度累积时报错
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=False)
        # 对每个调度器单独执行 step
        for scheduler in self.schedulers:
            scheduler.step()
        return grad_norm, grad_stats

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
        """将 MoE 运行时负载统计写入 TensorBoard。

        只保留最有代表性的信号:
        - MoE/drop_rate        标量:   跨层平均丢 token 率 (κ 是否够用, 应 <10%)
        - MoE/router_entropy   标量:   跨层平均归一化路由熵 (1.0=完全均匀)
        - MoE/heatmap          图像:   (step × 层) 热力图, 每次验证更新
        - MoE/heatmap_3d      图像:   同数据的三维折线图
        - MoE/expert_load      直方图: 所有层×专家的负载分布 (横轴=负载占比,
          纵轴=样本频次; 观察是否有专家空转/过载)
        """
        n = len(moe_stats)
        load_all = []
        drop_all = torch.zeros(n)
        entropy_all = torch.zeros(n)
        for i, (layer_idx, s) in enumerate(moe_stats):
            load_all.append(s["expert_load"])
            drop_all[i] = s["drop_rate"]
            entropy_all[i] = s["router_entropy"]
        writer.add_scalar("MoE/drop_rate", drop_all.mean().item(), step)
        writer.add_scalar("MoE/router_entropy",
                          entropy_all.mean().item(), step)
        writer.add_histogram("MoE/expert_load", torch.cat(load_all), step)
        # 逐帧累积 (step × 层) 矩阵, 每次验证都画全历史热力图
        if not hasattr(self, "_moe_heat_hist"):
            self._moe_heat_hist = {"drop": [], "entropy": []}
        self._moe_heat_hist["drop"].append(drop_all.numpy())
        self._moe_heat_hist["entropy"].append(entropy_all.numpy())
        self._log_moe_heatmap(writer, step)

    def _log_moe_heatmap(self, writer, step):
        """把截至目前的全历史逐层指标写成 3D 折线图 + 2D 热力图双视图。

        - MoE/heatmap_3d  图像: 每条线=一层, z=drop_rate, 每次验证追加
        - MoE/heatmap     图像: 左=drop_rate 右=router_entropy 热力图
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        hist_drop = np.asarray(
            self._moe_heat_hist["drop"])      # [T, n_layers]
        hist_ent = np.asarray(
            self._moe_heat_hist["entropy"])    # [T, n_layers]
        n_layer = hist_drop.shape[1]

        # ---- 3D 折线图: 整体走势 (x=帧, y=层, z=值) ----
        fig3d = plt.figure(figsize=(8.5, 5.2))
        ax3d = fig3d.add_subplot(111, projection="3d")
        x = np.arange(len(hist_drop))
        for li in range(n_layer):
            ax3d.plot(
                x, np.full(len(x), li), hist_drop[:, li],
                label=f"L{li}", linewidth=1.8,
            )
        ax3d.set_xlabel("校验帧")
        ax3d.set_ylabel("层")
        ax3d.set_zlabel("丢token率")
        ax3d.set_yticks(range(n_layer))
        ax3d.legend(loc="upper left", ncol=2, fontsize=8)
        ax3d.view_init(elev=22, azim=-60)
        fig3d.canvas.draw()
        img3d = np.ascontiguousarray(
            np.asarray(fig3d.canvas.buffer_rgba())[:, :, :3])
        writer.add_image("MoE/heatmap_3d", img3d, step, dataformats="HWC")
        plt.close(fig3d)

        # ---- 2D 热力图: drop_rate / router_entropy ----
        fig = plt.figure(figsize=(14, 4.2))
        gs = GridSpec(1, 2, width_ratios=[1.5, 1.5], wspace=0.3)
        for col, (tag, data) in enumerate(
            [("drop_rate", hist_drop), ("router_entropy", hist_ent)]
        ):
            ax = fig.add_subplot(gs[col])
            vmax = max(1.0, data.max())
            im = ax.imshow(
                data.T, aspect="auto", cmap="viridis",
                vmin=0.0, vmax=vmax,
            )
            ax.set_yticks(range(n_layer))
            ax.set_yticklabels([f"L{i}" for i in range(n_layer)])
            ax.set_xlabel(f"校验帧 (每帧={self.config.val_interval_step}步)")
            ax.set_title(f"MoE {tag}")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.canvas.draw()
        # matplotlib>=3.8: buffer_rgba() 代替 tostring_rgb()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        img = np.ascontiguousarray(buf)
        writer.add_image("MoE/heatmap", img, step, dataformats="HWC")
        plt.close(fig)

    def _active_params(self):
        """MoE 激活参数量: 专家权重按每 token 实际激活比例 K/N 折算, 其余全量。
        (总参数量含全部专家权重, 但推理/训练时每个 token 只经过 K 个专家)
        """
        args = self.config.model_args
        model = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model
        if not args.use_moe:
            return sum(p.numel() for p in model.parameters())
        total = 0
        for name, p in model.named_parameters():
            numel = p.numel()
            # MoE 专家权重 (w_gate/w_up/w_down) 只算激活的 K/N 部分
            if "mlp." in name and name.endswith(("w_gate", "w_up", "w_down")):
                numel = numel * args.n_experts_per_tok / args.n_experts
            total += numel
        return total

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
        if self.config.model_args.use_moe:
            active_params = self._active_params()
            print(
                f"激活参数（MoE 专家×{self.config.model_args.n_experts_per_tok}/"
                f"{self.config.model_args.n_experts}）：{active_params/1e6:.3f}M "
                f"({active_params/total_params*100:.1f}%)"
            )
            print(
                f"计算量（按激活参数）：{(nums_token * active_params * 6)/1e12:.2f} "
                f"TFLOPs * {self.config.epochs} = "
                f"{(nums_token * active_params * 6 * self.config.epochs)/1e12:.2f}TFLOPs"
            )
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

        # 预分配 .grad 缓冲，避免 CUDAGraph(torch.compile reduce-overhead) 梯度累积时
        # 访问被后续运行覆盖的 grad 张量（RMSNorm.forward 报错即此原因）
        for p in self.model.parameters():
            if p.requires_grad and p.grad is None:
                p.grad = torch.zeros_like(p)

        for epoch in range(self.config.epochs):
            bar = tqdm(self.train_loader, unit="step")
            # 跳过已训练的epoch
            if epoch < self.start_epoch:
                print(f"跳过已训练的epoch: {epoch}")
                continue
            elif epoch == self.start_epoch:
                bar.update(self.start_step)

            self.current_epoch = epoch

            # M4 修复：多 epoch 时每个 epoch 用 (seed + epoch) 重新打乱数据排列，
            # 否则第 2 个 epoch 起数据顺序与第 1 个完全相同（DataLoader 用 shuffle=False，
            # 且 set_permute_seed 只在 __init__ 调用一次）。种子含 epoch 保证续训可复现。
            # 穿透 random_split 产生的 Subset 拿到底层 PretrainTokenIDDataset。
            if self.config.dataset_shuffle_seed is not None:
                base_ds = self.train_loader.dataset
                while isinstance(base_ds, torch.utils.data.Subset):
                    base_ds = base_ds.dataset
                base_ds.set_permute_seed(self.config.dataset_shuffle_seed + epoch)

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

            acc_num = 0.0   # 累加组中有效 token 的 loss 分子（仅日志用，float；不再保留 autograd 图）
            acc_den = 0      # 累加组中有效 token 数
            for i, (train_inputs, train_targets, train_mask) in enumerate(
                self.train_loader
            ):
                # 跳过已训练的step（非 resume 时 start_step=0，不跳过任何 batch；
                # resume 时从 start_step 位置继续，`i <` 保证不重复消费）
                if i < self.start_step and epoch == self.start_epoch:
                    continue
                self.current_step = i
                # 提前计算 need_val：基于 i 和 step 后的 global_step（即 global_step+1）
                need_val = ((i % self.config.val_interval_step == 0) or (
                    (self.global_step) % self.config.ckpt_interval_step == 0
                )) and i > 0
                # 仅在需要记录时开启MoE统计，避免每步额外开销
                if self.moe_collector is not None:
                    self.moe_collector.set_enabled(need_val)
                num, den = self._train_step(train_inputs, train_targets, train_mask)
                if self.moe_collector is not None:
                    self.moe_collector.set_enabled(False)

                # 跨 micro-batch 累加：每个 micro-batch 立即 backward（释放本 batch 的 autograd
                # 图，显存只占 1 个 micro-batch），把「未归一化」的 num 梯度累加到 param.grad；
                # 边界处整体除以 Σden，等价于对 Σnum/Σden 做一次 backward，但显存回到单 batch 水平
                # （retain-graph 方案会因同时保留 batch_acceleration 个图导致 ~2x 显存暴涨）。
                den_int = int(den.item())
                acc_num += float(num.item())   # 仅日志用（有效 token loss 分子累加）
                acc_den += den_int

                self.scaler.scale(num).backward()

                is_step_boundary = (
                    (self.current_step + 1) % self.config.batch_acceleration == 0
                ) or (self.current_step + 1 == len(self.train_loader))
                if is_step_boundary:
                    if acc_den > 0:
                        # 把累加梯度除以 Σden，得到全局 token 加权的梯度（与 Σnum/Σden 的梯度一致）
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad.div_(acc_den)
                    grad_norm, grad_stats = self._optimizer_step(capture_grad_stats=need_val)
                    loss = acc_num / max(acc_den, 1)   # 整组 token 加权 loss（日志）
                    acc_num, acc_den = 0.0, 0
                else:
                    grad_norm = None
                    grad_stats = None
                    loss = float(num.item()) / max(den_int, 1)   # 单 micro-batch token 加权 loss（仅日志）
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
                    test_text = self.generate_test("人工智能")
                    writer.add_text(
                        "GeneratedText",
                        f"epoch_{epoch}_step_{i}: {test_text}",
                        self.global_step,
                    )
                    # checkpoint 保存（若是 ckpt 触发点）
                    # 注意：global_step 在上方已 +1，而 need_val 是按自增前的
                    # global_step 计算的，故这里用 global_step-1 对齐触发口径；
                    # 否则跨 epoch 时 i 与 global_step 错位，保存条件永远不成立（旧 bug：
                    # 261005 训练 71% 无任何 step-ckpt 落盘）
                    is_ckpt_step = (
                        (self.global_step - 1) % self.config.ckpt_interval_step == 0
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
        vocab_size = self.config.model_args.vocab_size
        device = self.device

        # 聚合统计放在 GPU 张量上，循环结束后统一 .item() 一次性同步，
        # 避免每 batch 一次 item() 强制 GPU->CPU 同步打断流水线（验证提速关键）
        loss_acc = torch.zeros((), device=device)
        entropy_acc = torch.zeros((), device=device)
        top1_acc = torch.zeros((), device=device, dtype=torch.long)
        top5_acc = torch.zeros((), device=device, dtype=torch.long)
        total_tokens = torch.zeros((), device=device, dtype=torch.long)

        with torch.inference_mode():
            for val_inputs, val_targets, val_mask in self.val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_mask = val_mask.to(device)
                with torch.autocast(str(device), enabled=self.config.use_amp, dtype=torch.bfloat16):
                    val_output = self.model(
                        val_inputs,
                        padding_mask=(val_inputs != self.config.model_args.pad_id),
                    )
                    logits = val_output.view(-1, vocab_size)
                    targets_flat = val_targets.view(-1)
                    mask_flat = val_mask.view(-1)
                    loss = self.criterion(logits, targets_flat)
                    loss_num = (loss * mask_flat).sum()   # 该 batch 有效 token 的 loss 总和
                    loss_acc += loss_num.float()          # 跨 batch 累加分子；分母用 total_tokens

                # 计算 entropy / top-k accuracy（用 float32 精度，仅对有效 token）
                probs = torch.softmax(logits.float(), dim=-1)
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum(dim=-1)  # (B*L,)
                valid_mask = mask_flat > 0
                entropy_acc += (entropy * mask_flat).sum()

                pred_top1 = probs.argmax(dim=-1)
                top1_acc += ((pred_top1 == targets_flat) & valid_mask).sum()

                if vocab_size >= 5:
                    _, pred_top5 = probs.topk(5, dim=-1)
                    top5_acc += (
                        (
                            pred_top5 == targets_flat.unsqueeze(-1)
                        ).any(dim=-1) & valid_mask
                    ).sum()
                else:
                    top5_acc = top1_acc  # vocab 不足时退化为 top1

                total_tokens += valid_mask.sum()

        n = len(self.val_loader)
        if n == 0:
            # val_loader 为空（数据过小时 drop_last 可能产生 0 个 val batch），
            # 避免除零崩溃，返回无效指标让调用处正常走日志
            print("警告: 验证集为空（0 个 batch），跳过本轮验证")
            weight_stats = self._compute_weight_stats()
            return (
                float("inf"),
                float("inf"),
                {
                    "entropy": 0.0,
                    "top1_acc": 0.0,
                    "top5_acc": 0.0,
                    **weight_stats,
                },
            )
        total = total_tokens.item()
        avg_loss = loss_acc.item() / max(total, 1)   # 与训练一致：按有效 token 数全局加权
        ppl = math.exp(avg_loss)
        avg_entropy = entropy_acc.item() / max(total, 1)
        top1_acc_val = top1_acc.item() / max(total, 1)
        top5_acc_val = top5_acc.item() / max(total, 1)

        # 参数统计量（仅算一次，模型参数在 val 期间不变）
        weight_stats = self._compute_weight_stats()

        return avg_loss, ppl, {
            "entropy": avg_entropy,
            "top1_acc": top1_acc_val,
            "top5_acc": top5_acc_val,
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

    def generate_test(self, start: str = "今天", gen_len: int = 25, verbose: bool = True,
                      temperature: float = 0.7, top_k: int = 20, top_p: float = None,
                      repetition_penalty: float = 1.0, frequency_penalty: float = 1.5):
        """文本生成测试（返回文本；verbose=True 时打印到控制台）"""
        self.model.eval()
        ans = self.generator.generate(
            start_token=start, gen_seq_len=gen_len, print_out=False,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, frequency_penalty=frequency_penalty,
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
