import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch
import bitsandbytes as bnb
import numpy as np
import matplotlib.pyplot as plt
import tokenizers
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import gc
import json

from dataclasses import dataclass
from typing import Optional, Dict, Any


# ---------------------------------------------------#
#   工具组件
# ---------------------------------------------------#
from utils import model_structure, TextGenerator, WarmUpCosineLR, DebugTimer
from dataset import PretrainTextDataset
from models_250830 import MyLM, MyLMArgs
from pre_train import PreTrainer


t = DebugTimer()


@dataclass
class TrainingConfig:
    """训练配置参数"""

    # 数据配置
    data_dir: str = r"data_sft.txt"
    tokenizer_dir: str = r"bpe_tokenizer_6k_0724_ChatML.json"
    model_save_dir: str = r"model\model_sft.pth"
    ckpt_save_dir: str = r"ckpt\ckpt.pth"
    log_dir: str = r"logs"
    padding_side = "left"

    # 训练参数
    seed: int = 42
    epochs: int = 1
    batch_size: int = 16
    batch_acceleration: int = 10
    dataset_downsample: int = 0.1
    valset_rate: float = 0.005
    val_interval_step: int = 600
    seq_max_len=192

    # 优化参数
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 5
    use_amp: bool = False

    model_args = MyLMArgs(
        d_model=480,
        d_inner=960,
        d_head=96,
        n_heads=None,
        n_layers=3,
        vocab_size=None,
        seq_max_len=seq_max_len,
        use_moe=False,
        n_experts=None,
        n_experts_per_tok=None,
        d_conv = None,
        conv_bias = None,
        ffn_bias = False,
        attn_bias = True,
        dropout = 0.1
    )

    # 新增参数：checkpoint保存间隔步数
    ckpt_interval_step: int = float("inf")
    train_from: Optional[str] = r"model\model_6k0724-0830-22M-768Mtok.pth"


class ContinueTrainer(PreTrainer):
    def __init__(self, config: TrainingConfig):
        config.resume_from = None
        super().__init__(config)

        if config.train_from is not None:
            self.load_checkpoint(config.train_from)
    def _build_model(self):
        """构建模型"""
        model = MyLM(self.config.model_args)
        return model

    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # 恢复模型状态
        if 'model_state_dict' in checkpoint:
            # 如果checkpoint包含完整的训练状态
            state_dict = checkpoint['model_state_dict']
            if any([k.startswith("module.") for k in state_dict.keys()]):
                print("该模型使用了DataParallel")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                print(f'{str(e)[:70]}...')
                miss, unexpect = self.model.load_state_dict(state_dict, strict=False)
                print(f'已使用非严格加载\n缺失{len(miss)}个参数，未匹配{len(unexpect)}个参数')
                if len(miss) < 10:
                    print(f'缺失参数：{miss}')
                if len(unexpect) < 10:
                    print(f'未匹配参数：{unexpect}')
        else:
            # 如果checkpoint只包含模型权重
            state_dict = checkpoint
            if any([k.startswith("module.") for k in state_dict.keys()]):
                print("该模型使用了DataParallel")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                print(f'{str(e)[:70]}...')
                miss, unexpect = self.model.load_state_dict(state_dict, strict=False)
                print(f'已使用非严格加载\n缺失{len(miss)}个参数，未匹配{len(unexpect)}个参数')
                if len(miss) < 10:
                    print(f'缺失参数：{miss}')
                if len(unexpect) < 10:
                    print(f'未匹配参数：{unexpect}')

        # 加载多个调度器的状态（按照PyTorch文档建议，在优化器之前加载）
        if "scheduler_states" in checkpoint and checkpoint.get("scheduler_states"):
            for i, sched_state in enumerate(checkpoint["scheduler_states"]):
                if i < len(self.schedulers):
                    self.schedulers[i].load_state_dict(sched_state)
        elif "scheduler_state_dict" in checkpoint:
            # 兼容旧版本checkpoint
            self.schedulers.load_state_dict(checkpoint["scheduler_state_dict"])

        # 加载多个优化器的状态
        if "optimizer_states" in checkpoint and checkpoint.get("optimizer_states"):
            for i, opt_state in enumerate(checkpoint["optimizer_states"]):
                if i < len(self.optimizers):
                    self.optimizers[i].load_state_dict(opt_state)
        elif "optimizer_state_dict" in checkpoint:
            # 兼容旧版本checkpoint
            self.optimizers.load_state_dict(checkpoint["optimizer_state_dict"])

        # 恢复训练状态
        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]
            self.start_epoch = checkpoint["epoch"]
            self.start_step = checkpoint["current_step"]
        
        # 恢复随机状态（防止数据shuffle混乱）
        if "rng_states" in checkpoint:
            rng_states = checkpoint["rng_states"]
            torch.set_rng_state(rng_states["torch"])
            if rng_states["cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states["cuda"])
            random.setstate(rng_states["random"])
            np.random.set_state(rng_states["numpy"])


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = ContinueTrainer(config)
    trainer.log()
    trainer.train()
    trainer.plot_losses()

    # 交互式测试
    MAX_LEN = 100
    T = 0.6
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
                        frequency_penalty=1,
                        top_k=20,
                        print_out=False,
                    )
                )
            )
