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
from dataset import TextDatasetV4
import models_250718 as models
from pre_train import PreTrainer


t = DebugTimer()


@dataclass
class TrainingConfig:
    """训练配置参数"""

    # 数据配置
    data_dir: str = r"data_sft.txt"
    tokenizer_dir: str = r"bpe_tokenizer_6k_0717.json"
    model_save_dir: str = r"model\model_6k0717-0718-16M-1_5G_insturct.pth"
    ckpt_save_dir: str = r"ckpt\ckpt.pth"
    log_dir: str = r"logs"
    padding_side = "left"

    # 训练参数
    seed: int = 42
    epochs: int = 1
    batch_size: int = 32
    batch_acceleration: int = 12
    dataset_downsample: int = 10
    valset_rate: float = 0.02
    val_interval_step: int = 120

    # 优化参数
    learning_rate: float = 8e-5
    min_learning_rate: float = 8e-6
    warmup_steps: int = 3
    use_amp: bool = False

    # 模型参数
    d_model: int = 384
    d_inner: int = int(((384 * (8 / 3)) // 64) * 64)
    n_layers: int = 2
    use_moe: bool = False
    n_experts: int = 3
    vocab_size: int = None  # 运行时获取
    seq_max_len: int = None  # 运行时获取

    # 新增参数：checkpoint保存间隔步数
    ckpt_interval_step: int = float("inf")
    train_from: Optional[str] = r"model\model_6k0717-0718-16M-1_5G.pth"


class ContinueTrainer(PreTrainer):
    def __init__(self, config: TrainingConfig):
        config.resume_from = None
        super().__init__(config)

        if config.train_from is not None:
            self.load_checkpoint(config.train_from)
    def _build_model(self):
        """构建模型"""
        args = models.MyLMArgs(
            d_model=self.config.d_model,
            d_inner=self.config.d_inner,
            n_layers=self.config.n_layers,
            use_moe=self.config.use_moe,
            n_experts=self.config.n_experts,
            vocab_size=self.config.vocab_size,
            seq_max_len=self.config.seq_max_len,
            conv_bias=False,
            ffn_bias=False,
            attn_bias=False,
            dropout=0.1,
        )
        model = models.MyLM(args)
        return model

    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        print(f"加载checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=False)
        if any([k.startswith("module.") for k in state_dict.keys()]):
            print("该模型使用了DataParallel")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # 恢复模型状态
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


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = ContinueTrainer(config)
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
                        frequency_penalty=10,
                        print_out=False,
                    )
                )
            )
