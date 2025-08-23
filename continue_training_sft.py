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
from models_250723 import MyLM, MyLMArgs
from continue_training import ContinueTrainer


t = DebugTimer()


@dataclass
class TrainingConfig:
    """训练配置参数"""

    # 数据配置
    data_dir: str = r"data_sft256_ChatML_old.txt"
    tokenizer_dir: str = r"bpe_tokenizer_6k_0717.json"
    model_save_dir: str = r"model\model_6k0717-0723-18M-1_7G_insturct_chatml_test.pth"
    ckpt_save_dir: str = r"ckpt\ckpt.pth"
    log_dir: str = r"logs"
    padding_side = "left"

    # 训练参数
    seed: int = 42
    epochs: int = 1
    batch_size: int = 16
    batch_acceleration: int = 10
    dataset_downsample: int = 5
    valset_rate: float = 0.01
    val_interval_step: int = 120

    # 优化参数
    learning_rate: float = 8e-5
    min_learning_rate: float = 8e-6
    warmup_steps: int = 5
    use_amp: bool = False

    model_args = MyLMArgs(
        d_model=432,
        d_inner=int(((432 * (8 / 3)) // 64) * 64),
        d_head=72,
        n_heads=6,
        n_layers=2,
        vocab_size=None,
        seq_max_len=None,
        use_moe=False,
        n_experts=None,
        n_experts_per_tok=None,
        d_conv=None,
        conv_bias=None,
        ffn_bias=False,
        attn_bias=True,
        dropout=0.1,
    )

    # 新增参数：checkpoint保存间隔步数
    ckpt_interval_step: int = float("inf")
    train_from: Optional[str] = r"model\model_6k0717-0723-18M-1_7G.pth"


class SFTTrainer(ContinueTrainer):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        # 获取特殊token的ID
        self.answer_start_ids = self.tokenizer.encode("<|im_start|>assistant\n").ids
        self.bos_ids = self.tokenizer.encode("<|im_start|>").ids
        # self.assistant_ids = self.tokenizer.encode("assistant\n").ids
        # self.user_ids = self.tokenizer.encode("user\n").ids
        # self.system_ids = self.tokenizer.encode("system\n").ids
        self.eos_ids = self.tokenizer.encode("<|im_end|>").ids
        self.end_ids = self.tokenizer.encode("<|endoftext|>").ids
        self.criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=0)

    def _find_subsequence(
        self, seq: torch.Tensor[int], subseq: torch.Tensor[int]
    ) -> torch.Tensor[int]:
        """
        找到子串在序列中的最后一个位置
        """
        # 创建滑动窗口视图 (seq_len - sub_len + 1, sub_len)
        windows = seq.unfold(0, len(subseq), 1)
        # 比较每个窗口与子序列
        matches = (windows == subseq).all(dim=1)
        # 获取匹配位置的索引
        pos = (torch.nonzero(matches, as_tuple=False).squeeze(dim=1) + 1) * len(
            self.answer_start_ids
        ) - 1
        return pos

    def _create_loss_mask(self, inputs):
        """
        采用Qwen策略：
        - <|im_start|>, <|im_end|>等特殊token不作mask；
        - 对system和每轮query的内容添加mask；
        - 每轮对话中的角色信息（"system\n", "user\n", "assistant\n"）添加mask。
        （目前无system角色）
        """
        batch_size, seq_len = inputs.shape
        loss_mask = torch.zeros_like(inputs)
        for i, seq in enumerate(inputs):
            # 第一步：回答不mask
            ans_start_pos = self._find_subsequence(seq, self.answer_start_ids)
            ans_end_pos = self._find_subsequence(seq, self.eos_ids)
            loss_mask[i, ans_start_pos + 1 : ans_end_pos - 1] = 1
            
            # 第二步：特殊token不mask
            bos_pos = self._find_subsequence(seq, self.bos_ids)
            loss_mask[i, bos_pos] = 1
            eos_pos = self._find_subsequence(seq, self.eos_ids)
            loss_mask[i, eos_pos] = 1
            end_pos = self._find_subsequence(seq, self.end_ids)
            loss_mask[i, end_pos] = 1

            

        return loss_mask

    def _train_step(self, inputs, targets):
        NotImplementedError("TODO: 把mask引入训练")


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = SFTTrainer(config)
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
