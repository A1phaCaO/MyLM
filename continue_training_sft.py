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
        找到子串在序列中的所有位置
        """
        if len(subseq) == 0:
            return torch.tensor([], dtype=torch.long, device=seq.device)
        
        # 创建滑动窗口视图 (seq_len - sub_len + 1, sub_len)
        if len(seq) < len(subseq):
            return torch.tensor([], dtype=torch.long, device=seq.device)
        
        windows = seq.unfold(0, len(subseq), 1)
        # 比较每个窗口与子序列
        matches = (windows == subseq).all(dim=1)
        # 获取匹配位置的索引
        match_indices = torch.nonzero(matches, as_tuple=False).squeeze(dim=1)
        
        if match_indices.numel() == 0:  # 没有找到匹配
            return torch.tensor([], dtype=torch.long, device=seq.device)
        elif match_indices.dim() == 0:  # 只有一个匹配
            match_indices = match_indices.unsqueeze(0)
        
        # 返回每个匹配的结束位置（子序列最后一个元素的索引）
        positions = match_indices + len(subseq) - 1
        return positions

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
            # 第一步：找到回答的开始和结束位置，并对回答内容设置mask为1
            ans_start_pos_list = self._find_subsequence(seq, self.answer_start_ids)
            ans_end_pos_list = self._find_subsequence(seq, self.eos_ids)
            
            # 如果找到了回答开始和结束位置，则对回答内容设置mask为1
            if len(ans_start_pos_list) > 0:
                # 对于每个问答对，设置回答内容的mask为1
                for j, ans_start_pos in enumerate(ans_start_pos_list):
                    # 找到对应的回答结束位置（下一个eos或最后一个eos）
                    if j < len(ans_end_pos_list):
                        ans_end_pos = ans_end_pos_list[j]
                    elif len(ans_end_pos_list) > 0:
                        # 如果开始位置比结束位置多，使用最后一个结束位置
                        ans_end_pos = ans_end_pos_list[-1]
                    else:
                        # 如果没有对应的结束位置，使用序列末尾
                        ans_end_pos = seq_len - 1
                    
                    # 设置回答内容的mask为1（从回答开始的下一个位置到回答结束位置）
                    start_idx = min(ans_start_pos + 1, seq_len - 1)
                    end_idx = min(ans_end_pos + 1, seq_len)  # +1 因为切片是左闭右开
                    if start_idx < seq_len and end_idx > start_idx:
                        loss_mask[i, start_idx:end_idx] = 1
            
            # 第二步：特殊token不mask，即设置mask为1（表示参与损失计算）
            bos_pos_list = self._find_subsequence(seq, self.bos_ids)
            for pos in bos_pos_list:
                if pos < seq_len:
                    loss_mask[i, pos] = 1
                    
            eos_pos_list = self._find_subsequence(seq, self.eos_ids)
            for pos in eos_pos_list:
                if pos < seq_len:
                    loss_mask[i, pos] = 1
                    
            end_pos_list = self._find_subsequence(seq, self.end_ids)
            for pos in end_pos_list:
                if pos < seq_len:
                    loss_mask[i, pos] = 1

            

        return loss_mask

    def _train_step(self, inputs, targets):
        """单步训练（含梯度累加）"""
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # 创建loss mask
        loss_mask = self._create_loss_mask(inputs)

        with torch.autocast(str(self.device), enabled=self.config.use_amp):
            output = self.model(inputs)
            # 使用自定义的损失函数，reduction='none'以应用mask
            loss_per_token = self.criterion(
                output.view(-1, self.config.model_args.vocab_size), targets.view(-1)
            )
            # 应用loss mask
            masked_loss = loss_per_token * loss_mask.view(-1)
            # 计算平均损失，只考虑mask为1的位置
            loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)  # 防止除零

        loss = loss / self.config.batch_acceleration

        self.scaler.scale(loss).backward()
        # 梯度裁剪（可选）

        if ((self.current_step + 1) % self.config.batch_acceleration == 0) or (
            self.current_step + 1 == len(self.train_loader)
        ):
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        return loss.item() * self.config.batch_acceleration

    def validate(self) -> float:
        """验证过程，使用loss mask进行SFT验证"""
        self.model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for val_inputs, val_targets in self.val_loader:
                val_inputs = val_inputs.to(self.device)
                val_targets = val_targets.to(self.device)
                
                # 创建loss mask
                loss_mask = self._create_loss_mask(val_inputs)
                
                with torch.autocast(str(self.device), enabled=self.config.use_amp):
                    val_output = self.model(val_inputs)
                    # 使用自定义的损失函数，reduction='none'以应用mask
                    loss_per_token = self.criterion(
                        val_output.view(-1, self.config.model_args.vocab_size),
                        val_targets.view(-1),
                    )
                    # 应用loss mask
                    masked_loss = loss_per_token * loss_mask.view(-1)
                    # 计算平均损失，只考虑mask为1的位置
                    loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)  # 防止除零
                    
                val_loss_sum += loss.item()
        return val_loss_sum / len(self.val_loader)


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
