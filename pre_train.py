import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch

import bitsandbytes as bnb

import numpy as np
import matplotlib.pyplot as plt

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
import models
import tokenizers

t = DebugTimer()

@dataclass
class TrainingConfig:
    """训练配置参数"""
    # 数据配置
    data_dir: str = "data.txt"
    tokenizer_dir: str = "bpe_tokenizer_6k.json"
    model_save_dir: str = "model\model.pth"
    
    # 训练参数
    seed: int = 42
    epochs: int = 3
    batch_size: int = 32
    batch_acceleration: int = 1
    dataset_downsample: int = 60
    valset_rate: float = 0.005
    val_interval_step: int = 800

    
    # 优化参数
    learning_rate: float = 5e-3
    min_learning_rate: float = 5e-4
    warmup_steps: int = 60
    use_amp: bool = False
    
    # 模型参数
    d_model: int = 128
    d_inner: int = int(128 * (8/3))
    n_layers: int = 2
    use_moe: bool = False
    n_experts: int = 3
    vocab_size: int = None # 运行时获取
    seq_max_len: int = None # 运行时获取


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizers.Tokenizer.from_file(config.tokenizer_dir)
        self.train_loader, self.val_loader = self._build_dataloader()
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self._build_optimizer()
        self.scaler = torch.GradScaler(self.device, enabled=config.use_amp)
        self.generator = TextGenerator(self.model, self.tokenizer, self.device)
        
        # 用于扩展的属性
        self.current_epoch = 0
        self.global_step = 0
        self.train_loss_log = []
        self.val_loss_log = []
        self.lr_log = []

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
        args = models.MyLMArgs(
            d_model=self.config.d_model,
            d_inner=self.config.d_inner,
            n_layers=self.config.n_layers,
            use_moe=self.config.use_moe,
            n_experts=self.config.n_experts,
            vocab_size=self.tokenizer.get_vocab_size(),
            seq_max_len=self.seq_max_len, 
            conv_bias=False,
            ffn_bias=False,
            attn_bias=False,
            dropout=0.1,
        )
        model = models.MyLM(args)
        model_structure(model)
        return model

    def _build_dataloader(self):
        """构建数据加载器"""
        dataset = TextDatasetV4(
            self.config.data_dir,
            downsample=self.config.dataset_downsample,
            tokenizer=self.tokenizer,
            re_tokenize=False,
            batch=False,
        )
        self.seq_max_len  = dataset.seq_max_len
        val_dataset_len = int(len(dataset) * self.config.valset_rate)
        train_dataset_len = len(dataset) - val_dataset_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_dataset_len, val_dataset_len]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=False
        )
        
        print(f"词数: {self.tokenizer.get_vocab_size()}")
        print(f"数据集数量：{len(dataset)}")
        print(f"训练集数量：{train_dataset_len}")
        print(f"测试集数量：{val_dataset_len}")
        
        return train_loader, val_loader

    def _build_optimizer(self):
        """构建优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            amsgrad=True
        )
        
        scheduler = WarmUpCosineLR(
            optimizer,
            total_epochs=self.config.epochs * (len(self.train_loader)) + 1,
            warmup_epochs=self.config.warmup_steps,
            min_lr=self.config.min_learning_rate,
        )
        return optimizer, scheduler

    def save_checkpoint(self, path: str, is_final: bool = False):
        """保存checkpoint"""
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.train_loss_log[-1] if self.train_loss_log else None,
        }
        torch.save(state, path)
        if is_final:
            torch.save(self.model, self.config.model_save_dir)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint（预留扩展接口）"""
        # TODO: 实现checkpoint加载逻辑
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        

    def _train_step(self, inputs, targets, step_idx):
        """单步训练（含梯度累加）"""
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.autocast(str(self.device), enabled=self.config.use_amp):
            output = self.model(inputs)
            loss = self.criterion(
                output.view(-1, self.model.model_args.vocab_size), 
                targets.view(-1)
            )
        
        loss = loss / self.config.batch_acceleration

        self.scaler.scale(loss).backward()
        # 梯度裁剪（可选）
        # self.scaler.unscale_(self.optimizer)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if ((step_idx + 1) % self.config.batch_acceleration == 0) or (step_idx + 1 == len(self.train_loader)):
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        
        return loss.item() * self.config.batch_acceleration

    def train(self):
        """训练主循环"""
        self._set_seed()
        gc.collect()
        print("~~~训练咯~~~")
        
        # 多卡训练
        if torch.cuda.device_count() > 1:
            print(f"多卡训练: {torch.cuda.device_count()} 张GPU")
            self.model = nn.DataParallel(self.model)

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            train_loss_sum = 0
            bar = tqdm(self.train_loader, unit="batch")
            torch.cuda.empty_cache()
            
            for i, (train_inputs, train_targets) in enumerate(self.train_loader):
                loss = self._train_step(train_inputs, train_targets, i)

                
                train_loss_sum += loss
                self.train_loss_log.append(loss)
                self.lr_log.append(float(self.scheduler.get_last_lr()[0]))
                self.global_step += 1
                
                # 验证
                if i % self.config.val_interval_step == 0:
                    val_loss = self.validate()
                    self.val_loss_log.append((self.global_step, val_loss))
                
                # 进度显示
                bar.update(1)
                bar.postfix = f"Loss: {round(loss, 2)} lr: {'{:.3e}'.format(self.scheduler.get_last_lr()[0])}"
            
            bar.close()
            
            # 周期性验证
            val_loss = self.validate()
            self.val_loss_log.append((self.global_step, val_loss))
            
            # 文本生成测试
            self.generate_test()
            
            # 每epoch完整模型保存
            self.save_checkpoint(f"{''.join(self.config.model_save_dir.split(r'\\')[:-1])}epoch_ckpt_{epoch}.pth", is_final=True)

            print(f"学习率{self.scheduler.get_last_lr()}")
            print(f"Epoch: {epoch+1}/{self.config.epochs}, avg_train_loss: {train_loss_sum/len(self.train_loader)}, avg_test_loss: {val_loss}")
        
        # 保存最终模型
        self.save_checkpoint(self.config.model_save_dir, is_final=True)

    def validate(self) -> float:
        """验证过程"""
        self.model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for val_inputs, val_targets in self.val_loader:
                val_inputs = val_inputs.to(self.device)
                val_targets = val_targets.to(self.device)
                with torch.autocast(str(self.device), enabled=self.config.use_amp):
                    val_output = self.model(val_inputs)
                    loss = self.criterion(
                        val_output.view(-1, self.model.model_args.vocab_size), 
                        val_targets.view(-1)
                    )
                val_loss_sum += loss.item()
        return val_loss_sum / len(self.val_loader)

    def generate_test(self, start: str = "我", gen_len: int = 25):
        """文本生成测试"""
        ans = self.generator.generate(
            start_token=start,
            gen_seq_len=gen_len,
            print_out=False
        )
        ans = ans[len(start):]  # 截掉start_token
        print(f"(input){start}->", end="")
        for token in ans:
            print(token, end=" ")
        print("\n")

    def plot_losses(self):
        """绘制损失曲线"""
        fig, ax1 = plt.subplots(figsize=(16, 10))
        
        # 提取验证步骤和损失
        val_steps = [step for step, loss in self.val_loss_log]
        val_losses = [loss for step, loss in self.val_loss_log]
        
        # 绘制左侧的训练损失和验证损失
        ax1.plot(range(len(self.train_loss_log)), self.train_loss_log, label="Train Loss")
        ax1.plot(val_steps, val_losses, "o-", label="Test Loss")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper left")
        
        # 创建右侧y轴
        ax2 = ax1.twinx()
        ax2.plot(range(len(self.lr_log)), self.lr_log, label="Learning Rate", color="c", linestyle="--")
        ax2.set_ylabel("Learning Rate")
        ax2.tick_params(axis="y")
        ax2.legend(loc="upper right")
        
        plt.title("Train and Test Loss Curves with Learning Rate")
        plt.show()




if __name__ == "__main__":
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()
    trainer.plot_losses()
    
    # 交互式测试
    MAX_LEN = 100
    T = 0.8
    while True:
        start = input("In>>")
        if start[:2] == 'T=':
            T = float(start[2:])
            print(f'T={T}')
        else:
            print(f'T={T}\n' +
                "".join(
                    trainer.generator.generate(
                        start_token=start,
                        gen_seq_len=MAX_LEN,
                        temperature=T,
                        frequency_penalty=10,
                        print_out=False
                    )
                )
            )