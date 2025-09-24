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

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


# ---------------------------------------------------#
#   工具组件
# ---------------------------------------------------#
from utils import model_structure, TextGenerator, WarmUpCosineLR, DebugTimer
from dataset import TextDatasetV4, RuntimeTextDatasetV4, StreamingTextDataset
from models import MyLMArgs, MyLM

t = DebugTimer()


@dataclass
class TrainingConfig:
    """训练配置参数"""

    # 数据配置
    data_dir: str = r"data_large_ChatML.txt"
    tokenizer_dir: str = r"bpe_tokenizer_6k_0724_ChatML.json"
    model_save_dir: str = r"model\model_state.pth"
    ckpt_save_dir: str = r"ckpt\ckpt.pth"
    config_save_dir: str = r"config.json"
    log_dir: str = r"logs"
    padding_side = "left"

    # 训练参数
    seed: int = 42
    epochs: int = 4
    batch_size: int = 32
    batch_acceleration: int = 4
    dataset_downsample: int = 0.001
    valset_rate: float = 0.01
    val_interval_step: int = 1000
    seq_max_len=192

    # 优化参数
    learning_rate: float = 5e-3
    min_learning_rate: float = 5e-4
    warmup_steps: int = 1
    use_amp: bool = False

    model_args = MyLMArgs(
        d_model=192,
        d_inner=int(((192 * (8 / 3)) // 64) * 64),
        d_head=64,
        n_heads=None,
        n_layers=1,
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
    ckpt_interval_step: int = 1000
    # 新增参数：断点续训的checkpoint路径
    # resume_from: Optional[str] = r"ckpt\ckpt_epoch_1_step_27000.pth"
    resume_from: Optional[str] = None


class PreTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._set_seed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizers.Tokenizer.from_file(config.tokenizer_dir)
        self.config.model_args.vocab_size = len(self.tokenizer.get_vocab())
        self.train_loader, self.val_loader = self._build_dataloader()
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer, self.scheduler = self._build_optimizer()
        self.scaler = torch.GradScaler(self.device, enabled=config.use_amp)
        self.generator = TextGenerator(
            self.model, self.tokenizer, self.device, padding_side=config.padding_side
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
        return model

    def _build_dataloader(self):
        """构建数据加载器"""
        dataset = StreamingTextDataset(
            self.config.data_dir,
            downsample=self.config.dataset_downsample,
            seq_max_len=self.config.seq_max_len,
            tokenizer=self.tokenizer,
            re_tokenize=False,
            batch=False,
            padding_side=self.config.padding_side,
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
            pin_memory=False,
            num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=1
        )

        return train_loader, val_loader

    def _build_optimizer(self):
        """构建优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            amsgrad=False,
            betas=(0.85, 0.999),
            eps=1e-6,
            weight_decay=0.005,
        )

        scheduler = WarmUpCosineLR(
            optimizer,
            total_epochs=(
                self.config.epochs
                * (len(self.train_loader) // self.config.batch_acceleration + 1)
            )
            + 1,
            warmup_epochs=self.config.warmup_steps,
            min_lr=self.config.min_learning_rate,
        )
        return optimizer, scheduler

    def save_checkpoint(self, path: str, is_final: bool = False):
        # 检测是否是DataParallel模式
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        """保存checkpoint"""
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "current_step": self.current_step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
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
        if is_final:
            torch.save(model_state_dict, self.config.model_save_dir)

    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # 恢复模型状态
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

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

    def _train_step(self, inputs, targets):
        """单步训练（含梯度累加）"""
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.autocast(str(self.device), enabled=self.config.use_amp):
            output = self.model(inputs)
            loss = self.criterion(
                output.view(-1, self.config.model_args.vocab_size), targets.view(-1)
            )

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

    def log(self):
        total_params = model_structure(self.model)
        print(f"本次训练参数：")
        print(f"词数: {self.tokenizer.get_vocab_size()}")
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
        writer = SummaryWriter(log_dir=self.config.log_dir)
        print("~~~训练咯~~~")

        # 多卡训练
        if torch.cuda.device_count() > 1:
            print(f"多卡训练: {torch.cuda.device_count()} 张GPU")
            self.model = nn.DataParallel(self.model)
        val_loss = 0
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

            torch.cuda.empty_cache()

            for i, (train_inputs, train_targets) in enumerate(self.train_loader):
                # 跳过已训练的step
                if i <= self.start_step and epoch == self.start_epoch:
                    continue
                # # 如果是从断点恢复训练 打印提示
                # if i == self.start_step + 1:
                #     print(f"\r从断点恢复训练: {i}", end="")
                self.current_step = i
                loss = self._train_step(train_inputs, train_targets)

                train_loss_sum += loss
                self.train_loss_log.append((self.global_step, loss))
                # 添加TensorBoard训练损失记录
                writer.add_scalar("Loss/train", loss, self.global_step)
                # 记录学习率时添加step信息
                self.lr_log.append(
                    (self.global_step, float(self.scheduler.get_last_lr()[0]))
                )
                self.global_step += 1
                # 添加TensorBoard学习率记录
                writer.add_scalar(
                    "LearningRate",
                    float(self.scheduler.get_last_lr()[0]),
                    self.global_step,
                )

                # 验证阶段
                if i % self.config.val_interval_step == 0:
                    val_loss = self.validate()
                    # 文本生成测试
                    self.generate_test("人工智能是")
                    self.val_loss_log.append((self.global_step, val_loss))
                    # 添加TensorBoard验证损失记录
                    writer.add_scalar("Loss/val", val_loss, self.global_step)

                # 进度显示
                bar.update(1)
                bar.postfix = f"train_loss: {loss:.2f} test_loss: {val_loss:.2f} lr: {self.scheduler.get_last_lr()[0]:.2e}"

                   

                # 每n步直接保存checkpoint
                if self.global_step % self.config.ckpt_interval_step == 0:
                    val_loss = self.validate()
                    self.val_loss_log.append((self.global_step, val_loss))
                    # 生成带步数的checkpoint路径
                    ckpt_path = f"{self.config.ckpt_save_dir.rsplit('.', 1)[0]}_epoch_{self.current_epoch}_step_{self.global_step}.pth"
                    self.save_checkpoint(ckpt_path, is_final=False)

            bar.close()

            # 周期性验证
            val_loss = self.validate()
            self.val_loss_log.append((self.global_step, val_loss))

            # 文本生成测试
            self.generate_test()

            # 修改：epoch级保存使用更明确的文件名格式
            self.save_checkpoint(
                f"{self.config.ckpt_save_dir.rsplit('.', 1)[0]}_epoch_{epoch}.pth",
                is_final=True,
            )

            # 每个epoch记录生成文本
            test_text = self.generate_test(gen_len=100)
            writer.add_text(
                "GeneratedText", f"epoch_{epoch}: {test_text}", self.global_step
            )

            print(f"学习率{self.scheduler.get_last_lr()}")
            print(
                f"Epoch: {epoch+1}/{self.config.epochs}, avg_train_loss: {train_loss_sum/len(self.train_loader)}, avg_test_loss: {val_loss}"
            )

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
                        val_output.view(-1, self.config.model_args.vocab_size),
                        val_targets.view(-1),
                    )
                val_loss_sum += loss.item()
        return val_loss_sum / len(self.val_loader)

    def generate_test(self, start: str = "我", gen_len: int = 25):
        """文本生成测试（修改为返回文本）"""
        self.model.eval()
        ans = self.generator.generate(
            start_token=start, gen_seq_len=gen_len, print_out=False
        )
        ans = ans[len(start) :]  # 截掉start_token
        result = "".join(ans)
        # 保持原有控制台输出
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
        ax1.plot(train_steps, train_losses, label="Train Loss")  # 修改为使用step作为x轴
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
    config_dict = asdict(config.model_args)
    with open(config.config_save_dir, "w") as f:
        json.dump(config_dict, f, indent=4)
    trainer = PreTrainer(config)
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
