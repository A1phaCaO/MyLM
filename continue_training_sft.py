import os
# 设置环境变量以解决OpenMP库重复初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# 注意：Windows 上 torch.compile(inductor) 需先设置 PYTHONUTF8=1，
# 否则读模板时 GBK 解码崩溃：$env:PYTHONUTF8='1'; uv run python continue_training_sft.py

import torch
import torch.nn as nn
import time
import json
import gc
import random
import numpy as np

from dataclasses import dataclass, asdict
from typing import Optional

from models import MyLM, MyLMArgs
from dataset import SFTTextDataset
from pre_train import PreTrainer
from utils import DebugTimer


t = DebugTimer()


@dataclass
class TrainingConfig:
    """SFT 训练配置参数（机制对齐 pre_train.py：WSD 调度 + Muon/AdamW8bit 双优化器
    + bf16 AMP + torch.compile + checkpoint 自动清理/断点续训）"""

    # 数据配置
    # SFT 数据：每行 = 一段 ChatML 对话（未预分词），格式示例：
    #   "<|im_start|>user\n问题内容<|im_end|>\n<|im_start|>assistant\n回答内容<|im_end|>\n"
    data_dir: str = r"data/data_sft512v3.txt"
    tokenizer_dir: str = r"tokenizer/bbpe_tokenizer_7k_260723_xl.json"
    model_save_dir: str = r"model\model_m_sftv2.pth"
    ckpt_save_dir: str = r"ckpt\sft_ckpt.pth"
    config_save_dir: str = r"model\config_m_sftv2.json"
    log_dir: str = r"logs/sft/" +"sft_m_v2"+ time.strftime("%y%m%d-%H%M")
    padding_side: str = "right"   # 右 pad：左截断保留回答尾部 + pad 加在尾部，
    #                                 causal mask 天然隔离 pad，不再依赖 seq_mask 零假设

    # 训练参数
    seed: int = 42
    epochs: int = 2
    batch_size: int = 32
    batch_acceleration: int = 1
    dataset_downsample: float = 1.0   # (0,1)=降采样, 1=全量, >1=重复（SFTTextDataset 语义）
    valset_rate: float = 0.01
    val_interval_step: int = 500
    seq_max_len: int = 512   # 与 xl pretrain 模型一致
    use_compile: bool = True
    compile_mode: str = "max-autotune"
    # compile 时是否将 MoE 层排除在外（已过时，保持默认 False）
    exclude_moe_from_compile: bool = False

    # 优化参数（SFT 微调远小于 pretrain 的 4e-3）
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5   # WSD 衰减到峰值 LR 的 50%（若需按项目约定衰减到 10% 改为 1e-5）
    lr_decay_start_rate: int = 0.8    # 最后 20% 步数线性衰减
    warmup_steps: int = 2
    use_amp: bool = True

    model_args = MyLMArgs(
        d_model=512,
        latent_moe=False,
        d_latent=256,
        d_inner=int(((512 * (8 / 3)) // 64) * 64),
        d_head=128,
        n_heads=None,
        n_layers=6,
        vocab_size=None,
        seq_max_len=seq_max_len,
        use_moe=False,
        n_experts=12,
        n_experts_per_tok=2,
        d_conv=None,
        conv_bias=None,
        ffn_bias=False,
        attn_bias=True,
        dropout=0.0,          # SFT 微调关闭 dropout
        base_init_std=0.02,
    )

    # checkpoint 保存间隔步数（自动清理：保留最近 ckpt_keep_recent 个，
    # 更早的每隔 ckpt_keep_stride 个保留 1 个）
    ckpt_interval_step: int = 1000
    ckpt_keep_recent: int = 3
    ckpt_keep_stride: int = 3
    # 断点续训（PreTrainer 原生支持）：完整训练状态 checkpoint
    resume_from: Optional[str] = None
    # 数据集固定种子 shuffle（续训按已消费位置继续，数据不重复）
    dataset_shuffle_seed: Optional[int] = 42

    # SFT 特有：从 pretrain 权重开始微调（纯权重文件或完整 checkpoint 均可）
    train_from: Optional[str] = r"model\model_dense_m_0813v2.pth"


class SFTTrainer(PreTrainer):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        # 与父类一致：criterion 用 reduction="none"（父类 PreTrainer 已是如此），
        # 由 mask 对有效 token 加权，归一化在 _train_step / validate 内按有效 token 数完成。
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        # 从 pretrain 权重继续训练（宽松加载，见 load_checkpoint）。
        # 与 resume_from 互斥：resume 的完整训练状态已含权重，再用 train_from
        # 覆盖会导致"权重=pretrain、优化器=恢复态"的混合状态
        if config.train_from is not None and config.resume_from is None:
            self.load_checkpoint(config.train_from)
        # 不再需要 _zero_pad_embedding：SFT 已改为右 padding，pad 在尾部，
        # causal mask 天然隔离，不依赖 seq_mask 的零假设。
        # （旧版左 padding 时置零 embedding 行试图让 seq_mask 生效，但
        #  attn_bias=True 的 o_proj 可学习偏置会在训练几步后使 pad 行
        #  残差流非零，从第 2 层起 seq_mask 失效——置零只是安慰剂。）

    def _build_dataloader(self):
        """构建数据加载器（SFT mask 数据集）"""
        dataset = SFTTextDataset(
            self.config.data_dir,
            tokenizer_path=self.config.tokenizer_dir,  # 内部独立加载，不被 TextGenerator 污染
            seq_max_len=self.config.seq_max_len,
            downsample=self.config.dataset_downsample,
            padding_side=self.config.padding_side,
            shuffle_seed=self.config.dataset_shuffle_seed,
        )
        val_dataset_len = int(len(dataset) * self.config.valset_rate)
        train_dataset_len = len(dataset) - val_dataset_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_dataset_len, val_dataset_len]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            # 已由 SFTTextDataset(shuffle_seed=...) 固定排列，
            # 续训按已消费 batch 位置继续，数据不重复
            shuffle=False,
            drop_last=True,   # 保证 batch 形状恒定（torch.compile reduce-overhead 稳定）
            pin_memory=True,
            num_workers=4,
            prefetch_factor=4,
            persistent_workers=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
            num_workers=2,
            prefetch_factor=3,
            persistent_workers=True
        )

        return train_loader, val_loader

    def load_checkpoint(self, checkpoint_path: str):
        """加载 checkpoint（宽松版，兼容 pretrain 产物）：
        - 支持纯权重 dict 与完整训练状态 dict
        - 自动剥离 DataParallel 的 "module." 与 torch.compile 的 "_orig_mod." 前缀
        - 纯权重时跳过优化器/调度器/训练状态恢复（恢复即覆盖，微调语义不符）
        """
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            has_full_state = True
        else:
            state_dict = checkpoint
            has_full_state = False

        # 兼容旧命名（如 models_250723 时代的 ckpt，权重键带 module.）
        state_dict = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }
        # 剔除 shape 不匹配的键：典型是 RoPE 的 cos_cached/sin_cached，
        # 尺寸随 seq_max_len 变化（pretrain 256 → SFT 512），load_state_dict
        # 对 size mismatch 无论 strict 与否都会抛异常（权重整体加载失败 =
        # 模型保持随机初始化，loss 会接近 ln(vocab)）。剔除后模型保留
        # 自己初始化的大尺寸缓冲（值由位置公式决定，与训练无关）。
        # 注意必须在裸模型（self.model._orig_mod）上过滤/加载：self.model
        # 是 torch.compile 的 OptimizedModule，其 state_dict 键带 _orig_mod.
        # 前缀，与剥离前缀后的 checkpoint 键不一致（否则全部键被剔除，权重
        # 一个都加载不上，等价于随机初始化）。
        model = getattr(self.model, "_orig_mod", self.model)
        model_sd = model.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f'{str(e)[:70]}...')
            miss, unexpect = model.load_state_dict(state_dict, strict=False)
            print(f'已使用非严格加载\n缺失{len(miss)}个参数，未匹配{len(unexpect)}个参数')
            if len(miss) < 10:
                print(f'缺失参数：{miss}')
            if len(unexpect) < 10:
                print(f'未匹配参数：{unexpect}')

        if not has_full_state:
            print("纯权重 checkpoint：跳过优化器/调度器/训练状态恢复")
            return

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

    def generate_test(self, start: str = "你好", gen_len: int = 80,
                      temperature: float = 0.7, top_k: int = 20,
                      top_p: float = 0.95, repetition_penalty: float = 1.1,
                      frequency_penalty: float = 1.0, verbose: bool = True):
        """对话式生成测试：包装成 ChatML 并在 <|im_end|> 处停止。
        采样超参与 run_model_for_state.py 对齐（temperature/top_k/top_p/
        repetition_penalty/frequency_penalty），保证训练 val 与推理一致。"""
        self.model.eval()
        # 字面 \n 对齐 SFT 数据格式（tokenizer 把真实换行归为 <|unk|>）
        prompt = f"<|im_start|>user\\n{start}<|im_end|>\\n<|im_start|>assistant\\n"
        im_end_id = self.tokenizer.token_to_id("<|im_end|>")
        ans = self.generator.generate(
            start_token=prompt,
            gen_seq_len=gen_len,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            print_out=False,
            eos_id=im_end_id,  # 生成到 <|im_end|> 自动停止
        )
        text = ans  # generate 现在返回 str（旧版 "".join(list) 已不需要）
        # 兜底：若 eos_id 未命中（模型未学会停止），手动截断
        cut = text.find("<|im_end|>", len(prompt))
        if cut != -1:
            text = text[:cut]
        result = text[len(prompt):]
        if verbose:
            print(f"(input){start}\n-> {result}")
        return result


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = SFTTrainer(config)
    config_dict = asdict(config.model_args)
    with open(config.config_save_dir, "w") as f:
        json.dump(config_dict, f, indent=4)
    trainer.log()
    trainer.train()
    trainer.plot_losses()

    # 交互式测试
    while True:
        start = input("In>>")
        if start[:2] == "T=":
            T = float(start[2:])
            print(f"T={T}")
        elif start[:2] == "L=":
            gen_len = int(start[2:])
            print(f"L={gen_len}")
        else:
            print(
                trainer.generate_test(
                    start=start,
                    gen_len=80,
                    temperature=0.7,
                    top_k=20,
                    frequency_penalty=1.2,
                    verbose=False,
                )
            )