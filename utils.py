import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tokenizers


class TextGenerator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: tokenizers.Tokenizer,
        device,
        padding_side="right",
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        # self.padding_side = padding_side
        if isinstance(model, nn.DataParallel):
            print("该模型使用了DataParallel")
            self.model = model.module
        else:
            self.model = model
        self.seq_max_len = self.model.args.seq_max_len
        self.padding_side = padding_side
        self.tokenizer.enable_padding(direction=padding_side, length=self.seq_max_len)
        self.tokenizer.enable_truncation(
            max_length=self.seq_max_len, direction=padding_side
        )

    def generate(
        self,
        start_token: str,
        gen_seq_len=30,
        temperature=0.7,
        frequency_penalty=1.5,
        top_k=20,
        print_out=True,
    ):
        with torch.no_grad():
            self.model.eval()
            tokens = [start_token]  # 无padding
            # 初始化全序列ID列表（包含start_token）
            all_token_ids = self.tokenizer.encode(start_token).ids

            for i in range(gen_seq_len):

                all_token_ids = self.tokenizer.encode("".join(tokens)).ids
                # 模型前向传播
                input_tensor = (
                    torch.tensor(all_token_ids).int().unsqueeze(0).to(self.device)
                )
                out = self.model(input_tensor)

                # 根据padding方向选择logits位置
                if self.padding_side == "right":
                    logits = out[0, len(tokens) - 1, :]
                elif self.padding_side == "left":
                    logits = out[0, -1, :]
                else:
                    raise ValueError("padding_side must be 'right' or 'left'")

                # 频率惩罚（修复后）
                if frequency_penalty != 0:
                    # 使用当前全序列计算频率
                    tokens_tensor = torch.tensor(all_token_ids, device=self.device)
                    unique, counts = torch.unique(tokens_tensor, return_counts=True)

                    # 创建惩罚张量（与logits同设备）
                    penalty = torch.zeros_like(logits)
                    penalty[unique] = counts.float() * frequency_penalty
                    logits = logits - penalty

                # top_k 处理
                if top_k is not None and top_k > 0:
                    # 获取 top_k 以外的 token 索引
                    indices_to_remove = (
                        logits < torch.topk(logits, top_k)[0][..., -1, None]
                    )
                    logits[indices_to_remove] = -float("Inf")

                # 采样下一个token
                probabilities = F.softmax(logits / temperature, dim=-1)
                next_token_id = probabilities.multinomial(num_samples=1).item()

                # 更新序列
                tokens.append(
                    self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                )
                if print_out:
                    print(tokens[-1], end=" ", flush=True)

            return tokens


class DebugTimer:
    def __init__(self, name=None):
        self.start_time = None
        self.name = name

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.timer_start(self.name)
            result = func(*args, **kwargs)
            self.timer_stop()
            return result

        return wrapper

    def timer_start(self, name=None):
        self.start_time = time.perf_counter()
        if name is not None:
            self.name = name
        print(f"{self.name}:", end="")

    def timer_stop(self):
        elapsed_time = round(time.perf_counter() - self.start_time, 4)
        print(f"{elapsed_time}s")


def _format_string(s, length, fill_char=" "):
    """辅助函数：将字符串格式化为指定长度，不足部分用 fill_char 填充"""
    return s.ljust(length, fill_char)


def model_structure(model):
    """打印模型结构信息，包括权重名称、形状和参数数量"""
    print("-" * 90)
    print(
        "|"
        + _format_string("weight name", 31)
        + "|"
        + _format_string("weight shape", 42)
        + "|"
        + _format_string("number", 13)
        + "|"
    )
    print("-" * 90)

    total_params = 0
    type_size = 1  # 如果是浮点数就是4

    for key, param in model.named_parameters():
        # 格式化输出
        formatted_key = _format_string(key, 30)
        shape_str = _format_string(str(param.shape), 40)
        param_count = param.numel()
        formatted_count = _format_string(str(param_count), 10)

        print(f"| {formatted_key} | {shape_str} | {formatted_count} |")
        total_params += param_count

    print("-" * 90)
    print(f"The total number of parameters: {total_params}")
    print(
        f"The parameters of Model {model._get_name()}: {total_params * type_size / 1e6:.4f}M"
    )
    print("-" * 90)
    return total_params


class WarmUpCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr=0, last_step=-1):
        """
        Args:
            optimizer (Optimizer): 包装的优化器。
            total_steps (int): 总的训练步数。
            warmup_steps (int): warm-up 的步数。
            min_lr (float): 最小学习率，默认为 0。
            last_epoch (int): 上一轮的索引，默认为 -1。
        """
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        last_epoch = last_step
        super(WarmUpCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warm-up 阶段：线性增加学习率
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # 余弦退火阶段
            current_step = self.last_epoch - self.warmup_steps
            total_cosine_steps = self.total_steps - self.warmup_steps
            if total_cosine_steps <= 0:
                return [base_lr for base_lr in self.base_lrs]
            return [
                self.min_lr
                + (base_lr - self.min_lr)
                * (
                    1
                    + torch.cos(
                        torch.tensor(current_step / total_cosine_steps * torch.pi)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]


class WarmUpStableDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps,
        stable_steps,
        decay_mode="linear",
        min_lr=0,
        last_step=-1,
    ):
        """
        WarmUpStableDecay学习率调度器
        该调度器包含三个阶段：
        1. 预热阶段：从min_lr线性增长至基础学习率
        2. 稳定阶段：保持基础学习率不变
        3. 衰减阶段：线性衰减至min_lr

        Args:
            optimizer (Optimizer): 包装的优化器
            total_steps (int): 总的训练步数
            warmup_steps (int): 预热步数
            stable_steps (int): 稳定步数
            decay_mode (str): 衰减模式，可选linear, exp
            min_lr (float): 最小学习率，默认为0
            last_epoch (int): 上一步的索引，默认为-1
        """
        assert decay_mode in ["linear", "exp"], "decay_mode 必须是 linear 或 exp"
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.min_lr = min_lr
        self.decay_mode = decay_mode
        last_epoch = last_step
        super(WarmUpStableDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 预热阶段：从min_lr线性增长至基础学习率
            return [
                self.min_lr
                + (base_lr - self.min_lr) * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch < self.warmup_steps + self.stable_steps:
            # 稳定阶段：保持基础学习率不变
            return self.base_lrs
        else:
            # 衰减阶段：从基础学习率指数衰减至min_lr
            # 计算衰减步数
            decay_steps = self.last_epoch - (self.warmup_steps + self.stable_steps)
            # 总衰减步数
            total_decay_steps = self.total_steps - (
                self.warmup_steps + self.stable_steps
            )

            if total_decay_steps <= 0:
                # 如果没有衰减阶段，返回基础学习率
                return self.base_lrs

            # 计算每一步的衰减因子
            lrs = []
            for base_lr in self.base_lrs:
                if self.decay_mode == "linear":
                    # 线性衰减
                    progress = min(decay_steps / total_decay_steps, 1.0)
                    current_lr = base_lr + (self.min_lr - base_lr) * progress
                
                elif self.decay_mode == "exp":
                    # 指数衰减
                    if base_lr != 0 and self.min_lr >= 0 and base_lr > self.min_lr:
                        # 计算衰减率，确保在总衰减步数后达到min_lr
                        # lr = base_lr * decay_rate^(total_decay_steps) = min_lr
                        # 所以 decay_rate = (min_lr / base_lr)^(1 / total_decay_steps)
                        decay_rate = pow(
                            max(self.min_lr / base_lr, 1e-10), 1.0 / total_decay_steps
                        )
                        # 计算当前步骤的学习率
                        current_lr = base_lr * pow(decay_rate, decay_steps)
                    elif self.min_lr >= 0 and base_lr <= self.min_lr:
                        # 如果基础学习率已经小于等于最小学习率，则保持最小学习率
                        current_lr = self.min_lr
                lrs.append(current_lr)

            return lrs


# 使用示例
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 创建一个虚拟的模型和优化器
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_steps = 1000
    # 创建WSD学习率调度器
    scheduler = WarmUpStableDecayLR(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=5,
        stable_steps=500,
        min_lr=1e-4,
        decay_mode="linear",
    )

    # 记录学习率变化
    lrs = []
    for step in range(total_steps):
        lrs.append(scheduler.get_lr()[0])  # 获取第一个参数组的学习率
        scheduler.last_epoch = step  # 模拟调度器内部计数

    # 绘制学习率变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.title("WarmUpStableDecay Learning Rate Schedule")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()

    print(f"第0步学习率: {lrs[0]:.6f}")
    print(f"第50步学习率: {lrs[50]:.6f}")
    print(f"第150步学习率: {lrs[150]:.6f}")
    print(f"第400步学习率: {lrs[400]:.6f}")
