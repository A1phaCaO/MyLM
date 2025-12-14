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
        self.tokenizer.enable_truncation(max_length=self.seq_max_len, direction=padding_side)

    def generate(
    self,
    start_token: str,
    gen_seq_len=30,
    temperature=0.7,
    frequency_penalty=0.1,
    top_k=20,
    print_out=True,
    ):
        with torch.no_grad():
            self.model.eval()
            tokens = [start_token] # 无padding
            # 初始化全序列ID列表（包含start_token）
            all_token_ids = self.tokenizer.encode(start_token).ids

            for i in range(gen_seq_len):
                
                all_token_ids = self.tokenizer.encode(''.join(tokens)).ids
                # 模型前向传播
                input_tensor = torch.tensor(all_token_ids).int().unsqueeze(0).to(self.device)
                out = self.model(input_tensor)
                
                # 根据padding方向选择logits位置
                if self.padding_side == "right":
                    logits = out[0, len(tokens)-1, :]  
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
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                
                # 采样下一个token
                probabilities = F.softmax(logits / temperature, dim=-1)
                next_token_id = probabilities.multinomial(num_samples=1).item()
                
                # 更新序列
                tokens.append(self.tokenizer.decode([next_token_id], skip_special_tokens=False))
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
    def __init__(self, optimizer, total_epochs, warmup_epochs, min_lr=0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): 包装的优化器。
            total_epochs (int): 总的训练轮数。
            warmup_epochs (int): warm-up 的轮数。
            min_lr (float): 最小学习率，默认为 0。
            last_epoch (int): 上一轮的索引，默认为 -1。
        """
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super(WarmUpCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warm-up 阶段：线性增加学习率
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # 余弦退火阶段
            current_epoch = self.last_epoch - self.warmup_epochs
            total_cosine_epochs = self.total_epochs - self.warmup_epochs
            return [
                self.min_lr
                + (base_lr - self.min_lr)
                * (
                    1
                    + torch.cos(
                        torch.tensor(current_epoch / total_cosine_epochs * torch.pi)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]



