from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tokenizers
import math

from models import MoEFFN


class TextGenerator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: tokenizers.Tokenizer,
        device,
        padding_side="none",  # generate() 的默认 padding 方向（调用方可覆盖）
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.padding_side = padding_side
        if isinstance(model, nn.DataParallel):
            print("该模型使用了DataParallel")
            self.model = model.module
        else:
            self.model = model
        self.seq_max_len = self.model.args.seq_max_len
        # 不再变异共享 tokenizer（旧版 enable_padding/enable_truncation 会污染
        # 同一实例的后续 encode 调用；SFTTextDataset 为此不得不独立加载 tokenizer）。
        # 截断在 generate() 中手动处理。

    def generate(
        self,
        start_token: str,
        gen_seq_len=30,
        temperature=0.7,
        frequency_penalty=1.5,
        top_k=20,
        top_p=None,
        repetition_penalty=1.0,
        print_out=True,
        eos_id=None,
        padding_side=None,
    ):
        """自回归生成。

        vs 旧版修复：
        - 维护 token ID 列表，不再每步 re-encode 全串（避免 BBPE 跨边界重切分
          导致 token 序列漂移）
        - 整体 decode（增量打印），避免逐 token decode 产生 BBPE 残缺字节 �
        - bf16 autocast 对齐训练精度
        - 可选 eos_id 停止（SFT 传 <|im_end|> id）

        Args:
            padding_side: 默认取构造时传入的值（"none"）；"none" = 仅尾部截断、
                （纯 causal）；"left" = 左 pad 到 seq_max_len；"right" = 右 pad
                到 seq_max_len。后两者显式构造布尔 padding_mask 传入模型
                （key/query 双向屏蔽，pad 位置不参与注意力）。显式 mask 按
                pad 位置构造、而非 (id != pad_id) 推断，避免真实 EOS(id=0)
                被误判为 pad。右 pad 时 logits 取最后一个真实 token 位置。

        Returns:
            str: 完整解码文本（旧版返回 list[str]，调用方 "".join() 对 str 同样兼容）
        """
        with torch.no_grad():
            if padding_side is None:
                padding_side = self.padding_side
            self.model.eval()
            token_ids = list(self.tokenizer.encode(start_token).ids)
            pad_id = self.model.args.pad_id
            prev_decoded = self.tokenizer.decode(
                token_ids, skip_special_tokens=False)

            for i in range(gen_seq_len):
                # 截断到 seq_max_len（保留尾部 = 最新上下文）
                context = token_ids[-self.seq_max_len:]
                n_ctx = len(context)

                if padding_side == "none" or n_ctx >= self.seq_max_len:
                    # 无 pad：仅 causal mask（推理无 padding）
                    input_tensor = torch.tensor(
                        [context], dtype=torch.long, device=self.device
                    )
                    padding_mask = None
                    logits_pos = -1
                else:
                    pad_len = self.seq_max_len - n_ctx
                    if padding_side == "left":
                        ids = [pad_id] * pad_len + context
                        valid = [False] * pad_len + [True] * n_ctx
                    elif padding_side == "right":
                        ids = context + [pad_id] * pad_len
                        valid = [True] * n_ctx + [False] * pad_len
                    else:
                        raise ValueError(
                            f"padding_side 必须是 'none'/'left'/'right'，got {padding_side}"
                        )
                    input_tensor = torch.tensor(
                        [ids], dtype=torch.long, device=self.device
                    )
                    padding_mask = torch.tensor(
                        [valid], dtype=torch.bool, device=self.device
                    )
                    # 右 pad 时最后一个位置是 pad，logits 取最后一个真实 token
                    logits_pos = n_ctx - 1 if padding_side == "right" else -1

                with torch.autocast(
                    str(self.device), enabled=True, dtype=torch.bfloat16
                ):
                    out = self.model(input_tensor, padding_mask=padding_mask)
                logits = out[0, logits_pos, :].float()

                # 经典重复惩罚（repetition_penalty，HuggingFace 风格）：
                # 对上下文里出现过的 token 直接缩放其 logits，抑制已经说过的词。
                # 与 frequency_penalty 互补——后者按出现频次线性减分，
                # 前者是乘性且对频次不敏感，更稳。
                if repetition_penalty != 1.0:
                    for prev_id in set(context):
                        if logits[prev_id] > 0:
                            logits[prev_id] /= repetition_penalty
                        else:
                            logits[prev_id] *= repetition_penalty

                # 频率惩罚（按相对频率归一化：penalty = 出现频率 × 系数。
                # 旧实现 penalty = counts × 系数，长上下文下 counts 随序列增长
                # 累积，高频词 logits 被减几十导致完全被抑制、输出退化，
                # 而短上下文下惩罚又几乎为 0，效果随输入长度剧烈波动）
                if frequency_penalty != 0:
                    ctx_tensor = torch.tensor(context, device=self.device)
                    unique, counts = torch.unique(
                        ctx_tensor, return_counts=True)
                    penalty = torch.zeros_like(logits)
                    penalty[unique] = (
                        counts.float() * frequency_penalty / max(len(context), 1)
                    )
                    logits = logits - penalty

                # top_p 核采样（nucleus）：按概率从高到低累加，截掉累计超过
                # top_p 的尾部 token；至少保留概率最高的 1 个，避免全 -inf。
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_to_remove = cumulative_probs > top_p
                    sorted_to_remove[0] = False
                    logits[sorted_indices[sorted_to_remove]] = -float("Inf")

                # top_k
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    indices_to_remove = (
                        logits < torch.topk(logits, k)[0][..., -1, None]
                    )
                    logits[indices_to_remove] = -float("Inf")

                # 采样
                probabilities = F.softmax(logits / temperature, dim=-1)
                next_token_id = probabilities.multinomial(
                    num_samples=1).item()

                # EOS 停止
                if eos_id is not None and next_token_id == eos_id:
                    break

                token_ids.append(next_token_id)

                if print_out:
                    # 增量 decode：整体解码后取 diff，正确处理多字节 BBPE token
                    new_decoded = self.tokenizer.decode(
                        token_ids, skip_special_tokens=False)
                    print(new_decoded[len(prev_decoded):], end="", flush=True)
                    prev_decoded = new_decoded

            if print_out:
                print()

            return self.tokenizer.decode(token_ids, skip_special_tokens=False)


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


class MoEStatsCollector:
    """通过 forward hook 采集 FixedCap+DS-bias MoE 的运行时负载统计，零侵入。

    用法:
        collector = MoEStatsCollector(n_experts, n_experts_per_tok)
        collector.register(model)      # 在 DataParallel 包裹后调用
        collector.set_enabled(True)    # 需要记录的 forward 前开启
        ...model(inputs)...
        collector.set_enabled(False)
        for layer_idx, stats in collector.stats(): ...

    统计基于 DS-bias 修正后的 logits（与实际 topk 路由完全一致），
    只保留 3 个关键信号:
    - expert_load: 各专家实际 token 占比 (n_experts,)
    - drop_rate:   FixedCap 超容量丢 token 率 (每专家 M 槽装不下的比例)
    - router_entropy: 归一化路由熵 (1.0=完全均匀)
    """

    def __init__(self, n_experts: int, n_experts_per_tok: int):
        self.n_experts = n_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.enabled = False
        self.layer_stats = {}  # layer_idx -> stats dict
        self._handles = []
        self._mlps = {}        # layer_idx -> MoEFFN (读取 last_drop/expert_bias)

    def register(self, model: nn.Module):
        for layer_idx, block in enumerate(model.blocks):
            mlp = getattr(block, "mlp", None)
            if not isinstance(mlp, MoEFFN):
                continue
            self._mlps[layer_idx] = mlp
            handle = mlp.router.register_forward_hook(
                lambda mod, args, output, idx=layer_idx: self._hook(idx, output)
            )
            self._handles.append(handle)
        return self

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def stats(self):
        """返回 [(layer_idx, stats), ...]，stats 字段见 _compute"""
        return sorted(self.layer_stats.items())

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @torch._dynamo.disable
    def _hook(self, layer_idx: int, router_logits):
        if not self.enabled:
            return
        mlp = self._mlps.get(layer_idx)
        latest_drop = float(getattr(mlp, "last_drop", 0.0)) if mlp is not None else 0.0
        self.layer_stats[layer_idx] = self._compute(router_logits, latest_drop, mlp)

    @torch.no_grad()
    def _compute(self, router_logits, latest_drop=0.0, mlp=None):
        """由 router logits 计算负载均衡指标（与 MoEFFN 实际路由同一份 logits）
        drop_rate 取该层最近一次 forward 的 last_drop (hook 触发时上一轮的值,
        步级观测不影响趋势)。
        """
        N = self.n_experts
        K = self.n_experts_per_tok
        logits = router_logits.detach().float()  # [B, S, N]
        # DS bias 修正后与实际 topk 路由完全一致
        if mlp is not None and hasattr(mlp, "expert_bias"):
            logits = logits + mlp.expert_bias.float()
        _, topk_indices = torch.topk(logits, K, dim=-1)
        counts = torch.bincount(topk_indices.view(-1), minlength=N)
        total = counts.sum().clamp(min=1)
        f = counts.float() / total
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(-1)
        return {
            "expert_load": f.cpu(),
            "drop_rate": latest_drop,
            "router_entropy": (entropy.mean() / math.log(N)).item(),
        }


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
