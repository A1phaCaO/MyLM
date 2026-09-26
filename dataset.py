import torch
import collections
from utils import DebugTimer
from tqdm import tqdm
import tokenizers
import json
import sys
import random
import numpy as np
from typing import Optional

# import tracemalloc


class PretrainTextDataset(torch.utils.data.Dataset):
    """
    流式文本数据集，避免将整个数据集加载到内存中
    只存储文件路径和行位置信息，在需要时才读取特定行
    返回格式：(input, output, mask)，其中mask为loss mask，包括pad mask
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: tokenizers.Tokenizer,
        seq_max_len: int = 192,
        downsample: int = 1,
        batch: bool = None,  # 兼容性参数
        re_tokenize: bool = False,
        padding_side: str = "right",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_max_len = seq_max_len
        self.re_tokenize = re_tokenize
        self.padding_side = padding_side

        # 构建行索引，只存储行的偏移位置而不是内容
        self.line_offsets = []
        self._build_line_index(downsample)

    def _build_line_index(self, downsample: int):
        """构建行偏移索引，避免加载整个文件"""
        with open(self.data_dir, "rb") as f:
            offset = 0
            line_count = 0
            while True:
                self.line_offsets.append(offset)
                line = f.readline()
                if not line:
                    break

                offset += len(line)
                line_count += 1
            self.line_offsets = random.sample(
                self.line_offsets, k=int(len(self.line_offsets) * downsample)
            )

    def pad_seq(
        self,
        seq: list[int],
        max_len: int,
        truncation=True,
        padding_value=0,
        padding_side="left",
    ):
        """
        对序列进行填充
        Args:
            seq: 序列
            max_len: 最大长度
            padding_value: 填充值
            padding_side: 填充方向
        Returns:
            填充后的序列
        """
        # 截断
        if truncation:
            if padding_side == "right":
                seq = seq[:max_len]
            elif padding_side == "left":
                seq = seq[-max_len:]
            else:
                raise ValueError("padding_side must be 'left' or 'right'")

        # 填充
        if len(seq) < max_len:
            if padding_side == "left":
                seq = [padding_value] * (max_len - len(seq)) + seq
            elif padding_side == "right":
                seq = seq + [padding_value] * (max_len - len(seq))
            else:
                raise ValueError("padding_side must be 'left' or 'right'")

        return seq

    def _create_loss_mask(
        self, padded_seq: list[int], padding_value: int = 0
    ) -> list[int]:
        """
        创建loss mask，用于标识需要计算loss的位置
        Args:
            padded_seq: 填充后的序列
            padding_value: padding的值（通常为0）
        Returns:
            mask: 有效位置为1，padding位置为0
        """
        return [0 if token == padding_value else 1 for token in padded_seq]

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, index):
        """
        根据索引获取数据样本，只在需要时读取特定行
        返回格式：(input, output, mask)
        """
        # 根据索引定位并读取特定行
        with open(self.data_dir, "r", encoding="utf-8") as f:
            f.seek(self.line_offsets[index])
            line = f.readline().strip()

        # 进行tokenization
        if self.re_tokenize:
            # 如果需要重新分词，直接使用原始字符串
            raw = self.tokenizer.encode(line).ids
        else:
            # 如果使用预分词数据，需要先将字符串分割成列表
            raw = self.tokenizer.encode(line.split(" "), is_pretokenized=True).ids

        raw = self.pad_seq(
            raw,
            max_len=self.seq_max_len,
            truncation=True,
            padding_value=0,
            padding_side=self.padding_side,
        )

        # 将列表转换为tensor
        raw_tensor = torch.tensor(raw, dtype=torch.long)

        inputs = raw_tensor[:-1].contiguous()
        outputs = raw_tensor[1:].contiguous()

        # 生成loss mask（对应output位置）
        mask = self._create_loss_mask(outputs.cpu().tolist())
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return (inputs, outputs, mask_tensor)


class PretrainTokenIDDataset(torch.utils.data.Dataset):
    """
    二进制 token ID 数据集（配合 generate_dataset_v3.py 产物）。
    直接从 .npy (2D uint16, shape=(N, SENTENCE_MAXLEN)) 加载，mmap 流式访问。
    无需 tokenizer，__getitem__ 不做任何编码。
    返回格式：(input, output, mask)，mask 基于 pad_value=0。

    约定：存储长度 = seq_max_len + 1（因 __getitem__ 做 [:-1]/[1:] 切片）。
    若存储长度 > seq_max_len + 1，truncate；若 < seq_max_len + 1，pad。
    推荐：生成时 SENTENCE_MAXLEN = 训练 seq_max_len + 1，loader 无需 pad。
    """

    def __init__(
        self,
        data_dir: str,
        seq_max_len: int = 192,
        downsample: int = 1,
        padding_side: str = "right",
        pad_value: int = 0,
        dtype=np.uint16,
        shuffle_seed: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_max_len = seq_max_len
        self.padding_side = padding_side
        self.pad_value = pad_value
        self.dtype = dtype
        # mmap 流式加载：不会一次性把整个文件读进内存
        self.data = np.load(data_dir, mmap_mode="r")
        self.n_samples = self.data.shape[0]
        self.storage_len = self.data.shape[1]   # SENTENCE_MAXLEN，通常 = seq_max_len + 1
        # 下采样：随机抽取索引（与 PretrainTextDataset 行为对齐）
        if downsample != 1:
            k = int(self.n_samples * downsample)
            self.indices = random.sample(range(self.n_samples), k)
        else:
            self.indices = list(range(self.n_samples))
        # 固定种子 shuffle：perm 由种子完全决定，跨进程可复现，
        # 断点续训时只需按已消费位置继续（配合 DataLoader shuffle=False），
        # 不会再像 RandomSampler 那样生成全新排列导致已训数据重复。
        self.perm = None
        self.shuffle_seed = shuffle_seed
        if shuffle_seed is not None:
            self.set_permute_seed(shuffle_seed)

    def set_permute_seed(self, seed: int):
        """重设 shuffle 种子（每个 epoch 换一个新种子即可得到不同排列）。
        用独立 RandomState，不消耗训练全局 RNG。
        """
        self.shuffle_seed = seed
        rng = np.random.RandomState(int(seed))
        self.perm = rng.permutation(self.n_samples).astype(np.int64)
        return self

    def __getstate__(self):
        """序列化时丢弃 memmap：numpy 2.x 的 memmap 已无自定义 __reduce__，
        pickle 会把整个文件内容拷成 bytes（730MB 级），经 Windows spawn 管道
        传输会触发 OSError: [Errno 22]。其余属性（indices/perm 等）照常序列化。
        """
        state = self.__dict__.copy()
        state.pop("data", None)
        return state

    def __setstate__(self, state):
        """反序列化时按 data_dir 重新打开 memmap（mmap 惰性映射，开销可忽略）。"""
        self.__dict__.update(state)
        self.data = np.load(self.data_dir, mmap_mode="r")

    def pad_seq(
        self,
        seq: np.ndarray,
        max_len: int,
        padding_value: int = 0,
        padding_side: str = "right",
    ) -> np.ndarray:
        """对序列进行截断+填充（与 PretrainTextDataset.pad_seq 逻辑一致）"""
        if padding_side == "right":
            seq = seq[:max_len]
            if len(seq) < max_len:
                seq = np.concatenate(
                    [seq, np.full(max_len - len(seq), padding_value, dtype=seq.dtype)]
                )
        elif padding_side == "left":
            seq = seq[-max_len:]
            if len(seq) < max_len:
                seq = np.concatenate(
                    [np.full(max_len - len(seq), padding_value, dtype=seq.dtype), seq]
                )
        else:
            raise ValueError("padding_side must be 'left' or 'right'")
        return seq

    def _create_loss_mask(self, padded_seq, padding_value: int = 0) -> list[int]:
        """生成 loss mask：有效位置为 1，padding 位置为 0"""
        return [0 if int(t) == padding_value else 1 for t in padded_seq]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        根据索引获取数据样本，只在需要时读取对应行（mmap）
        返回格式：(input, output, mask)
        """
        # 从 mmap 拷出一行（转为本机 dtype）
        src = self.indices[index]
        if self.perm is not None:
            src = int(self.perm[src])
        row = np.array(self.data[src], dtype=self.dtype)
        # 截断/填充到 seq_max_len + 1（因为后续 [:-1]/[1:] 切片会少 1）
        target_len = self.seq_max_len + 1
        raw = self.pad_seq(
            row,
            max_len=target_len,
            padding_value=self.pad_value,
            padding_side=self.padding_side,
        )
        raw_tensor = torch.tensor(raw, dtype=torch.long)

        inputs = raw_tensor[:-1].contiguous()   # length = seq_max_len
        outputs = raw_tensor[1:].contiguous()    # length = seq_max_len

        # 生成 loss mask（对应 output 位置）
        mask = self._create_loss_mask(outputs.cpu().tolist(), padding_value=self.pad_value)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return (inputs, outputs, mask_tensor)


class SFTTextDataset(torch.utils.data.Dataset):
    """
    SFT（指令微调）数据集：基于 PretrainTextDataset 流式索引改造，
    接口对齐 PretrainTokenIDDataset（返回 inputs/outputs/mask 三元组）。

    数据格式（未预分词 ChatML，两种写法都支持）：
    - 单行完整对话："<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>"
    - 消息级真实换行（一行一条消息），自动按对话结构合并为样本
    样本切分规则：上个 assistant 回合以 <|im_end|> 闭合后，出现 <|im_start|> 即
    开始新样本。注意：文本流没有对话边界标记，消息级换行的多轮对话会在第一个
    assistant 回答后拆成多个单轮样本（对 SFT 训练无害，每个片段仍是完整对话）；
    单行完整对话不受影响，整行作为一个样本。

    集成 SFT loss mask 生成（Qwen 策略，对齐旧 SFT 脚本语义）：
    - assistant 回答正文 = 1（计 loss），不含角色标记与 <|im_end|>
    - 特殊 token（<|im_start|>/<|im_end|>/<|beginoftext|>）= 1（可关闭）
    - user/system 内容、角色标记、pad = 0
    - pad 复用 <|endoftext|>（id=0，对齐 Qwen：EOS=pad 同一 token；
      tokenizer 里的 <|pad|>(id=2) 是保留字段，不用于训练）；
      因此行尾 <|endoftext|> 与 pad 位置同样被 mask 强制为 0（不计 loss）
    - 固定种子 shuffle：续训按已消费位置继续，数据不重复

    注意：tokenizer 以路径传入并在内部独立加载，避免训练时
    TextGenerator 修改同一实例 padding/truncation 配置导致编码异常。
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        seq_max_len: int = 256,
        downsample: float = 1.0,
        padding_side: str = "right",
        pad_value: int = 0,
        shuffle_seed: Optional[int] = None,
        # 字面 \n（backslash+n）：SFT 数据文件用字面 \n 做消息分隔，
        # tokenizer 的 ByteLevel 预分词器会把真实换行(0x0A)归为 <|unk|>(id=3)，
        # 因此必须用字面 \n 才能正确匹配 assistant 前缀并保留有意义的 token
        assistant_prefix: str = "<|im_start|>assistant\\n",
        mark_special_tokens: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
        self.seq_max_len = seq_max_len
        self.padding_side = padding_side
        self.pad_value = pad_value
        self.mark_special_tokens = mark_special_tokens

        # 预计算特殊 token id 序列（mask 生成时做子序列匹配）
        self.im_start_ids = np.asarray(
            self.tokenizer.encode("<|im_start|>").ids, dtype=np.int64)
        self.im_end_ids = np.asarray(
            self.tokenizer.encode("<|im_end|>").ids, dtype=np.int64)
        eot_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.eot_ids = np.asarray([eot_id if eot_id is not None else 0],
                                  dtype=np.int64)
        bot_id = self.tokenizer.token_to_id("<|beginoftext|>")
        self.bot_ids = (np.asarray([bot_id], dtype=np.int64)
                        if bot_id is not None else None)
        self.assistant_prefix_ids = np.asarray(
            self.tokenizer.encode(assistant_prefix).ids, dtype=np.int64)

        # 构建样本索引（chatml 结构感知，只存字节偏移不读内容）
        self.samples = []
        self._build_sample_index(downsample)
        self.indices = list(range(len(self.samples)))
        # 固定种子 shuffle（与 PretrainTokenIDDataset 相同机制）
        self.perm = None
        self.shuffle_seed = None
        if shuffle_seed is not None:
            self.set_permute_seed(shuffle_seed)

    def _build_sample_index(self, downsample: float):
        """按 ChatML 结构切分对话样本。downsample 语义（对齐 generate_dataset_v3）：
        - (0, 1): 降采样，随机抽取该百分比的行
        - 1     : 不采样，全量
        - > 1   : 重复（repeat）int(downsample) 次
        切分规则（状态机）：上一个 assistant 回合 <|im_end|> 闭合后，
        一旦出现 <|im_start|> 即开始新样本；否则内容并入当前样本
        （兼容单行完整对话与消息级真实换行两种格式）。
        """
        samples = []
        last_pos = 0
        cur_start = -1
        cur_ended = False  # 当前样本最近一个 assistant 回合是否已闭合
        last_role = None   # 样本内最近一条消息的角色（跨行有效）
        # 二进制模式：字节偏移跨 open 安全（文本模式 tell/seek 是代理值,重开文件会错位）
        start_mark = "<|im_start|>".encode("utf-8")
        assistant_mark = "<|im_start|>assistant".encode("utf-8")
        user_mark = "<|im_start|>user".encode("utf-8")
        system_mark = "<|im_start|>system".encode("utf-8")
        with open(self.data_dir, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                last_pos = f.tell()
                is_start = start_mark in line
                # 行内消息闭合状态由行内最后一条消息角色（last_role）驱动，
                # 与行是否以 <|im_end|> 结尾无关：
                # 截断产生的未闭合行（assistant 头无 im_end）也视为闭合，若并入
                # 后续行会拼成超长样本，左截断会把开头 <|im_start|> 切掉、结构残破；
                # 独立成样本时 _build_sft_mask 对无 im_end 的回答有"算到序列末尾"兜底。
                if cur_start < 0:
                    cur_start = pos
                elif is_start and cur_ended:
                    # 上一个 assistant 回合已闭合，本行开启新消息 → 新对话样本
                    samples.append((cur_start, pos))
                    cur_start = pos
                    cur_ended = False
                # 行内角色：取行内最后一个 <|im_start|> 之后的消息角色（覆盖跨行状态）
                if is_start:
                    seg = line[line.rfind(start_mark):]
                    if seg.startswith(assistant_mark):
                        last_role = "assistant"
                    elif seg.startswith(user_mark):
                        last_role = "user"
                    elif seg.startswith(system_mark):
                        last_role = "system"
                    else:
                        last_role = "other"
                # 最近一条消息是 assistant 即视为回合已闭合：
                # 下一行出现任何 <|im_start|> 消息（包括 user 行）都会触发上面的切分，
                # 因此消息级换行的多轮对话在每个 assistant 回合后即拆成单轮样本。
                # 该行为与类文档字符串一致，是数据无对话边界标记下的已知妥协
                # （切分后 cur_ended 复位，不会出现"合并多轮"）。
                if last_role == "assistant":
                    cur_ended = True
        if cur_start >= 0:
            samples.append((cur_start, last_pos))

        if downsample < 1:
            samples = random.sample(samples, max(1, int(len(samples) * downsample)))
        elif downsample > 1:
            samples = samples * int(downsample)
        self.samples = samples

        # ---- 静默失效保护：确认数据格式与 SFT mask 约定一致 ----
        # SFT mask 依赖「字面反斜杠-n」匹配 assistant 前缀；若数据误用真实换行(0x0A)，
        # tokenizer 的 ByteLevel 会将其归为 <|unk|>(id=3)，前缀永远匹配不上 →
        # 每个样本回答正文 mask 全 0 → 模型只训到结构特殊 token（灾难性且静默）。
        # 构建期即拦下，避免 loss 照降、人看不出。
        prefix = self.assistant_prefix_ids
        if prefix is None or len(prefix) == 0:
            raise ValueError(
                "assistant_prefix 编码为空，无法生成 SFT mask；"
                "请检查 tokenizer 是否注册了 <|im_start|> 等特殊 token"
            )
        im_start_id = self.tokenizer.token_to_id("<|im_start|>")
        if im_start_id is None or int(im_start_id) not in prefix.tolist():
            raise ValueError(
                "assistant_prefix 未包含 <|im_start|> token，请检查 tokenizer 配置"
            )
        K = min(50, len(self.samples))
        if K == 0:
            raise ValueError("数据集为空（0 个样本），请检查数据文件与切分规则")
        hit = 0
        for si in range(K):
            s, e = self.samples[si]
            with open(self.data_dir, "rb") as f:
                f.seek(s)
                txt = f.read(e - s).decode("utf-8", errors="ignore")
            ids = np.asarray(self.tokenizer.encode(txt).ids, dtype=np.int64)
            if self._find_subseq(ids, prefix).size > 0:
                hit += 1
        if hit == 0:
            raise RuntimeError(
                f"SFT mask 自检失败：抽检 {K} 个样本中 0 个能匹配到 assistant 前缀 "
                f"{prefix.tolist()}。极可能是数据用了真实换行而非字面 '\\n'。"
                f"请确认 {self.data_dir} 的消息分隔符为字面反斜杠-n。"
            )
        if hit < K:
            print(
                f"[SFT 警告] 抽检 {K} 个样本仅 {hit} 个命中 assistant 前缀，"
                f"其余可能无 assistant 回合或格式不规整。"
            )

    def set_permute_seed(self, seed: int):
        """重设 shuffle 种子（每个 epoch 换新种子得到不同排列）。
        用独立 RandomState，不消耗训练全局 RNG。
        """
        self.shuffle_seed = seed
        rng = np.random.RandomState(int(seed))
        self.perm = rng.permutation(len(self.indices)).astype(np.int64)
        return self

    def pad_seq(
        self,
        seq: np.ndarray,
        max_len: int,
        padding_value: int = 0,
        padding_side: str = "right",
    ) -> np.ndarray:
        """截断+填充（与 PretrainTokenIDDataset.pad_seq 逻辑一致）"""
        if padding_side == "right":
            seq = seq[:max_len]
            if len(seq) < max_len:
                seq = np.concatenate(
                    [seq, np.full(max_len - len(seq), padding_value,
                                  dtype=seq.dtype)])
        elif padding_side == "left":
            seq = seq[-max_len:]
            if len(seq) < max_len:
                seq = np.concatenate(
                    [np.full(max_len - len(seq), padding_value,
                             dtype=seq.dtype), seq])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")
        return seq

    def _pad_pair(
        self,
        ids: np.ndarray,
        mask: np.ndarray,
    ):
        """ids 与其 loss mask 同步截断+填充（长度=seq_max_len+1，后续 [:-1]/[1:] 切片少 1）。
        mask 先于截断生成，保证保留尾部（assistant 回答）时 loss 标记不丢失。

        SFT 截断策略：无论 pad 方向，始终左截断（保留尾部 = assistant 回答）。
          - right pad：左截断 + 右填充 → pad 在尾部，causal mask 天然隔离，
            不再依赖 seq_mask 的零假设（attn_bias 可学习偏置会破坏该假设）。
          - left  pad：左截断 + 左填充（旧行为，保留兼容）。
        """
        max_len = self.seq_max_len + 1
        pad_v = self.pad_value
        if self.padding_side == "right":
            # 左截断：保留尾部（assistant 回答）；右填充：pad 加在尾部
            ids = ids[-max_len:]
            mask = mask[-max_len:]
            if len(ids) < max_len:
                n = max_len - len(ids)
                ids = np.concatenate([ids, np.full(n, pad_v, dtype=ids.dtype)])
                mask = np.concatenate([mask, np.zeros(n, dtype=mask.dtype)])
        elif self.padding_side == "left":
            ids = ids[-max_len:]
            mask = mask[-max_len:]
            if len(ids) < max_len:
                n = max_len - len(ids)
                ids = np.concatenate([np.full(n, pad_v, dtype=ids.dtype), ids])
                mask = np.concatenate([np.zeros(n, dtype=mask.dtype), mask])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")
        return ids, mask

    @staticmethod
    def _find_subseq(seq: np.ndarray, sub: np.ndarray) -> np.ndarray:
        """返回 sub 在 seq 中的所有匹配起始位置（np.int64 数组）"""
        n = len(sub)
        if n == 0 or len(seq) < n:
            return np.array([], dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(seq, n)
        return np.flatnonzero((windows == sub).all(axis=1))

    def _build_sft_mask(self, raw_ids: np.ndarray) -> np.ndarray:
        """生成 SFT loss mask（float32，长度=len(raw_ids)，对应 output 位置）：
        - assistant 回答正文（角色标记之后、<|im_end|> 之前）= 1
        - 特殊 token（im_start/im_end/endoftext/beginoftext）= 1（可关闭）
        - user/system 内容、角色标记、pad = 0
        多轮对话：每个 assistant 段分别处理；
        无 <|im_end|> 时以序列末尾兜底（末尾可能是被截断的回答）。
        """
        n = len(raw_ids)
        mask = np.zeros(n, dtype=np.float32)
        seq = np.asarray(raw_ids, dtype=np.int64)

        # 1) 特殊 token 不 mask（Qwen 策略）
        if self.mark_special_tokens:
            for sub in (self.im_start_ids, self.im_end_ids, self.eot_ids):
                for p in self._find_subseq(seq, sub):
                    mask[p:p + len(sub)] = 1.0
            if self.bot_ids is not None:
                for p in self._find_subseq(seq, self.bot_ids):
                    mask[p:p + len(self.bot_ids)] = 1.0

        # 2) assistant 回答正文置 1（不覆盖特殊 token 已置 1 的位置，两者区域不重叠）
        pre_len = len(self.assistant_prefix_ids)
        for s in self._find_subseq(seq, self.assistant_prefix_ids):
            content_start = min(s + pre_len, n)
            # 找该轮最近的 <|im_end|>（只在该轮剩余部分找，避免跨轮）
            ends = self._find_subseq(seq[content_start:], self.im_end_ids)
            content_end = n if len(ends) == 0 else int(content_start + ends[0])
            mask[content_start:content_end] = 1.0

        # 3) 兜底：pad 区强制 0。pad 复用 <|endoftext|>（id=0，对齐 Qwen），
        #    因此数据中行尾的 <|endoftext|> 与 pad 位置统一不计 loss
        mask[seq == self.pad_value] = 0.0
        return mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        读取样本段 → tokenize → pad(seq_max_len+1) → inputs/outputs + SFT mask
        返回 (inputs, outputs, mask)，接口对齐 PretrainTokenIDDataset。
        """
        src = self.indices[index]
        if self.perm is not None:
            src = int(self.perm[src])
        start, end = self.samples[src]
        # 二进制模式按字节偏移读取（与索引构建一致，跨 open 安全）
        with open(self.data_dir, "rb") as f:
            f.seek(start)
            text = f.read(end - start).decode("utf-8", errors="ignore").strip()
        if not text:
            # 空样本兜底：用 assistant 前缀而非 <|endoftext|>（id=0=pad，
            # _build_sft_mask 末尾 mask[seq==pad]=0 会把整个 mask 清零，
            # 导致样本静默零贡献）
            # 必须用字面 \n（与 assistant_prefix 一致）：真实换行(0x0A)会被
            # tokenizer 编码为 <|unk|>(id=3)，前缀匹配不上、兜底 mask 失效
            text = "<|im_start|>assistant\\n"
        ids = self.tokenizer.encode(text).ids

        # 在完整序列上生成 mask，再与 ids 同步截断+pad：
        # 否则左截断切掉 assistant 前缀后，回答正文会失去 loss 标记
        ids_arr = np.asarray(ids, dtype=np.int64)
        mask_arr = self._build_sft_mask(ids_arr)
        raw, mask_raw = self._pad_pair(ids_arr, mask_arr)
        raw_tensor = torch.tensor(raw, dtype=torch.long)

        inputs = raw_tensor[:-1].contiguous()
        outputs = raw_tensor[1:].contiguous()

        # 在完整 raw 上生成 mask，切片取 outputs 对应位置
        mask = torch.tensor(mask_raw[1:], dtype=torch.float32)
        return (inputs, outputs, mask)


class RuntimeTextDatasetV4(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: tokenizers.Tokenizer,
        seq_max_len: int = 192,
        downsample: int = 1,
        re_tokenize: bool = False,
        batch: bool = None,  # 兼容性参数
        padding_side: str = "right",
    ):
        """初始化模型。

        Args:
            data_dir (str): 数据目录的路径。
            seq_max_len (int): 序列的最大长度。
            vocab_size (int): 词汇表的大小。
            downsample (int): 数据下采样率，控制是否对数据进行下采样。
            batch (bool): 是否使用batch流程，速度提升但无进度条

        Returns:
            None
        """
        super().__init__()
        self.batch_pipeline = batch
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.seq_max_len = seq_max_len
        self.load_and_preprocess_data(
            data_dir, downsample
        )  # 加载并预处理数据目录中的数据
        self.re_tokenize = re_tokenize
        print(sys.getsizeof(self.raw_data))

    def pad_seq(
        self,
        seq: list[int],
        max_len: int,
        truncation=True,
        padding_value=0,
        padding_side="left",
    ):
        """
        对序列进行填充
        Args:
            seq: 序列
            max_len: 最大长度
            padding_value: 填充值
            padding_side: 填充方向
        Returns:
            填充后的序列
        """
        # 截断
        if truncation:
            if padding_side == "right":
                seq = seq[:max_len]
            elif padding_side == "left":
                seq = seq[-max_len:]
            else:
                raise ValueError("padding_side must be 'left' or 'right'")

        # 填充
        if len(seq) < max_len:
            if padding_side == "left":
                seq = [padding_value] * (max_len - len(seq)) + seq
            elif padding_side == "right":
                seq = seq + [padding_value] * (max_len - len(seq))
            else:
                raise ValueError("padding_side must be 'left' or 'right'")

        return seq

    @DebugTimer("加载并生成训练数据")
    def load_and_preprocess_data(self, data_dir, downsample):
        """
        加载并预处理数据目录中的文本数据，将其编码为token序列，并生成输入输出对

        参数：
            data_dir (str): 文本数据文件路径，每行一个样本
            downsample (float|bool): 下采样率，数值类型时按间隔采样，布尔值时控制是否启用采样


        返回值：
            None: 处理结果存储在类的input_data和output_data属性中
        """
        # 逐行读取文件并按指定采样率抽取数据行，避免一次性加载整个文件到内存
        self.input_data = []
        self.output_data = []
        self.raw_data = []
        self.word2idx = self.tokenizer.get_vocab()

        line_count = 0

        with open(data_dir, encoding="utf-8") as f:
            data_len = sum(1 for line in f)
            f.seek(0)
            for line in tqdm(f, total=data_len):
                if line_count % downsample == 0:  # 实现下采样功能
                    # 单条处理模式
                    line_ = line.strip()
                    self.raw_data.append(line_)
                line_count += 1
        # self.raw_data = torch.tensor(self.raw_data, dtype=torch.long)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        """
        根据索引获取数据样本
        参数:
            index (int): 索引值
        返回值:
            tuple: 输入数据和输出数据的元组，以tensor形式表示
        """
        # 根据是否需要重新分词来决定如何处理数据
        if self.re_tokenize:
            # 如果需要重新分词，直接使用原始字符串
            raw = self.tokenizer.encode(self.raw_data[index]).ids
        else:
            # 如果使用预分词数据，需要先将字符串分割成列表
            raw = self.tokenizer.encode(
                self.raw_data[index].split(" "), is_pretokenized=True
            ).ids

        raw = self.pad_seq(
            raw,
            max_len=self.seq_max_len,
            truncation=True,
            padding_value=0,
            padding_side=self.padding_side,
        )

        # 将列表转换为tensor
        raw_tensor = torch.tensor(raw, dtype=torch.long)

        return (raw_tensor[:-1].contiguous(), raw_tensor[1:].contiguous())


class TextDatasetV4(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: tokenizers.Tokenizer,
        downsample: int,
        seq_max_len: int = None,  # 兼容性参数
        re_tokenize=False,
        batch=True,
        padding_side="right",
    ):
        """初始化模型。

        Args:
            data_dir (str): 数据目录的路径。
            vocab_size (int): 词汇表的大小。
            downsample (float或bool): 数据下采样率，控制是否对数据进行下采样。
            batch (bool): 是否使用batch流程，速度提升但无进度条

        Returns:
            None
        """
        super().__init__()
        self.batch_pipeline = batch
        self.tokenizer = tokenizer
        self.padding_side = padding_side

        self.load_and_preprocess_data(
            data_dir, downsample, re_tokenize
        )  # 加载并预处理数据目录中的数据

        self.pad_data()  # 将预处理后的数据编码为词汇表索引

        self.seq_max_len = len(
            self.raw_data[0]
        )  # 设置最大序列长度（padding后所有序列长度一致）

    @DebugTimer("加载并生成训练数据")
    def load_and_preprocess_data(self, data_dir, downsample, re_tokenize):
        """
        加载并预处理数据目录中的文本数据，将其编码为token序列，并生成输入输出对

        参数：
            data_dir (str): 文本数据文件路径，每行一个样本
            downsample (float|bool): 下采样率，数值类型时按间隔采样，布尔值时控制是否启用采样
            re_tokenize (bool): 是否重新进行分词处理。若为False则假定数据已用空格预分词
            batch (bool): 是否启用批量处理流程

        返回值：
            None: 处理结果存储在类的input_data和output_data属性中
        """
        # 逐行读取文件并按指定采样率抽取数据行，避免一次性加载整个文件到内存
        self.input_data = []
        self.output_data = []
        self.raw_data = []
        self.word2idx = self.tokenizer.get_vocab()
        unk_token = json.loads(self.tokenizer.to_str())["model"]["unk_token"]

        line_count = 0

        with open(data_dir, encoding="utf-8") as f:
            data_len = sum(1 for line in f)
            f.seek(0)
            for line in tqdm(f, total=data_len):
                if line_count % downsample == 0:  # 实现下采样功能
                    if re_tokenize:
                        # 重新分词处理
                        if self.batch_pipeline:
                            # 批量处理模式下先收集数据
                            self.raw_data.append(line.strip())
                        else:
                            # 单条处理模式
                            line_ = line.strip()
                            line_ = torch.tensor(self.tokenizer.encode(line_).ids)
                            self.raw_data.append(line_)
                    else:
                        # 预分词数据处理
                        if self.batch_pipeline:
                            # 批量处理模式下先收集数据
                            self.raw_data.append(line.split(" "))
                        else:
                            # 单条处理模式
                            line_ = []
                            for i in line.split(" "):
                                try:
                                    line_.append(self.word2idx[i])
                                except KeyError:
                                    line_.append(self.word2idx[unk_token])

                            self.raw_data.append(line_)
                # if line_count == 80_0000:
                #     tracemalloc.start()
                # if line_count % 10_0000 == 9_9999 and line_count > 80_0000:
                #     # 获取当前内存快照
                #     current, peak = tracemalloc.get_traced_memory()
                #     print(f"当前内存使用: {current / 1024 / 1024:.2f} MB")
                #     print(f"峰值内存使用: {peak / 1024 / 1024:.2f} MB")

                #     # 查看内存分配最多的前5行代码
                #     snapshot = tracemalloc.take_snapshot()
                #     top_stats = snapshot.statistics('lineno')

                #     print("内存分配最多的前5行:")
                #     for stat in top_stats[:5]:
                #         print(stat)
                # line_count += 1

        # 如果是批量处理模式，现在进行批量编码
        if self.batch_pipeline and self.raw_data:
            if re_tokenize:
                # 对原始文本进行批量编码
                lines_stripped = [line.strip() for line in self.raw_data]
                encoded_lines = self.tokenizer.encode_batch_fast(lines_stripped)
            else:
                # 对预分词数据进行批量编码
                encoded_lines = self.tokenizer.encode_batch_fast(
                    self.raw_data, is_pretokenized=True
                )

            # 重新构建input_data和output_data
            self.raw_data = [torch.tensor(line.ids) for line in tqdm(encoded_lines)]

    @DebugTimer("编码数据")
    def pad_data(self):
        if self.batch_pipeline:
            ...
        # 确保所有元素都是张量而不是列表
        self.raw_data = [
            torch.tensor(line) if not isinstance(line, torch.Tensor) else line
            for line in self.raw_data
        ]
        self.raw_data = torch.nn.utils.rnn.pad_sequence(
            self.raw_data, batch_first=True, padding_side=self.padding_side
        )

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        """
        根据索引获取数据样本
        参数:
            index (int): 索引值
        返回值:
            tuple: 输入数据和输出数据的元组，以tensor形式表示
        """

        return (
            self.raw_data[index][:-1].clone().detach(),
            self.raw_data[index][1:].clone().detach().long(),
        )


class TextDatasetV3(torch.utils.data.Dataset):
    def __init__(self, data_dir, vocab_size, downsample):
        """初始化模型。

        Args:
            data_dir (str): 数据目录的路径。
            vocab_size (int): 词汇表的大小。
            downsample (float或bool): 数据下采样率，控制是否对数据进行下采样。

        Returns:
            None
        """
        super().__init__()
        self.tokenizer = lambda x: x.split(" ")  # 定义基于空格的简单分词器
        self.load_and_preprocess_data(
            data_dir, downsample
        )  # 加载并预处理数据目录中的数据
        self.build_vocab(vocab_size)  # 根据指定大小构建词汇表
        self.encode_data()  # 将预处理后的数据编码为词汇表索引
        self.seq_max_len = len(
            self.input_data[0]
        )  # 设置最大序列长度（padding后所有序列长度一致）

    @DebugTimer("加载并生成训练数据")
    def load_and_preprocess_data(self, data_dir, downsample):
        with open(data_dir, encoding="utf-8") as f:
            lines = f.read().splitlines()[::downsample]
        self.input_data = [self.tokenizer(line)[:-1] for line in lines]
        self.output_data = [self.tokenizer(line)[1:] for line in lines]

    @DebugTimer("构建词典")
    def build_vocab(self, vocab_size):
        all_words = [
            word for line in self.input_data + self.output_data for word in line
        ]
        word_counts = collections.Counter(all_words).most_common(vocab_size)
        self.vocab = Vocab(word_counts, specials=["<PAD>", "<UNK>"])
        self.word2index = self.vocab.stoi
        self.index2word = self.vocab.itos
        self.word_nums = len(self.vocab)

    @DebugTimer("编码数据")
    def encode_data(self):
        word2index = self.word2index
        self.input_data = [
            torch.tensor(
                [word2index.get(word, self.vocab.default_index) for word in sentence]
            )
            for sentence in tqdm(self.input_data)
        ]
        self.output_data = [
            torch.tensor(
                [word2index.get(word, self.vocab.default_index) for word in sentence]
            )
            for sentence in tqdm(self.output_data)
        ]

        self.input_data = torch.nn.utils.rnn.pad_sequence(
            self.input_data, batch_first=True
        )
        self.output_data = torch.nn.utils.rnn.pad_sequence(
            self.output_data, batch_first=True
        )

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        """
        根据索引获取数据样本
        参数:
            index (int): 索引值
        返回值:
            tuple: 输入数据和输出数据的元组，以tensor形式表示
        """
        return (
            self.input_data[index].clone().detach(),
            self.output_data[index].clone().detach().long(),
        )


class Vocab:
    def __init__(self, word_counts, specials=["<PAD>", "<UNK>"]):

        # 先添加特殊token到词典
        self.stoi = {}
        self.itos = []
        for special in specials:
            self.stoi[special] = len(self.stoi)

        # 添加普通单词
        for word, _ in word_counts:
            self.stoi.setdefault(word, len(self.stoi))

        # 设置默认索引为`<UNK>`的索引
        self.itos = list(self.stoi.keys())
        self.default_index = self.stoi["<UNK>"]

    def __call__(self, word):
        """
        调用这个方法比直接调用
        self.stoi.get(word, self.default_index)
        慢很多...
        """
        # 如果单词不在词典中，返回`<UNK>`的索引
        return self.stoi.get(word, self.default_index)

    def __len__(self):
        return len(self.stoi)

    def build_from_dict(self, word_dict):
        self.stoi = word_dict
        self.itos = list(self.stoi.keys())
        self.default_index = self.stoi["<UNK>"]


if __name__ == "__main__":
    import time

    dataset = PretrainTextDataset(
        r"data_large_ChatML.txt",
        downsample=10,
        tokenizer=tokenizers.Tokenizer.from_file(r"tokenizer/bpe_tokenizer_6k_0724_ChatML.json"),
        re_tokenize=False,
        # batch=True,
        seq_max_len=192,
        padding_side="left",
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, pin_memory=False, num_workers=0
    )
    t1 = time.perf_counter()
    for i, (inputs, targets) in enumerate(train_loader):
        if i % 100 == 0:
            print((time.perf_counter() - t1) * 10, "ms")
            t1 = time.perf_counter()
