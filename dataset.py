import torch
import collections
from utils import DebugTimer
from tqdm import tqdm
import tokenizers
import json
import sys

# import tracemalloc


class StreamingTextDataset(torch.utils.data.Dataset):
    """
    流式文本数据集，避免将整个数据集加载到内存中
    只存储文件路径和行位置信息，在需要时才读取特定行
    """
    def __init__(
        self,
        data_dir: str,
        tokenizer: tokenizers.Tokenizer,
        seq_max_len: int = 192,
        downsample: int = 1,
        batch: bool = None, # 兼容性参数
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
        with open(self.data_dir, 'rb') as f:
            offset = 0
            line_count = 0
            while True:
                if line_count % downsample == 0:
                    self.line_offsets.append(offset)
                
                line = f.readline()
                if not line:
                    break
                    
                offset += len(line)
                line_count += 1
    
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

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, index):
        """
        根据索引获取数据样本，只在需要时读取特定行
        """
        # 根据索引定位并读取特定行
        with open(self.data_dir, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[index])
            line = f.readline().strip()
        
        # 进行tokenization
        if self.re_tokenize:
            # 如果需要重新分词，直接使用原始字符串
            raw = self.tokenizer.encode(line).ids
        else:
            # 如果使用预分词数据，需要先将字符串分割成列表
            raw = self.tokenizer.encode(
                line.split(" "), is_pretokenized=True
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


class RuntimeTextDatasetV4(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: tokenizers.Tokenizer,
        seq_max_len: int = 192,
        downsample: int = 1,
        re_tokenize: bool = False, 
        batch: bool = None, # 兼容性参数
        padding_side: str = "right",
    ):
        """初始化模型。

        Args:
            data_dir (str): 数据目录的路径。
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
    dataset = StreamingTextDataset(
        r"data_large_ChatML.txt",
        downsample=10,
        tokenizer=tokenizers.Tokenizer.from_file(r"bpe_tokenizer_6k_0724_ChatML.json"),
        re_tokenize=False,
        # batch=True,
        seq_max_len=192,
        padding_side="left",
    )
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            pin_memory=False,
            num_workers=0
        )
    t1 = time.perf_counter()
    for i, (inputs, targets) in enumerate(train_loader):
        if i % 100==0:
            print((time.perf_counter() - t1)*10, 'ms')
            t1 = time.perf_counter()
