import torch
import collections
from utils import DebugTimer
from tqdm import tqdm
import tokenizers
import json

class TextDatasetV4(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: tokenizers.Tokenizer,
        downsample: int,
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
            self.input_data[0]
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
                对于
            
        返回值：
            None: 处理结果存储在类的input_data和output_data属性中
        """
        # 读取文件并按指定采样率抽取数据行
        with open(data_dir, encoding="utf-8") as f:
            lines = f.read().splitlines()[::downsample]
        self.input_data = []
        self.output_data = []
        self.word2idx = self.tokenizer.get_vocab()
        unk_token = json.loads(self.tokenizer.to_str())['model']['unk_token']
        # 根据是否重新分词选择不同处理分支
        if re_tokenize:
            # 批量处理模式：快速清理文本并进行批量编码
            if self.batch_pipeline:
                lines = [line.strip() for line in lines]
                lines = self.tokenizer.encode_batch_fast(lines)
            
            # 单条处理模式：带进度条逐行处理
            else:
                for line in tqdm(lines):
                    line_ = line.strip()
                    line_ = torch.tensor(self.tokenizer.encode(line_).ids)
                    self.input_data.append(line_[:-1])
                    self.output_data.append(line_[1:])
        else:
            # 批量处理预分词数据：按空格拆分后批量编码
            if self.batch_pipeline:
                lines = [line.split(" ") for line in lines]
                lines = self.tokenizer.encode_batch_fast(lines, is_pretokenized=True)
            # 单条处理预分词数据：逐行拆分并编码
            else:
                for line in tqdm(lines):
                    line_ = []
                    for i in line.split(" "):
                        # if i in self.word2idx:
                        #     line_.append(self.word2idx[i])
                        # else:
                        #     line_.append(self.word2idx[unk_token])
                        try:
                            line_.append(self.word2idx[i])
                        except KeyError:
                            line_.append(self.word2idx[unk_token])
                        

                    self.input_data.append(torch.tensor(line_[:-1]))
                    self.output_data.append(torch.tensor(line_[1:]))
        
        # 批量模式下统一转换编码结果为张量
        if self.batch_pipeline:
            self.input_data = [torch.tensor(line.ids[:-1]) for line in tqdm(lines)]
            self.output_data = [torch.tensor(line.ids[1:]) for line in tqdm(lines)]

    @DebugTimer("编码数据")
    def pad_data(self):
        if self.batch_pipeline:
            ...
            # 训练过程暂时保留torch pad
        self.input_data = torch.nn.utils.rnn.pad_sequence(
            self.input_data, batch_first=True, padding_side=self.padding_side
        )
        self.output_data = torch.nn.utils.rnn.pad_sequence(
            self.output_data, batch_first=True, padding_side=self.padding_side
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
        self.load_and_preprocess_data(data_dir, downsample)  # 加载并预处理数据目录中的数据
        self.build_vocab(vocab_size)  # 根据指定大小构建词汇表
        self.encode_data()  # 将预处理后的数据编码为词汇表索引
        self.seq_max_len = len(self.input_data[0])  # 设置最大序列长度（padding后所有序列长度一致）

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
        