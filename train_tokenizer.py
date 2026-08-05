import torch
import tokenizers
import string
from tokenizers import (
    normalizers,
    models,
    pre_tokenizers,
    trainers,
    processors,
    decoders,
    Tokenizer,
    Regex,
)

BBPE = True  # 是否使用 ByteLevel BPE
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|beginoftext|>",
    "<|pad|>",
    "<|unk|>",
    "<|im_end|>",
    "<|im_start|>",
]

# 1. 初始化 BPE 模型
tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>", cache_capacity=0))


if BBPE:
    tokenizer.normalizer = normalizers.NFC()
else:
    tokenizer.normalizer = normalizers.NFKD()

# 2. 设置预分词器
if BBPE:
    PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX),
                             behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
else:
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

if BBPE:
    initial_alphabet = []
else:
    initial_alphabet = list(string.ascii_letters) + \
        list(string.digits) + ["\n"]

print(initial_alphabet)

# 3. 定义训练器
trainer = trainers.BpeTrainer(
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=initial_alphabet,
    vocab_size=7168,
    limit_alphabet=65535,
    min_frequency=2,
    show_progress=True,
)

# 4. 使用 torch DataLoader 加载数据
file_path = r"train_text\merged.txt"

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, batch_size):
        self.batch_size = batch_size
        self.lines = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.lines.append(line.strip())  # Remove newline characters

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        batch = self.lines[idx:idx + self.batch_size]
        return batch


# Create the dataset, and process the full file.
dataset = TextDataset(file_path, batch_size=1)
dataset_len = len(dataset)
# DataLoader for efficient batch processing
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

# 使用 train_from_iterator 替代 train(files, trainer)
tokenizer.train_from_iterator(dataloader, trainer=trainer, length=dataset_len)

if BBPE:
    tokenizer.post_processor = processors.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

# 5. 保存分词器
tokenizer.save(r"bbpe_tokenizer_6k_260715.json")
