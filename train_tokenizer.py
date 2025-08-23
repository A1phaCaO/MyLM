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
)


BBPE = False

# 1. 初始化 BPE 模型
tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.normalizer = normalizers.NFKD()

# 2. 设置预分词器
tokenizer.pre_tokenizer = (
    pre_tokenizers.ByteLevel() if BBPE else pre_tokenizers.BertPreTokenizer()
)
initial_alphabet = list(string.ascii_letters) + list(string.digits) + ['\n']
print(initial_alphabet)
# 3. 定义训练器
trainer = trainers.BpeTrainer(
    special_tokens=["<|endoftext|>", "<|unk|>", "<|im_end|>", "<|im_start|>"],
    initial_alphabet=initial_alphabet,
    vocab_size=6140,  # 词汇表大小
    limit_alphabet=4800,
    min_frequency=1,  # pair最小出现频率
    show_progress=True,
)

# 4. 训练分词器
files = [
    # r"Train_text\SkyPile2023-14_zh_head_0000_downsample2x_processed.jsonl",
    r"train_text\WanJuan1.0part-000036-a894b46e-downsample20x-processed.txt",
    r"train_text\SkyPile2022-40_zh_middle_0011_processed.txt", # 通用知识
    r"train_text\时政文章.txt", # 时政
    r"train_text\斗罗大陆4终极斗罗.txt", # 泛化
    r"train_text\高三议论文-作文网20220310-20200806.txt", # 泛化
    r"train_text\SFT\Infinity-Instruct-Gen-00000-of-00015-processed.txt" # 优化对话
]
tokenizer.train(files, trainer)

if BBPE:
    tokenizer.post_processor = processors.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

# 5. 保存分词器
tokenizer.save("bpe_tokenizer_6k_0724_ChatML.json")
