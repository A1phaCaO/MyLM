import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from model import *
from utils import TextGenerator
from tokenizers import Tokenizer  # 引入 tokenizers 库
import tokenizers



# 导入模型文件
model_dir = r"ckpt\ckpt_epoch_17.pth"
tokenizer_dir = r"bpe_tokenizer_6k_0517.json"
model = torch.load(model_dir, weights_only=False)

# 使用 tokenizers 库加载 tokenizer
tokenizer = Tokenizer.from_file(tokenizer_dir)

test_generator = TextGenerator(model, tokenizer, 'cuda', padding_side="left")
MAX_LEN = 100
T=0.7

while True:
    start = input("In>>")

    if start[:2] == 'T=':
        T = float(start[2:])
        print(f'T={T}')
    else:
        print(f'T={T}\n' +
            "".join(
                test_generator.generate(
                    start_token=start,
                    gen_seq_len=MAX_LEN,
                    temperature=T,
                    # top_k=50,
                    # top_p=0.7,
                    frequency_penalty=0.5,
                    print_out=False
                )
            )
        )
