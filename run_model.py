import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from models import *
from utils import TextGenerator
from tokenizers import Tokenizer  # 引入 tokenizers 库
import tokenizers

TestModel = MyLM
TestModelBlock = MyLMBlock
TestModelArgs = MyLMArgs

# 删除原有的 Vocab 类

# 导入模型文件
model_dir = r"C:\Users\laihong\Desktop\Pytorch_GPT\model\ckpt_3.pth"
tokenizer_dir = "bpe_tokenizer_6k.json"
model = torch.load(model_dir)

# 使用 tokenizers 库加载 tokenizer
tokenizer = Tokenizer.from_file(tokenizer_dir)

test_generator = TextGenerator(model, tokenizer, 'cuda')
MAX_LEN = 160
T=0.65
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
                    frequency_penalty=1,
                    print_out=False
                )
            )
        )
