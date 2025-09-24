import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import models_250830 as m
from utils import TextGenerator, model_structure
from tokenizers import Tokenizer  # 引入 tokenizers 库
import tokenizers



# 导入模型文件
model_dir = r"model\model_sft.pth"
tokenizer_dir = r"bpe_tokenizer_6k_0724_ChatML.json"
config_dir = r"model\config_0830.json"

with open(config_dir, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 使用 tokenizers 库加载 tokenizer
tokenizer = Tokenizer.from_file(tokenizer_dir)
# args = m.MyLMArgs(
#             d_model=288,
#             d_inner=int(((288 * (8 / 3)) // 64) * 64),
#             n_layers=2,
#             use_moe=False,
#             n_experts=None,
#             vocab_size=tokenizer.get_vocab_size(),
#             seq_max_len=512,
#             conv_bias=False,
#             ffn_bias=False,
#             attn_bias=False,
#             dropout=0.1,
#         )
args = m.MyLMArgs(
            d_model=config['d_model'],
            d_inner=config['d_inner'],
            n_layers=config['n_layers'],
            use_moe=config['use_moe'],
            n_experts=config['n_experts'],
            n_heads=config['n_heads'],
            d_head=config['d_head'],
            vocab_size=tokenizer.get_vocab_size(),
            seq_max_len=192,
            conv_bias=False,
            ffn_bias=False,
            attn_bias=True,
            dropout=0,
        )
print(config)
model = m.MyLM(args).to('cuda')
model_structure(model)
state_dict = torch.load(model_dir)
if any([k.startswith('module.') for k in state_dict.keys()]):
    print('该模型使用了DataParallel')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        


try:
    model.load_state_dict(state_dict, strict=True)
except Exception as e:
    print(f'{str(e)[:70]}...')
    miss, unexpect = model.load_state_dict(state_dict, strict=False)
    print(f'已使用非严格加载\n缺失{len(miss)}个参数，未匹配{len(unexpect)}个参数')
    if len(miss) < 10:
        print(f'缺失参数：{miss}')
    if len(unexpect) < 10:
        print(f'未匹配参数：{unexpect}')

test_generator = TextGenerator(model, tokenizer, 'cuda', padding_side="left")
MAX_LEN = 256
T=0.6
INSTURCT_MODE = True

while True:
    if INSTURCT_MODE:
        start = input("Ask>>")
    else:
        start = input("In>>")

    if start[:2] == 'T=':
        T = float(start[2:])
        print(f'T={T}')
    else:
        if INSTURCT_MODE:
            start = f"<|im_start|>user\n{start}<|im_end|>\n<|im_start|>assistant\n"
        print(f'temperature={T}\n' +
            "".join(
                test_generator.generate(
                    start_token=start,
                    gen_seq_len=MAX_LEN,
                    temperature=T,
                    top_k=20,
                    # top_p=0.7,
                    frequency_penalty=0.5,
                    print_out=False
                )
            )
        )
