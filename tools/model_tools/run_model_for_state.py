import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根（tools/model_tools/ 上两级）

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import models as m
from utils import TextGenerator, model_structure
from tokenizers import Tokenizer  # 引入 tokenizers 库
import tokenizers
import rich.markdown

# 导入模型文件
config_dir = r"model\config_sft_ca8conv4_synthetic_mix_nano.json"  # SFT 训练时的模型本征配置
tokenizer_dir = r"tokenizer/bbpe_tokenizer_7k_260723_xl.json"
# 权重要与 config 配套：这里默认加载 SFT 产出，否则 SFT 配置 + base 权重只会得到 base 模型的行为
model_dir = r"model\model_sft_ca8conv4_synthetic_mix_nano.pth"

with open(config_dir, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 使用 tokenizers 库加载 tokenizer
tokenizer = Tokenizer.from_file(tokenizer_dir)
# args = m.MyLMArgs(
#             d_model=256,
#             d_inner=int(((256 * (8 / 3)) // 64) * 64),
#             n_layers=1,
#             use_moe=False,
#             n_experts=None,
#             vocab_size=tokenizer.get_vocab_size(),
#             seq_max_len=192,
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
    d_latent=config['d_latent'],
    latent_moe=config['latent_moe'],
    n_experts_per_tok=config['n_experts_per_tok'],
    d_conv=config.get('d_conv', 4),              # 终版架构 conv4
    compress_ratio=config.get('compress_ratio', 8),  # 终版架构 ca8
    vocab_size=tokenizer.get_vocab_size(),
    seq_max_len=config['seq_max_len'],  # 必须与权重配套：SFT ckpt 是 768，base 是 384
    conv_bias=False,
    ffn_bias=False,
    attn_bias=True,
    dropout=0,
)
print(config)
model = m.MyLM(args).to('cuda')
model_structure(model)
raw = torch.load(model_dir, map_location='cuda', weights_only=False)
# 兼容 checkpoint dict 与裸 state_dict
if isinstance(raw, dict) and "model_state_dict" in raw and not any(
    k.endswith(".weight") or k.endswith(".bias") for k in raw.keys()
):
    print("检测到 checkpoint dict，提取 model_state_dict")
    state_dict = raw["model_state_dict"]
else:
    state_dict = raw

# 兼容多种前缀：DataParallel 的 module. 与 torch.compile 的 _orig_mod.
PREFIXES = ("module.", "_orig_mod.")
if any(k.startswith(PREFIXES) for k in state_dict.keys()):
    stripped = {}
    for k, v in state_dict.items():
        for p in PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
                break
        stripped[k] = v
    state_dict = stripped
    print(f"已剥离前缀 {PREFIXES}")

# RoPE 的 cos_cached/sin_cached 是按当前 seq_max_len 预计算的 buffer，
# 若 checkpoint 与当前模型 seq_max_len 不一致会导致 size mismatch。
# 它们会在前向时按实际 seq_len 重新切片，删除后不影响加载与计算。
ROPE_SUFFIXES = ("attn.cos_cached", "attn.sin_cached")
rope_buffers = [k for k in state_dict if k.endswith(ROPE_SUFFIXES)]
if rope_buffers:
    for k in rope_buffers:
        del state_dict[k]
    print(f"已丢弃 {len(rope_buffers)} 个 RoPE buffer（cos/sin_cached）")


def is_rope_buffer(key: str) -> bool:
    """RoPE buffer 不属于可训练权重：按 seq_max_len 预计算、前向时按实际长度切片，
    加载前已主动删除，因此「缺失」它们是预期行为，不算加载失败。"""
    return key.endswith(ROPE_SUFFIXES)


try:
    model.load_state_dict(state_dict, strict=True)
except Exception as e:
    print(f'{str(e)[:70]}...')
    miss, unexpect = model.load_state_dict(state_dict, strict=False)
    # 过滤掉预期内的 RoPE buffer 缺失，剩下的才是真问题
    real_miss = [k for k in miss if not is_rope_buffer(k)]
    print(
        f'已使用非严格加载\n'
        f'缺失 {len(miss)} 个（其中 RoPE buffer {len(miss) - len(real_miss)} 个，预期内）'
        f'，未匹配 {len(unexpect)} 个'
    )
    if real_miss:
        # 缺失真实权重时不能静默继续：这部分参数仍是随机初始化，输出会失真
        raise SystemExit(
            f"加载失败：缺 {len(real_miss)} 个权重参数（如 {real_miss[:5]}），"
            f"检查 config / 权重是否配套"
        ) from None
    if unexpect:
        raise SystemExit(f"加载失败：{len(unexpect)} 个参数未匹配：{unexpect[:5]}") from None

test_generator = TextGenerator(model, tokenizer, 'cuda', padding_side="none")
MAX_LEN = 256  # 生成步数上限（context 上限由 seq_max_len 决定，这里只控生成长度）
T = 0.7
TOP_P = 0.95         # 核采样，与 SFTTrainer.generate_test 对齐
REP_P = 1.1          # 经典重复惩罚，与 SFTTrainer.generate_test 对齐
INSTURCT_MODE = True  # True: SFT 指令模式，False: 纯文本续写模式
# SFT 训练是「只训 assistant 正文 + <|im_end|>」、前缀不计 loss：
# 推理端必须把 <|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n 完整给全，
# 且内容以 <|im_end|> 结束（模型正是靠这个 token 学会停）。模板必须与训练一致。
MSG_SEP = "\\n"  # 字面反斜杠+n（真实换行 0x0A 会被 tokenizer 编码成 <|unk|>）
ASSISTANT_PREFIX = f"<|im_start|>assistant{MSG_SEP}"
_im_end_id = tokenizer.token_to_id("<|im_end|>")


def build_prompt(user_text: str, history: list | None = None) -> str:
    """按 Qwen ChatML 拼提示：每条消息 <|im_start|>role\\n内容<|im_end|>\\n，末尾追加 assistant 头。

    history 为 [(role, text), ...] 的形式，role ∈ {"user", "assistant"}，
    内容统一保存为字面 \\n（与训练数据一致），显示时才替换成真实换行。
    """
    parts = list(history or [])
    parts.append(("user", user_text.replace("\n", MSG_SEP)))
    body = "".join(
        f"<|im_start|>{role}{MSG_SEP}{text}<|im_end|>{MSG_SEP}"
        for role, text in parts
    )
    return body + ASSISTANT_PREFIX


def split_generated(prompt: str, full: str) -> str:
    """只取模型生成的部分。

    按 token 偏移裁剪（而非字符串 split）：prompt 本身也含 "assistant" 字样，
    split 会误切；且 encode(prompt) 是 decode 的逆，偏移可靠。
    """
    n_prompt = len(tokenizer.encode(prompt).ids)
    return tokenizer.decode(
        tokenizer.encode(full).ids[n_prompt:], skip_special_tokens=False
    )


while True:
    if INSTURCT_MODE:
        start = input("Ask>>")
    else:
        start = input("In>>")

    if start[:2] == 'T=':
        T = float(start[2:])
        print(f'T={T}')
    elif start[:2] == 'P=':
        TOP_P = float(start[2:])
        print(f'TOP_P={TOP_P}')
    elif start[:2] == 'R=':
        REP_P = float(start[2:])
        print(f'REP_P={REP_P}')
    else:

        if INSTURCT_MODE:
            # 字面 \n 对齐 SFT 数据格式（tokenizer 把真实换行归为 <|unk|>）
            prompt = build_prompt(start)
        else:
            prompt = start
        print(f"原始输入：{prompt}")
        raw_ans = test_generator.generate(
            start_token=prompt,
            gen_seq_len=MAX_LEN,
            temperature=T,
            top_k=20,
            top_p=TOP_P,
            repetition_penalty=REP_P,
            frequency_penalty=0.5,
            print_out=True,
            eos_id=_im_end_id,
        )
        # 按 token 偏移只取生成部分，再把字面 \n 还原成换行用于显示
        gen_text = split_generated(prompt, raw_ans)
        pretty = gen_text.replace(MSG_SEP, "\n")
        rich.print(f"---最终输出---\n{pretty}")
