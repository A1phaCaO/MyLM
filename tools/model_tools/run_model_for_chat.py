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
config_dir = r"model\config_m_sftv2.json"  # SFT 模型用 SFT 配置（seq_max_len=512）
tokenizer_dir = r"tokenizer/bbpe_tokenizer_7k_260723_xl.json"
model_dir = r"model\model_m_sftv2.pth"

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
    seq_max_len=512,
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
rope_buffers = [k for k in state_dict if k.endswith(
    "attn.cos_cached") or k.endswith("attn.sin_cached")]
if rope_buffers:
    for k in rope_buffers:
        del state_dict[k]
    print(f"已丢弃 {len(rope_buffers)} 个 RoPE buffer（cos/sin_cached）")


try:
    model.load_state_dict(state_dict, strict=True)
except Exception as e:
    print(f'{str(e)[:70]}...')
    miss, unexpect = model.load_state_dict(state_dict, strict=False)
    print(f'已使用非严格加载\n缺失{len(miss)}个参数，未匹配{len(unexpect)}个参数')
    if len(miss) < 100:
        print(f'缺失参数：{miss}')
    if len(unexpect) < 100:
        print(f'未匹配参数：{unexpect}')

test_generator = TextGenerator(model, tokenizer, 'cuda', padding_side="none")
MAX_LEN = 256  # 生成步数上限（SFT config seq_max_len 为 512，此处仅控制生成长度）
T = 0.8
TOP_P = 0.95         # 核采样，与 SFTTrainer.generate_test 对齐
REP_P = 1.1          # 经典重复惩罚，与 SFTTrainer.generate_test 对齐
INSTURCT_MODE = True
_im_end_id = tokenizer.token_to_id("<|im_end|>")

# ============ 多轮对话开关 ============
MULTI_TURN = True           # True=多轮对话（保留上下文）；False=单轮（每次独立提问）
HISTORY_TOKEN_BUDGET = 384  # 多轮时历史上下文 token 预算（< seq_max_len=512，避免截断破坏格式）
# ===================================

history: list = []  # 多轮历史 [(role, text), ...]，role ∈ {"user", "assistant"}


def trim_history(h: list, budget: int) -> list:
    """历史 token 超预算时，从最旧的一轮（user+assistant 两条）开始丢弃。"""
    def total_len() -> int:
        return sum(len(tokenizer.encode(t).ids) for _, t in h)
    while len(h) >= 2 and total_len() > budget:
        h.pop(0)
        h.pop(0)
    return h


def build_prompt(h: list) -> str:
    """把多轮历史拼成 ChatML 格式 prompt（字面 \\n 对齐 SFT 数据格式）。"""
    parts = [f"<|im_start|>{role}\\n{text}<|im_end|>\\n" for role, text in h]
    parts.append("<|im_start|>assistant\\n")
    return "".join(parts)


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
    elif start.strip().lower() in ("clear", "cls", "new"):
        history.clear()
        print("已清空对话历史")
    else:
        if MULTI_TURN and INSTURCT_MODE:
            # 多轮：追加本轮问题，超预算时丢弃最旧轮次，再拼完整 prompt
            history.append(("user", start))
            history = trim_history(history, HISTORY_TOKEN_BUDGET)
            prompt = build_prompt(history)
        else:
            history.clear()
            if INSTURCT_MODE:
                # 字面 \n 对齐 SFT 数据格式（tokenizer 把真实换行归为 <|unk|>）
                prompt = f"<|im_start|>user\\n{start}<|im_end|>\\n<|im_start|>assistant\\n"
            else:
                prompt = start
        print("\r", end="")
        prompt_n = len(tokenizer.encode(prompt).ids)
        raw_ans = test_generator.generate(
            start_token=prompt,
            gen_seq_len=MAX_LEN,
            temperature=T,
            top_k=20,
            top_p=TOP_P,
            repetition_penalty=REP_P,
            frequency_penalty=1,
            eos_id=_im_end_id,
        )
        # 只取生成部分：按 token 偏移截取（prompt 未超长截断时，decode 与生成内部序列一致）
        ans_raw = tokenizer.decode(
            tokenizer.encode(raw_ans).ids[prompt_n:], skip_special_tokens=False
        )
        ans = ans_raw.replace("\\n", "\n")
        if MULTI_TURN and INSTURCT_MODE:
            # 历史保存字面 \n（与 SFT 训练格式一致），显示时才替换为真实换行
            history.append(("assistant", ans_raw))
        # rich.print("---")
        ans = rich.markdown.Markdown(f"---\nuser: {start}\n\n{ans}")
        rich.print(ans)