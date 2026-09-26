from tqdm import tqdm
from tokenizers import Tokenizer
import codecs
import os
import random
import numpy as np

# 加载tokenizer
TOKENIZER_PATH = r"tokenizer/bbpe_tokenizer_7k_260723_xl.json"
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# 通过字典形式定义路径和对应的采样比率（sample_rate）
# sample_rate 含义：
#   - (0, 1)  : 降采样，随机抽取该百分比的行。例如 0.001 表示随机抽取 0.1% 的行
#   - 1       : 不采样，使用全部数据
#   - > 1     : 重复（repeat），将数据重复若干次。例如 3 表示将数据重复 3 倍
path_sample_dict = {
    r"train_text\ultrafineweb-l3-mutistyle-cn-part0.txt": 0.63,
    r"train_text\ultrafineweb-zh-part-001-of-256-downsample8x.txt": 1,
    r"train_text\SkyPile2023-14_zh_middle_0010_processed.txt": 1,
    r"train_text\WanJuan1.0part-000036-a894b46e-downsample30x-processed.txt": 0.31,
    r"train_text\ultrafineweb-l3-mutistyle-en-part0.txt": 0.28,
    r"train_text\SkyPile2022-40_zh_middle_0011_processed.txt": 0.34
}

# 设定句子的最大长度
SENTENCE_MAXLEN = 256 + 1
BATCH_SIZE = 2048  # 设置合适的批量大小
# 定义分隔符和是否从符号位置开始切分句子的标志
SPLIT_SYMBOL = (
    "。",
    "，",
    "？",
    "；",
    "！",
    "!",
    "?",
)
SPLIT_FROM_SYMBOL = False
OUTPUT_PATH = r"data/medium_data256v3.npy"
# 输出 dtype：默认 uint16（vocab=7160 ≪ 65535），未来词表扩张到 >65535 时改 np.int32
OUTPUT_DTYPE = np.uint16
SHUFFLE = True  # 按BATCH_SIZE进行随机打乱
# BOS/EOS 配置常量（可修改）
BOS_TOKEN_STR = ""  # BOS token 字符串
# EOS token 字符串：从 tokenizer 的 added_tokens 中取 id=0 的内容（避免硬编码字符串出错）
# 注意：v2 中此处为空字符串 ""，导致 EOS 实际从未被添加（潜在 bug）。v3 修正此问题。
import json as _json
_tok_dict = _json.loads(tokenizer.to_str())
EOS_TOKEN_STR = ""
for _at in _tok_dict["added_tokens"]:
    if _at["id"] == 0:
        EOS_TOKEN_STR = _at["content"]
        break
print(f"EOS_TOKEN_STR 取自 tokenizer added_tokens[0]: {EOS_TOKEN_STR!r} (id=0)")

# 是否在生成的数据中添加 BOS/EOS 标记
ADD_BOS = False   # 是否在每个段落开头添加 BOS token
ADD_EOS = True   # 是否在每个段落结尾添加 EOS token


# ---------- 采样比率说明 ----------
# sample_rate in (0, 1) → 降采样：随机抽取对应百分比的行
# sample_rate == 1      → 不采样：使用全部数据
# sample_rate > 1       → 重复（repeat）：将全部数据重复 int(sample_rate) 次
# -----------------------------------

# dtype 安全检查：vocab_size 必须适配 OUTPUT_DTYPE
_vocab_size = tokenizer.get_vocab_size()
_max_id = np.iinfo(OUTPUT_DTYPE).max
assert _vocab_size <= _max_id, (
    f"vocab_size={_vocab_size} 超过 {OUTPUT_DTYPE} 上限 {_max_id}，请改用 np.int32"
)
print(f"dtype 安全检查通过：vocab_size={_vocab_size}, {OUTPUT_DTYPE} 上限={_max_id}")


def get_sampled_size_mb(original_size_mb, sample_rate):
    """根据采样比率计算预期输出大小（MB）"""
    if sample_rate <= 0:
        return 0
    elif sample_rate <= 1:
        return original_size_mb * sample_rate
    else:
        return original_size_mb * int(sample_rate)

def get_sample_label(sample_rate):
    """返回可读的采样比率标签"""
    if sample_rate < 1:
        return f"downsample {sample_rate*100:.2f}%"
    elif sample_rate == 1:
        return "1x (全量)"
    else:
        return f"repeat {int(sample_rate)}x"

print(
    f"{'文件路径':<50} {'原始大小(MB)':<9} {'采样后大小(MB)':<10} {'占比':<10} {'采样方式':<20} "
)
print("-" * 120)
total_size_mb = 0
total_sampled_size_mb = 0
# 首先计算总大小，用于计算占比
for INPUT_PATH in path_sample_dict.keys():
    sample_rate = path_sample_dict.get(INPUT_PATH, 1)
    original_size_bytes = os.path.getsize(INPUT_PATH)
    original_size_mb = original_size_bytes / (1024 * 1024)
    sampled_size_mb = get_sampled_size_mb(original_size_mb, sample_rate)
    total_size_mb += original_size_mb
    total_sampled_size_mb += sampled_size_mb

# 显示每个文件的统计信息，包括占比
for INPUT_PATH in path_sample_dict.keys():
    sample_rate = path_sample_dict.get(INPUT_PATH, 1)
    original_size_bytes = os.path.getsize(INPUT_PATH)
    original_size_mb = original_size_bytes / (1024 * 1024)
    sampled_size_mb = get_sampled_size_mb(original_size_mb, sample_rate)
    percentage = (
        (sampled_size_mb / total_sampled_size_mb) * 100
        if total_sampled_size_mb > 0
        else 0
    )
    print(
        f"{INPUT_PATH[:31]+'...'+INPUT_PATH[-20:]:<50} {original_size_mb:<15.2f} {sampled_size_mb:<15.2f} {f'{percentage:.1f}%':<10} {get_sample_label(sample_rate):<20}"
    )

print("-" * 120)
print(
    f"共计{len(path_sample_dict)}个文件".ljust(50)
    + f"{total_size_mb:.1f}MB".ljust(16)
    + f"{total_sampled_size_mb:.1f}MB".ljust(16)
    + f"100%".ljust(16)
)
print("=" * 120)


# 预查询 BOS/EOS token id（避免在循环里重复查）
BOS_TOKEN_ID = None
if ADD_BOS and BOS_TOKEN_STR:
    BOS_TOKEN_ID = tokenizer.token_to_id(BOS_TOKEN_STR)
    if BOS_TOKEN_ID is None:
        print(f"警告: BOS token '{BOS_TOKEN_STR}' 在 tokenizer 中未找到，将跳过添加")
EOS_TOKEN_ID = None
if ADD_EOS and EOS_TOKEN_STR:
    EOS_TOKEN_ID = tokenizer.token_to_id(EOS_TOKEN_STR)
    if EOS_TOKEN_ID is None:
        print(f"警告: EOS token '{EOS_TOKEN_STR}' 在 tokenizer 中未找到，将跳过添加")

# 全量收集所有 segment（每个 segment 是 list[int] token IDs）
all_segments = []

# 打开输入文件并处理
for INPUT_PATH in path_sample_dict.keys():
    # 获取当前文件的采样比率，默认为1（全量使用）
    sample_rate = path_sample_dict.get(INPUT_PATH, 1)

    # 每个文件单独处理
    with open(INPUT_PATH, "r", encoding="UTF-8", errors="ignore") as data:
        # 获取数据长度并重置文件读取位置
        try:
            data_len = len(list(data))
        except:
            # 如果第一次读取失败，尝试用latin-1重新打开文件
            data.close()
            data = codecs.open(INPUT_PATH, "r", encoding="latin-1", errors="ignore")
            data_len = len(list(data))
        data.seek(0)

        # 初始化存储输入输出数据的列表
        out_list = []

        # 尝试用UTF-8读取，如果失败则用latin-1读取
        try:
            all_lines = list(data)
        except:
            data.close()
            print(f"{INPUT_PATH} 文件编码错误，尝试用latin-1重新打开")
            data = codecs.open(INPUT_PATH, "r", encoding="latin-1", errors="ignore")
            all_lines = list(data)

        # 根据 sample_rate 进行采样或重复
        if sample_rate < 1:
            # 降采样：随机抽取 sample_rate 百分比的行
            num_samples = max(1, int(len(all_lines) * sample_rate))
            num_samples = min(num_samples, len(all_lines))
            if num_samples == 0:
                data_lines = []
                print(f"[降采样] {INPUT_PATH}: 空文件，跳过")
            else:
                data_lines = random.sample(all_lines, num_samples)
                print(f"[降采样] {INPUT_PATH}: {len(all_lines)} 行 → {len(data_lines)} 行 ({sample_rate*100:.2f}%)")
        elif sample_rate > 1:
            # 重复（repeat）：将全部数据重复 int(sample_rate) 次
            repeat_times = int(sample_rate)
            data_lines = all_lines * repeat_times
            print(f"[重复] {INPUT_PATH}: {len(all_lines)} 行 × {repeat_times} = {len(data_lines)} 行")
        else:
            # sample_rate == 1：使用全部数据
            data_lines = all_lines

        # 按 BATCH_SIZE 批量处理
        for i in tqdm(range(0, len(data_lines), BATCH_SIZE), desc=f"处理 {os.path.basename(INPUT_PATH)}"):
            batch = data_lines[i : i + BATCH_SIZE]
            batch = [item.strip("\n") for item in batch]

            # 批量编码
            encodings = tokenizer.encode_batch(batch)

            # 临时存储当前 batch 的 segment token 列表
            batch_out_list = []

            for encoding in encodings:
                tokens = encoding.ids  # token ID 的整数列表

                # 先在整行编码结果上添加 BOS/EOS（与 v2 一致：在切分前添加）
                if BOS_TOKEN_ID is not None:
                    tokens = [BOS_TOKEN_ID] + tokens
                if EOS_TOKEN_ID is not None:
                    tokens = tokens + [EOS_TOKEN_ID]

                # 按照 SENTENCE_MAXLEN 分割
                start_idx = 0
                while start_idx < len(tokens):
                    end_idx = start_idx + SENTENCE_MAXLEN

                    # 如果超过最大长度，尝试找到最近的分割符号
                    if SPLIT_FROM_SYMBOL and end_idx < len(tokens):
                        for j in range(end_idx, start_idx, -1):
                            if tokenizer.decode([tokens[j]]) in SPLIT_SYMBOL:
                                end_idx = j + 1
                                break

                    # 获取当前段落的 token（直接保存 token IDs，不再 decode 回文本）
                    segment_tokens = tokens[start_idx:end_idx]
                    batch_out_list.append(segment_tokens)
                    start_idx = end_idx

            # 当前 batch 内 shuffle
            if SHUFFLE:
                random.shuffle(batch_out_list)
            # 追加到全量
            all_segments.extend(batch_out_list)

# 全局 shuffle（让不同文件的数据充分混合）
if SHUFFLE:
    print("全局 shuffle 中...")
    random.shuffle(all_segments)

print(f"总样本数: {len(all_segments)}")

# Pad 每个 segment 到 SENTENCE_MAXLEN（pad_value=0），存为 2D numpy 数组
print(f"padding & saving 到 {OUTPUT_PATH} (dtype={OUTPUT_DTYPE})...")
padded = np.zeros((len(all_segments), SENTENCE_MAXLEN), dtype=OUTPUT_DTYPE)
for i, seg in enumerate(tqdm(all_segments, desc="padding & saving")):
    n = min(len(seg), SENTENCE_MAXLEN)
    padded[i, :n] = seg[:n]   # 剩余位置保持 0（pad_value）

np.save(OUTPUT_PATH, padded)
print(f"完成。shape={padded.shape}, dtype={padded.dtype}, "
      f"大小={padded.nbytes / 1024 / 1024:.1f}MB")
