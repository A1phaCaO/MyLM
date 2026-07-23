from tqdm import tqdm
from tokenizers import Tokenizer
import codecs
import os
import random

# 加载tokenizer
TOKENIZER_PATH = r"bpe_tokenizer_7k_260215.json"
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)


# 将输入路径按逗号分割成列表
# PATH_LIST = [input('输入文件地址：').strip("'").strip('"')]


# path_list = [
#     r"train_text\WanJuan1.0part-000036-a894b46e-downsample10x-processed.txt",
#     r"train_text\SkyPile2023-14_zh_head_000_processed.txt",
#     r"train_text\SkyPile2022-40_zh_middle_0011_processed.txt",
#     r"train_text\SkyPile2023-14_zh_middle_0010_processed.txt",
#     r"train_text\ultrafineweb-zh-part-001-of-256-downsample2x.txt"
#     r"train_text\Infinity-Instruct-Gen-00000-of-00015-sft2pretrain-processed.txt",
#     r"train_text\distill_r1_110k_sft2pretrain_processed.txt",
#     # ---pretrain---
#     # r"train_text\SFT\distill_r1_110k_sft_processed.txt",
#     # r"train_text\Beautiful-Chinese-processed.txt",
#     # r"train_text\SFT\Infinity-Instruct-Gen-00000-of-00015-processed.txt"
# ]

# 通过字典形式定义路径和对应的采样比率（旧版：按步长跳行采样，已弃用）
# path_sample_dict = {
#     r"train_text\WanJuan1.0part-000036-a894b46e-downsample30x-processed.txt": 0.025,   # 原步长40 → 1/40 = 2.5%
#     r"train_text\SkyPile2023-14_zh_head_0000_processed.txt": 0.05,                    # 原步长20 → 1/20 = 5%
#     r"train_text\SkyPile2022-40_zh_middle_0011_processed.txt": 0.00833,               # 原步长120 → 1/120 ≈ 0.833%
#     r"train_text\SkyPile2023-14_zh_middle_0010_processed.txt": 0.25,                  # 原步长4 → 1/4 = 25%
#     r"train_text\ultrafineweb-zh-part-001-of-256-downsample2x.txt": 0.25,              # 原步长4 → 1/4 = 25%
#     r"train_text\Infinity-Instruct-Gen-00000-of-00015-sft2pretrain-processed.txt": 0.0125,  # 原步长80 → 1/80 = 1.25%
#     r"train_text\distill_r1_110k_sft2pretrain_processed.txt": 0.001,                  # 原步长1000 → 1/1000 = 0.1%
#     r"train_text\时政文章.txt": 1,
# }

# 通过字典形式定义路径和对应的采样比率（sample_rate）
# sample_rate 含义：
#   - (0, 1)  : 降采样，随机抽取该百分比的行。例如 0.001 表示随机抽取 0.1% 的行
#   - 1       : 不采样，使用全部数据
#   - > 1     : 重复（repeat），将数据重复若干次。例如 3 表示将数据重复 3 倍
path_sample_dict = {
    r"train_text\ultrafineweb-l3-mutistyle-cn-part0.txt": 0.38,
    r"train_text\ultrafineweb-zh-part-001-of-256-downsample8x.txt": 0.4,
    r"train_text\SkyPile2023-14_zh_middle_0010_processed.txt": 1,
    r"train_text\WanJuan1.0part-000036-a894b46e-downsample30x-processed.txt": 0.13,   
    r"train_text\ultrafineweb-l3-mutistyle-en-part0.txt": 0.1,
}

# 设定句子的最大长度
SENTENCE_MAXLEN = 192 + 1
BATCH_SIZE = 1024  # 设置合适的批量大小
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
SPLIT_FROM_SYMBOL = True
OUTPUT_PATH = r"mini_data192hq.txt"
SHUFFLE = True  # 按BATCH_SIZE进行随机打乱
# BOS/EOS 配置常量（可修改）
BOS_TOKEN_STR = ""  # BOS token 字符串
EOS_TOKEN_STR = "<|endoftext|>"   # EOS token 字符串，请根据 tokenizer 中的对应 ID 填写

# 是否在生成的数据中添加 BOS/EOS 标记
ADD_BOS = False   # 是否在每个段落开头添加 BOS token
ADD_EOS = True   # 是否在每个段落结尾添加 EOS token

# # Packed 数据集配置（未实现对应训练代码）
USE_PACKED = False  # 是否使用packed模式（将多个样本拼接成连续序列）
PACKED_SEP_TOKEN = ""  # packed模式下样本间的分隔符


# ---------- 采样比率说明 ----------
# sample_rate in (0, 1) → 降采样：随机抽取对应百分比的行
# sample_rate == 1      → 不采样：使用全部数据
# sample_rate > 1       → 重复（repeat）：将全部数据重复 int(sample_rate) 次
# -----------------------------------

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
assert USE_PACKED == False, "Packed模式未实现对应训练代码"
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


# 打开输入文件和创建输出文件
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
            # 使用 min(num_samples, len(all_lines)) 防止空文件或少量行时 random.sample 抛出 ValueError
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

        # 打开输出文件进行追加写入
        with open(OUTPUT_PATH, "a", encoding="UTF-8") as output_data:
            for i in tqdm(range(0, len(data_lines), BATCH_SIZE)):
                batch = data_lines[i : i + BATCH_SIZE]
                batch = [item.strip("\n") for item in batch]

                # 批量编码
                encodings = tokenizer.encode_batch(batch)

                # 临时存储当前batch的结果
                batch_out_list = []

                if USE_PACKED:
                    # Packed模式：将所有样本拼接成一个长序列
                    all_tokens = []
                    
                    # 添加分隔符的token ID（如果配置了分隔符）
                    sep_token_id = None
                    if PACKED_SEP_TOKEN and PACKED_SEP_TOKEN.strip():
                        sep_token_id = tokenizer.token_to_id(PACKED_SEP_TOKEN)
                        if sep_token_id is None:
                            print(f"警告: packed分隔符 '{PACKED_SEP_TOKEN}' 在 tokenizer 中未找到，将使用空分隔符")
                    else:
                        sep_token_id = None  # 空字符串或仅空格时使用None
                    
                    # 拼接所有样本的tokens
                    for i, encoding in enumerate(encodings):
                        tokens = encoding.ids
                        
                        # 添加BOS（在packed模式下，根据配置决定是否在每个样本前添加BOS）
                        if ADD_BOS and BOS_TOKEN_STR:
                            bos_token_id = tokenizer.token_to_id(BOS_TOKEN_STR)
                            if bos_token_id is not None:
                                if i == 0 or (USE_PACKED and PACKED_ADD_BOS_PER_SAMPLE):  # 第一个样本或每个样本都添加BOS
                                    tokens = [bos_token_id] + tokens
                            else:
                                print(f"警告: BOS token '{BOS_TOKEN_STR}' 在 tokenizer 中未找到，跳过添加")
                        
                        # 添加EOS（每个样本结尾添加）
                        if ADD_EOS and EOS_TOKEN_STR:
                            eos_token_id = tokenizer.token_to_id(EOS_TOKEN_STR)
                            if eos_token_id is not None:
                                tokens = tokens + [eos_token_id]
                            else:
                                print(f"警告: EOS token '{EOS_TOKEN_STR}' 在 tokenizer 中未找到，跳过添加")
                        
                        # 添加到总序列
                        all_tokens.extend(tokens)
                        
                        # 如果不是最后一个样本，添加分隔符
                        if i < len(encodings) - 1 and sep_token_id is not None:
                            all_tokens.append(sep_token_id)
                    
                    # 按PACKED_MAX_LEN切分长序列
                    start_idx = 0
                    while start_idx < len(all_tokens):
                        end_idx = start_idx + SENTENCE_MAXLEN
                        
                        # 如果超过最大长度，截断
                        if end_idx > len(all_tokens):
                            end_idx = len(all_tokens)
                        
                        # 获取当前段落的 token
                        segment_tokens = all_tokens[start_idx:end_idx]
                        
                        # 将 token 转换为原始文本
                        segment_text = tokenizer.decode(
                            segment_tokens, skip_special_tokens=False
                        )
                        
                        # 暂存分割结果
                        batch_out_list.append(segment_text)
                        start_idx = end_idx
                        
                else:
                    # 原有逻辑：每个样本独立处理
                    for encoding in encodings:
                        tokens = encoding.ids  # 确保 tokens 是 token ID 的整数列表

                        # 先在整行编码结果上添加 BOS/EOS（关键修改：移到切分前）
                        if ADD_BOS and BOS_TOKEN_STR:
                            bos_token_id = tokenizer.token_to_id(BOS_TOKEN_STR)
                            if bos_token_id is not None:
                                tokens = [bos_token_id] + tokens
                            else:
                                print(f"警告: BOS token '{BOS_TOKEN_STR}' 在 tokenizer 中未找到，跳过添加")
                        if ADD_EOS and EOS_TOKEN_STR:
                            eos_token_id = tokenizer.token_to_id(EOS_TOKEN_STR)
                            if eos_token_id is not None:
                                tokens = tokens + [eos_token_id]
                            else:
                                print(f"警告: EOS token '{EOS_TOKEN_STR}' 在 tokenizer 中未找到，跳过添加")

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

                            # 获取当前段落的 token
                            segment_tokens = tokens[start_idx:end_idx]

                            # 将 token 转换为原始文本
                            segment_text = tokenizer.decode(
                                segment_tokens, skip_special_tokens=False
                            )

                            # 暂存分割结果，减少文件写入次数
                            batch_out_list.append(segment_text)
                            start_idx = end_idx

                # 将当前batch的结果写入文件
                if SHUFFLE:
                    random.shuffle(batch_out_list)
                output_data.write("\n".join(batch_out_list) + "\n")
                batch_out_list.clear()  # 清空当前batch的结果，以便下一个文件的处理结果不会与当前文件的结果混淆
