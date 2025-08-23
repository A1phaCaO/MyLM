from tqdm import tqdm
from tokenizers import Tokenizer
import codecs

# 加载tokenizer
TOKENIZER_PATH = r"bpe_tokenizer_6k_0724_ChatML.json"
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)


# 将输入路径按逗号分割成列表
# PATH_LIST = [input('输入文件地址：').strip("'").strip('"')]
PATH_LIST = [
    r"train_text\WanJuan1.0part-000036-a894b46e-downsample10x-processed.txt",
    r"train_text\Infinity-Instruct-Gen-00000-of-00015-sft2pretrain-processed.txt",
    r"train_text\distill_r1_110k_sft2pretrain_processed.txt",
    r"train_text\SkyPile2023-14_zh_head_0000_processed.txt",
    r"train_text\SkyPile2022-40_zh_middle_0011_processed.txt",
    r"train_text\SkyPile2023-14_zh_middle_0010_processed.txt",
    # ---pretrain---
    # r"train_text\SFT\distill_r1_110k_sft_processed.txt",
    # r"train_text\Beautiful-Chinese-processed.txt",
    # r"train_text\SFT\Infinity-Instruct-Gen-00000-of-00015-processed.txt"

]

# 设定句子的最大长度和数据集下采样率
SENTENCE_MAXLEN = 192+1
DATASET_DOWNSAMPLE= [7, 2, 2, 7, 3, 2]
# DATASET_DOWNSAMPLE = [1, 1]
BATCH_SIZE = 512  # 设置合适的批量大小
# 定义分隔符和是否从符号位置开始切分句子的标志
SPLIT_SYMBOL = ("。", "，", "？", "；", "！")
SPLIT_FROM_SYMBOL = False
OUTPUT_PATH = r"data_test_ChatML.txt"


# 打开输入文件和创建输出文件
for i, INPUT_PATH in enumerate(PATH_LIST):
    # 每个文件单独处理
    with open(INPUT_PATH, "r", encoding="UTF-8", errors='ignore') as data, open(
        OUTPUT_PATH, "a", encoding="UTF-8"
    ) as output_data:
        # 获取数据长度并重置文件读取位置
        try:
            data_len = len(list(data))
        except:
            # 如果第一次读取失败，尝试用latin-1重新打开文件
            data.close()
            data = codecs.open(INPUT_PATH, "r", encoding="latin-1", errors='ignore')
            data_len = len(list(data))
        data.seek(0)

        # 初始化存储输入输出数据的列表
        out_list = []

        # 批量处理数据

        # 尝试用UTF-8读取，如果失败则用latin-1读取
        try:
            data_lines = list(data)[::DATASET_DOWNSAMPLE[i]]
        except:
            data.close()
            data = codecs.open(INPUT_PATH, "r", encoding="latin-1", errors='ignore')
            data_lines = list(data)[::DATASET_DOWNSAMPLE[i]]

        for i in tqdm(range(0, len(data_lines), BATCH_SIZE)):
            batch = data_lines[i : i + BATCH_SIZE]
            batch = [item.strip("\n") for item in batch]

            # 批量编码
            encodings = tokenizer.encode_batch(batch)

            # 临时存储当前batch的结果
            batch_out_list = []

            for encoding in encodings:
                tokens = encoding.ids  # 确保 tokens 是 token ID 的整数列表

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
            output_data.write("\n".join(batch_out_list) + "\n")
            batch_out_list.clear()  # 清空当前batch的结果，以便下一个文件的处理结果不会与当前文件的结果混淆

# 关闭文件（实际上with语句已经自动关闭）
data.close()
output_data.close()