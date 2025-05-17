from tqdm import tqdm
from tokenizers import Tokenizer

# 加载tokenizer
TOKENIZER_PATH = r"bpe_tokenizer_6k_0517.json" 
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# 输入文件地址
PATH = input('输入文件地址：').strip("'").strip('"')

# 设定句子的最大长度和数据集下采样率
SENTENCE_MAXLEN = 192
DATASET_DOWNSAMPLE = 1
BATCH_SIZE = 128  # 设置合适的批量大小
# 定义分隔符和是否从符号位置开始切分句子的标志
SPLIT_SYMBOL = ('。', '，', '？', '；', '！')
SPLIT_FROM_SYMBOL = True

OUTPUT_PATH = r'data.txt'

# 打开输入文件和创建输出文件
data = open(PATH, 'r', encoding='UTF-8')
output_data = open(OUTPUT_PATH, 'a', encoding='UTF-8')

# 获取数据长度并重置文件读取位置
data_len = len(list(data))
data.seek(0)

# 初始化存储输入输出数据的列表
out_list = []

# 批量处理数据

data_lines = list(data)[::DATASET_DOWNSAMPLE]

for i in tqdm(range(0, len(data_lines), BATCH_SIZE)):
    batch = data_lines[i:i + BATCH_SIZE]
    batch = [item.strip('\n') for item in batch]
    
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
            segment_text = tokenizer.decode(segment_tokens)
            
            # 暂存分割结果，减少文件写入次数
            batch_out_list.append(segment_text)
            start_idx = end_idx
    
    # 将当前batch的结果写入文件
    output_data.write('\n'.join(batch_out_list) + '\n')

# 关闭文件
data.close()
output_data.close()