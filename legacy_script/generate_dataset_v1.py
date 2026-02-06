from tqdm import tqdm
"""
    将输入文件中的句子按照规则切分为多个句子，并保存到输出文件中。
"""
# 输入文件地址
PATH = repr(input('输入文件地址')).strip("'").strip('"')

# 设定句子的最大、最小长度和数据集下采样率
SENTENCE_MAXLEN = 192
SENTENCE_MINLEN = 192
DATASET_DOWNSAMPLE = 1
SENTENCE_LEN_STEP = 1

# 定义分隔符和是否从符号位置开始切分句子的标志
SPLIT_SYMBOL = ('。', '，', '？', '；', '！')

SPLIT_FROM_SYMBOL = True
RETAIN_SYMBOL = False

OUTPUT_PATH = r'data.txt'


# 打开输入文件和创建输出文件W
data = open(PATH, 'r', encoding='UTF-8')
output_data = open(OUTPUT_PATH, 'a', encoding='UTF-8')

# 获取数据长度并重置文件读取位置
data_len = len(list(data))
data.seek(0)

# 初始化存储输入输出数据的列表
out_list = []



# 对文件中的每一项进行处理
for i, item in enumerate(tqdm(list(data)[::DATASET_DOWNSAMPLE])):
    item = item.strip('\n')
    
    # 切分句子
    word_list = list(item)

    # word_list.insert(0, 'START')
    # word_list.append('END')
    list_len = len(word_list)

    # 根据句子长度生成不同长度的切分
    sentence_len = SENTENCE_MINLEN
    if sentence_len >= list_len:
        sentence_len = list_len-1
    while sentence_len <= SENTENCE_MAXLEN:
        j = 0
        while j < list_len-sentence_len:
            # 以sentence_len+1为宽的窗口在句子中滑动 步长为窗口宽度
            bias = 0 # 窗口切分的偏置
            is_find = False
            if SPLIT_FROM_SYMBOL:
                for symbol in SPLIT_SYMBOL:
                    while word_list[j+sentence_len+bias] != symbol:
                        if j+sentence_len+bias <= 0 or sentence_len+1 == list_len or sentence_len + bias <= 0:
                            bias = 0
                            is_find = True
                            break
                        bias -= 1
                    if is_find:
                        break
                    
            if RETAIN_SYMBOL:
                bias -= 1 # 保留符号
            out_list.append(word_list[j:j+sentence_len+bias+1])
            j += sentence_len + bias + 1
        
        sentence_len += SENTENCE_LEN_STEP
        if sentence_len >= list_len:
            break

    # 将切分结果写入输出文件
    for j in range(len(out_list)):
        out_str = ' '.join(out_list[j])
        output_data.write(out_str + '\n')
    out_list = []


# 关闭文件
data.close()
output_data.close()