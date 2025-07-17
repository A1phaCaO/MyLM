from tqdm import tqdm


STRIP_RETURN_NUMS = 4  # 每隔几个换行才替换

REPLACE_RETURN = 'RETURN'
PATH = repr(input('输入文件地址')).strip("'").strip('"')


data = open(PATH, 'r', encoding='UTF-8')
output_data = open(r'strip_return.txt', 'a', encoding='UTF-8')
data_len = len(list(data))
data.seek(0)

replace_counter = 0 # 当前已替换几个空格

for i, item in enumerate(tqdm(list(data))):
    if replace_counter >= STRIP_RETURN_NUMS:
        output_data.write(item)
        replace_counter = 0
    else:
        replace_counter += 1
        output_data.write(item.replace('\n', REPLACE_RETURN))