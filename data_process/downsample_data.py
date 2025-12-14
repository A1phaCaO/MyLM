from tqdm import tqdm
import os

# 获取输入参数
input_path = input("请输入输入文件路径：").strip("'").strip('"')
# 自动生成输出路径
base_name = os.path.basename(input_path)  # 获取文件名
name, ext = os.path.splitext(base_name)  # 分离文件名和扩展名
output_dir = os.path.dirname(input_path)  # 获取文件所在目录
output_path = os.path.join(output_dir, f"{name}_downsample{ext}")  # 生成新的输出路径

DOWNSAMPLE_STEP = 2

# 逐行读取并降采样处理
with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for i, line in enumerate(tqdm(infile)):
        if i % DOWNSAMPLE_STEP == 0:
            outfile.write(line)
