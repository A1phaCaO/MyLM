# 为txt文件每行添加后缀
import argparse
from tqdm import tqdm

def add_postfix_to_lines(input_file, output_file, postfix):
    """为文本文件每一行添加指定后缀"""
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile):
                # 去除可能存在的换行符后再添加后缀
                outfile.write(line.strip() + postfix + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为文本文件每一行添加指定后缀")
    parser.add_argument("--input_file", type=str, default=r"train_text\WanJuan1.0part-000036-a894b46e-downsample10x-processed.txt", help="输入文件路径")
    parser.add_argument("--output_file", type=str, default=r"train_text\WanJuan1.0part-000036-a894b46e-downsample10x-processed2.txt", help="输出文件路径")
    parser.add_argument("--postfix", type=str, default="[END]", help="要添加的后缀（默认值为_default）")
    
    args = parser.parse_args()
    add_postfix_to_lines(args.input_file, args.output_file, args.postfix)