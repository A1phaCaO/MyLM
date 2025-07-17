import json
from tqdm import tqdm
import argparse

def extract_content_from_jsonl(file_path, field_name='content'):
    """
    读取jsonl文件，提取指定字段内容
    :param file_path: jsonl文件路径
    :param field_name: 要提取的字段名称，默认'content'
    :return: 指定字段内容列表
    """
    contents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if line.strip():  # 跳过空行
                try:
                    data = json.loads(line)
                    contents.append(data.get(field_name, ''))
                except json.JSONDecodeError:
                    print(f"警告：跳过无法解析的行: {line[:50]}...")
    return contents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从JSONL文件中提取指定字段内容')
    parser.add_argument('--input', default=r"train_text\distill_r1_110k_sft_downsample.jsonl", help='输入的JSONL文件路径')
    parser.add_argument('--output', default=r"Train_text\processed_text.jsonl", help='输出文本文件路径')
    parser.add_argument('--field', default='instruction', help='要提取的字段名称（默认：content）')
    
    args = parser.parse_args()

    results = extract_content_from_jsonl(args.input, args.field)

    # 打印前3个结果验证
    for i, content in enumerate(results[:3], 1):
        print(f"\n=== 第{i}个{args.field}内容 ===\n{content[:200]}...")  # 截断显示前200字符

    # 将结果写入文件
    with open(args.output, 'w', encoding='utf-8') as f:
        for content in results:
            f.write(content.replace('\n', '\\n') + '\n')