"""从 processed_data.jsonl 中筛选中文（langdetect=zh-cn）语料，转成 ChatML 纯文本。

输出：train_text/SFT/processed_data_zh_sft.txt（每行一段 ChatML 对话），
格式与 continue_training_sft.py 的 SFTTextDataset 期望一致：
"<|im_start|>user\\n问题内容<|im_end|>\\n<|im_start|>assistant\\n回答内容<|im_end|>\\n"
"""
import json
import os
from tqdm import tqdm

INPUT_PATH = os.path.join(os.path.dirname(__file__), "processed_data.jsonl")
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "train_text", "SFT",
    "Infinity-Instruct-Gen-7MCore-00000-of-00015-processed.txt",
)

TARGET_LANG = "zh-cn"


def _escape(text):
    # 真实换行统一转成字面量 \n，保证每行 = 一条完整对话
    return text.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")


def to_chatml(conversations):
    turns = []
    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        text = _escape(turn["value"])
        turns.append(f"<|im_start|>{role}\\n{text}<|im_end|>\\n")
    # 行内分隔符为字面量 \n，行尾用真实 CRLF（对齐参考文件）
    return "".join(turns) + "\r\n"


def main():
    output_path = os.path.normpath(OUTPUT_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = filtered = 0
    with open(INPUT_PATH, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8", newline="\n"
    ) as outfile:
        for line in tqdm(infile):
            total += 1
            item = json.loads(line)
            if item.get("langdetect") != TARGET_LANG:
                continue
            outfile.write(to_chatml(item["conversations"]))
            filtered += 1

    print(f"共读取 {total} 条，筛选出中文语料 {filtered} 条，写入 {output_path}")


if __name__ == "__main__":
    main()
