"""从 train_text/RAW/Step-SFT-chunk 99.json 中抽取中文语料，转成 ChatML 纯文本。

仿照 extract_zh_sft.py：输出到 train_text/SFT/，每行一段 ChatML 对话。
原始文件为 654MB 的 JSON 数组（每条含 conversations），无语言标记字段，
用 CJK 启发式判定：整段对话中汉字占（汉字+ASCII字母）比例 > CJK_RATIO_THRESHOLD 视为中文。
流式 raw_decode 解析，避免整文件加载内存。
"""
import json
import os
from tqdm import tqdm

INPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "train_text", "RAW", "Step-SFT-chunk 99.json"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "train_text", "SFT", "step_sft_chunk99_zh_sft.txt"
)

CJK_RATIO_THRESHOLD = 0.5


def cjk_ratio(text):
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cjk + latin
    return cjk / total if total else 0.0


def is_chinese(item):
    parts = []
    for m in item["conversations"]:
        parts.append(m.get("content") or m.get("reasoning_content") or "")
    return cjk_ratio("".join(parts)) > CJK_RATIO_THRESHOLD


def to_chatml(item):
    turns = []
    for m in item["conversations"]:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content") or m.get("reasoning_content") or ""
        # 真实换行统一转成字面量 \n，保证每行 = 一条完整对话
        content = (
            content.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")
        )
        turns.append(f"<|im_start|>{role}\\n{content}<|im_end|>\\n")
    # 行内分隔符为字面量 \n，行尾用真实 CRLF（对齐参考文件）
    return "".join(turns) + "\r\n"


def main():
    input_path = os.path.normpath(INPUT_PATH)
    output_path = os.path.normpath(OUTPUT_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    decoder = json.JSONDecoder()
    total = filtered = 0
    buf = ""
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8", newline="\n"
    ) as outfile:
        bar = tqdm(desc="parsing")
        while True:
            chunk = infile.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while True:
                buf = buf.lstrip(" \t\r\n,")
                if buf.startswith("["):  # 跳过数组开头的 '['
                    buf = buf[1:]
                    continue
                if buf.startswith("]"):
                    buf = ""
                    break
                if not buf:
                    break
                try:
                    item, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    break
                buf = buf[idx:]
                total += 1
                bar.update(1)
                if is_chinese(item):
                    outfile.write(to_chatml(item))
                    filtered += 1
        bar.close()

    print(f"共读取 {total} 条，筛选出中文语料 {filtered} 条，写入 {output_path}")


if __name__ == "__main__":
    main()
