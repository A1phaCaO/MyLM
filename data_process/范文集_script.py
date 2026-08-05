"""
范文集预处理脚本
将范文集_MinerU_processed.txt 处理为适合LLM预训练的格式：
- 每行一个完整的训练样本（一篇作文或一个独立语段）
- 去除markdown标记、多余空行
"""

import re
import os


def read_raw_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """清洗文本"""
    text = re.sub(r"\$\s*([^$]+?)\s*\$", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def classify_header(header: str) -> str:
    """
    分类 ## 标题类型：
    - 'date': 日期标题 (2025.6.12)
    - 'video': 视频语段标题
    - 'date_video': 日期+视频语段合并标题 (2024.12.24 视频语段)
    - 'author': 作者行 (高三（1）班 田寒)
    - 'subsection': 小节标题 (一、美育的意义)
    - 'label': 标签标题 (二高范文 1:)
    - 'title': 普通文章标题
    """
    h = header.strip()
    if not h:
        return "empty"

    # 日期+视频合并 (2025.2.18视频语段, 2024.12.24 视频语段)
    if re.match(r"^\d{4}[\.\-/年]\s*\d{1,2}[\.\-/月]\s*\d{1,2}", h) and "视频" in h:
        return "date_video"

    # 纯日期
    if re.match(r"^\d{4}[\.\-/年]\s*\d{1,2}[\.\-/月]\s*\d{1,2}", h):
        return "date"

    # 视频语段
    if "视频语段" in h or "优秀视频" in h:
        return "video"

    # 作者行
    if re.match(r"^(高一|高二|高三|初中).{0,8}班\s+\S{2,4}$", h):
        return "author"

    # 小节标题
    if re.match(r"^[一二三四五六七八九十]+[、．.\s]", h):
        return "subsection"

    # 标签 (二高范文, # 2024.10.15)
    if re.match(r"^(二高范文|#)", h):
        return "label"

    return "title"


def has_author_suffix(text: str) -> bool:
    """判断段落是否以署名结尾，支持多种格式"""
    # 匹配 ——班级 姓名 或 ——姓名 或 —姓名
    return bool(re.search(r"[\u2014\-]{1,2}\s*(?:(?:高一|高二|高三|初中).{0,10}$|[\u4e00-\u9fff]{2,4}\s*$)", text.strip()))


def format_as_line(text: str) -> str:
    """将多行文本压缩为单行"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text).strip()
    return text.replace("\n", "\\n")


def parse_document(raw_text: str) -> list[str]:
    """解析范文集，返回训练样本列表。"""
    lines = raw_text.split("\n")

    # 提取所有 ## 行及其位置，构建 sections = [(type, header, body_lines), ...]
    header_positions = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("##"):
            header_text = stripped.lstrip("#").strip()
            header_positions.append((i, header_text))

    sections = []
    for idx, (pos, header) in enumerate(header_positions):
        end_pos = header_positions[idx + 1][0] if idx + 1 < len(header_positions) else len(lines)
        body_lines = [lines[j].strip() for j in range(pos + 1, end_pos) if lines[j].strip()]
        htype = classify_header(header)
        sections.append((htype, header, body_lines))

    samples = []
    i = 0

    while i < len(sections):
        htype, header, body = sections[i]

        # ==== 跳过: date, label, date_video (date_video 按 video 处理) ====
        if htype == "date":
            i += 1
            continue

        if htype == "label":
            i += 1
            continue

        # ==== 视频语段区域 ====
        if htype in ("video", "date_video"):
            all_paragraphs = list(body)
            i += 1
            # 收集后续段落，直到遇到 日期/视频/小节/标签/文章标题
            while i < len(sections):
                nt, nh, nb = sections[i]
                if nt in ("date", "video", "date_video", "subsection", "label"):
                    break
                # 遇到文章标题（title后面跟着author）停止
                if nt == "title" and i + 1 < len(sections) and sections[i + 1][0] == "author":
                    break
                # 遇到独立作者行停止
                if nt == "author":
                    break
                if nb:
                    all_paragraphs.extend(nb)
                i += 1

            # 按署名拆分：每个以 ——署名 结尾的段落是独立样本
            current_para = []
            for para in all_paragraphs:
                current_para.append(para)
                if has_author_suffix(para):
                    text = clean_text("\n".join(current_para))
                    if len(text) > 30:
                        samples.append(format_as_line(text))
                    current_para = []
            if current_para:
                text = clean_text("\n".join(current_para))
                if len(text) > 30:
                    samples.append(format_as_line(text))
            continue

        # ==== 小节标题 ====
        if htype == "subsection":
            section_paras = list(body)
            i += 1
            while i < len(sections):
                nt, nh, nb = sections[i]
                if nt in ("date", "video", "date_video", "subsection", "label"):
                    break
                if nt == "title" and i + 1 < len(sections) and sections[i + 1][0] == "author":
                    break
                if nt == "author":
                    break
                # 小节内可能有子标题
                if nh:
                    section_paras.append(nh)
                if nb:
                    section_paras.extend(nb)
                i += 1

            # 按署名拆分
            current_para = []
            for para in section_paras:
                current_para.append(para)
                if has_author_suffix(para):
                    text = f"{header}\n" + clean_text("\n".join(current_para))
                    text = clean_text(text)
                    if len(text) > 30:
                        samples.append(format_as_line(text))
                    current_para = []
            if current_para:
                text = f"{header}\n" + clean_text("\n".join(current_para))
                text = clean_text(text)
                if len(text) > 30:
                    samples.append(format_as_line(text))
            continue

        # ==== 完整文章: title + author + body ====
        if htype == "title" and i + 1 < len(sections) and sections[i + 1][0] == "author":
            title = header
            author = sections[i + 1][1]
            # 作者的body可能包含正文（如果正文紧跟在作者后面）
            author_body = sections[i + 1][2]
            i += 2

            # 收集正文：作者的body + 后续无标题(empty) section
            content_parts = list(author_body)
            while i < len(sections) and sections[i][0] == "empty":
                _, _, nb = sections[i]
                if nb:
                    content_parts.extend(nb)
                i += 1

            if content_parts:
                full_text = f"{title}\n{author}\n" + "\n\n".join(content_parts)
                full_text = clean_text(full_text)
                if len(full_text) > 30:
                    samples.append(format_as_line(full_text))
            continue

        # ==== 独立作者行（文章没有单独标题） ====
        if htype == "author":
            author = header
            # body 可能直接包含正文
            content_parts = list(body)
            i += 1
            # 继续收集后续无标题section
            while i < len(sections) and sections[i][0] == "empty":
                _, _, nb = sections[i]
                if nb:
                    content_parts.extend(nb)
                i += 1

            if content_parts:
                full_text = f"{author}\n" + "\n\n".join(content_parts)
                full_text = clean_text(full_text)
                if len(full_text) > 30:
                    samples.append(format_as_line(full_text))
            continue

        # ==== 兜底：其他有标题的section ====
        if htype == "title" or htype == "empty":
            parts = []
            if header:
                parts.append(header)
            parts.extend(body)
            i += 1
            # 也收集紧随其后的无标题section
            while i < len(sections) and sections[i][0] == "empty":
                _, _, nb = sections[i]
                if nb:
                    parts.extend(nb)
                i += 1

            if parts:
                text = clean_text("\n\n".join(parts))
                if len(text) > 30:
                    samples.append(format_as_line(text))
            continue

        # 不应到达这里
        i += 1

    return samples


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "..", "train_text", "范文集_MinerU_processed.txt")

    input_path = input(f"请输入输入文件路径（默认: {default_input}）：").strip().strip("'").strip('"')
    if not input_path:
        input_path = default_input

    input_path = os.path.normpath(input_path)
    if not os.path.exists(input_path):
        print(f"错误：文件不存在 - {input_path}")
        return

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, f"{base_name}_pretrain.txt")

    print(f"读取文件: {input_path}")
    raw_text = read_raw_text(input_path)

    print("解析文章...")
    samples = parse_document(raw_text)
    print(f"共解析出 {len(samples)} 个训练样本")

    lengths = [len(s) for s in samples]
    total_chars = sum(lengths)
    print(f"样本字符数统计:")
    print(f"  总计: {total_chars}")
    print(f"  最短: {min(lengths) if lengths else 0}")
    print(f"  最长: {max(lengths) if lengths else 0}")
    print(f"  平均: {total_chars // max(len(samples), 1)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample + "\n")

    print(f"\n已保存到: {output_path}")
    print(f"共 {len(samples)} 行，每行一个训练样本")

    print("\n--- 样本预览（前5个） ---")
    for idx, s in enumerate(samples[:5]):
        preview = s[:100].replace("\\n", " | ")
        print(f"  [{idx+1:2d}] ({len(s):5d}字) {preview}...")
    print(f"\n--- 样本预览（后3个） ---")
    for idx, s in enumerate(samples[-3:]):
        real_idx = len(samples) - 3 + idx + 1
        preview = s[:100].replace("\\n", " | ")
        print(f"  [{real_idx:2d}] ({len(s):5d}字) {preview}...")


if __name__ == "__main__":
    main()
