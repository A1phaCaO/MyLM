# -*- coding: utf-8 -*-
"""生成 SFT 训练数据：合并多个预处理好的 ChatML 对话文件，输出单文件（每行一个完整对话）。

与 generate_dataset_v3.py（pretrain）的差异：
- 不对对话做固定长度切段：超长对话由 LONG_MODE 切换「截断」/「丢弃」
  （截断时多轮对话优先在最近的 assistant 回合结束后截断使对话闭合，
  无法满足时回退头部截断，回答未闭合处由 SFTTextDataset 兜底）
- 按对话（行）粒度采样 / 去重 / shuffle

产出文件格式（每行一个完整对话）：
<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>

格式假设：消息分隔符用字面 \\n（反斜杠 n）转义，SFTTextDataset 读取时全局还原
为真实换行；若正文内容本身含字面 \\n（如代码/数学文本）也会一并被改写，
这是格式固有歧义，请在预处理侧保证正文不含字面 \\n。

使用：uv run python generate_dataset_sft.py
然后改 continue_training_sft.py 的 TrainingConfig.data_dir 指向输出文件。
所有配置见下方常量区。
"""
import os
import random
from tqdm import tqdm
from tokenizers import Tokenizer

# ======================= 配置（可修改） =======================
TOKENIZER_PATH = r"tokenizer/bbpe_tokenizer_7k_260723_xl.json"

# 输入：预处理好的 SFT 对话文件（每行一个完整 ChatML 对话）与采样比率
# sample_rate 含义：
#   - (0, 1)  : 降采样，随机抽取该百分比的行
#   - 1       : 不采样，使用全部数据
#   - > 1     : 重复（repeat），将数据重复 int(sample_rate) 次
SFT_PATH_DICT = {
    r"train_text\SFT\distill_r1_110k_sft_processed.txt": 0.7,
    r"train_text\SFT\Infinity-Instruct-Gen-00000-of-00015-processed.txt": 1,
    r"train_text\SFT\Infinity-Instruct-5-7-.txt": 1,
    r"train_text\SFT\step_sft_chunk99_zh_sft.txt": 1,
    r"train_text\SFT\Infinity-Instruct-Gen-7MCore-00000-of-00015-processed.txt": 1,
    r"train_text\ultrafineweb-l3-mutistyle-cn-part0.txt": 0.01,
    r"train_text\SFT\step_sft_chunk98_zh_sft.txt": 1,
    r"train_text\SFT\step_sft_chunk97_zh_sft.txt": 1,
}

OUTPUT_PATH = r"data/data_sft512v3.txt"  # 输出文件（每行一个完整对话）
BATCH_SIZE = 2048  # 批量编码大小
SEED = 42          # shuffle 随机种子
SHUFFLE = True     # 全局打乱
DEDUP = True       # 按对话内容去重（SFT 数据重复率高）
MAX_TOKENS = 512   # 超长阈值（对齐训练 seq_max_len）
MIN_TOKENS = 8     # 过短阈值，低于则丢弃
# 超长对话处理方式：
#   "truncate" : 截断超长对话（多轮对话优先在 assistant 回合后截断，见 TRUNCATE_AT_TURN）
#   "drop"     : 直接丢弃
LONG_MODE = "truncate"
# 截断方式（仅 LONG_MODE="truncate" 时生效）：
#   True  : 多轮对话优先在最近的 assistant 回合结束后截断（对话闭合）；
#           无法满足（单轮/首回合即超长）时回退为头部截断
#   False : 一律头部截断（保留前 MAX_TOKENS 个 token）
TRUNCATE_AT_TURN = True
# 超长直接丢弃阈值（仅 LONG_MODE="truncate" 时生效）：
#   总 token 数 > DROP_OVER_TOKENS 的对话直接丢弃，即使能截断也不截断
#   （过长对话截断后仅剩头部，训练价值低；设为 None 禁用）
DROP_OVER_TOKENS = 4096
# 截断后剩余部分处理（仅 LONG_MODE="truncate" 时生效）：
#   超长行在 assistant 回合后截断时，被裁掉的后续内容默认丢弃（后续对话可能
#   依赖前文）；设为 True 时，后续若含完整对话（以 <|im_start|> 开头）则拆分
#   到后续输出行（递归处理，残缺续行仍丢弃）
SPLIT_REMAINDER = False
FILTER_INVALID = False
# ==============================================================

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
TRUNCATE_LONG = LONG_MODE == "truncate"


def split_by_turns(line):
    """按 <|im_start|> 拆出完整回合文本（含前缀标记）。"""
    parts = line.split("<|im_start|>")
    return ["<|im_start|>" + p for p in parts[1:]]


def truncate_line(line, enc):
    """处理一条超长对话，返回 [(text, kind)]；kind: turn / head / keep。

    - turn: 在 assistant 回合结束后截断（对话闭合）；剩余部分默认丢弃，
      SPLIT_REMAINDER 时递归拆分后续完整对话
    - head: 兜底头部截断（单轮/首回合即超长），截断处若在回答中间由
      SFTTextDataset 的 mask 兜底；skip_special_tokens=False 保留 <|im_start|>
      等标记（tokenizers 默认 True 会跳过它们导致输出缺标记）
    """
    if TRUNCATE_AT_TURN:
        turn_texts = split_by_turns(line)
        if len(turn_texts) >= 2:  # 至少 2 个回合
            # 分段编码的 token 数 ≥ 拼接后整体编码（跨段合并只会减少 token），
            # 以分段和做保守估计，选出的组合拼接后必然不超长
            turn_lens = [len(t.ids) for t in tokenizer.encode_batch(turn_texts)]
            prefix, total = [], 0
            for txt, ln in zip(turn_texts, turn_lens):
                if total + ln > MAX_TOKENS:
                    break
                prefix.append(txt)
                total += ln
            if prefix and not prefix[-1].startswith("<|im_start|>assistant"):
                # 停在了 user 回合 → 回退到上一个 assistant 回合
                prefix = prefix[:-1]
            if prefix and prefix[-1].startswith("<|im_start|>assistant"):
                text = "".join(prefix)
                out = [(text, "turn")]
                if SPLIT_REMAINDER:
                    rest = line[len(text):]
                    # 后续内容递归处理：完整对话拆到后续行，残缺续行/残段丢弃
                    if rest:
                        rest_enc = tokenizer.encode(rest)
                        rest_n = len(rest_enc.ids)
                        if rest_n > MAX_TOKENS:
                            out += truncate_line(rest, rest_enc)
                        elif rest_n >= MIN_TOKENS:
                            out.append((rest, "keep"))
                return out
    # 兜底：头部截断到前 MAX_TOKENS 个 token（保留 <|im_start|> 等特殊标记）
    return [(tokenizer.decode(enc.ids[:MAX_TOKENS], skip_special_tokens=False), "head")]


def get_sample_label(sample_rate):
    if sample_rate < 1:
        return f"downsample {sample_rate*100:.2f}%"
    elif sample_rate == 1:
        return "1x (全量)"
    else:
        return f"repeat {int(sample_rate)}x"


# ---------- 统计展示 ----------
print(f"{'文件路径':<50} {'原始大小(MB)':<9} {'采样后大小(MB)':<10} {'占比':<10} {'采样方式':<20}")
print("-" * 120)
total_size_mb = 0
total_sampled_size_mb = 0
for INPUT_PATH in SFT_PATH_DICT.keys():
    sample_rate = SFT_PATH_DICT.get(INPUT_PATH, 1)
    original_size_mb = os.path.getsize(INPUT_PATH) / (1024 * 1024)
    if sample_rate <= 0:
        sampled_size_mb = 0
    elif sample_rate <= 1:
        sampled_size_mb = original_size_mb * sample_rate
    else:
        sampled_size_mb = original_size_mb * int(sample_rate)
    total_size_mb += original_size_mb
    total_sampled_size_mb += sampled_size_mb
for INPUT_PATH in SFT_PATH_DICT.keys():
    sample_rate = SFT_PATH_DICT.get(INPUT_PATH, 1)
    original_size_mb = os.path.getsize(INPUT_PATH) / (1024 * 1024)
    if sample_rate <= 0:
        sampled_size_mb = 0
    elif sample_rate <= 1:
        sampled_size_mb = original_size_mb * sample_rate
    else:
        sampled_size_mb = original_size_mb * int(sample_rate)
    percentage = (sampled_size_mb / total_sampled_size_mb) * 100 if total_sampled_size_mb > 0 else 0
    print(
        f"{INPUT_PATH[:31]+'...'+INPUT_PATH[-20:]:<50} {original_size_mb:<15.2f} "
        f"{sampled_size_mb:<15.2f} {f'{percentage:.1f}%':<10} {get_sample_label(sample_rate):<20}"
    )
print("-" * 120)
print(
    f"共计{len(SFT_PATH_DICT)}个文件".ljust(50)
    + f"{total_size_mb:.1f}MB".ljust(16)
    + f"{total_sampled_size_mb:.1f}MB".ljust(16)
    + "100%".ljust(16)
)
print("=" * 120)

# ---------- 读取 + 采样 + 合法性校验 ----------
all_lines = []  # (来源文件, 行内容)
line_counts = {}  # 采样后每文件行数
for INPUT_PATH in SFT_PATH_DICT.keys():
    sample_rate = SFT_PATH_DICT.get(INPUT_PATH, 1)
    with open(INPUT_PATH, "r", encoding="UTF-8", errors="ignore") as data:
        lines = [l.strip() for l in data]
    lines = [l for l in lines if l]  # 丢弃空行

    if sample_rate < 1:
        num_samples = max(1, min(int(len(lines) * sample_rate), len(lines)))
        lines = random.Random(SEED).sample(lines, num_samples) if num_samples else []
        print(f"[降采样] {INPUT_PATH}: {len(lines)} 行 ({sample_rate*100:.2f}%)")
    elif sample_rate > 1:
        lines = lines * int(sample_rate)
        print(f"[重复] {INPUT_PATH}: {len(lines) // int(sample_rate)} 行 × {int(sample_rate)} = {len(lines)} 行")
    else:
        print(f"[全量] {INPUT_PATH}: {len(lines)} 行")

    # 校验：必须是完整 ChatML 对话（以 im_start 开头且含 assistant 回合）
    if FILTER_INVALID:
        n_before = len(lines)
        lines = [l for l in lines if l.startswith("<|im_start|>") and "<|im_start|>assistant" in l]
        if n_before - len(lines):
            print(f"  !! {n_before - len(lines)} 行不是合法 ChatML 对话，已丢弃")
    all_lines.extend((INPUT_PATH, l) for l in lines)
    line_counts[INPUT_PATH] = len(lines)

print(f"合并后共 {len(all_lines)} 条对话")

# ---------- 去重（保序，保留首个出现的来源文件） ----------
dup_removed = {}
if DEDUP:
    n_before = len(all_lines)
    seen = {}
    for src, l in all_lines:
        if l in seen:
            dup_removed[src] = dup_removed.get(src, 0) + 1
        else:
            seen[l] = (src, l)
    all_lines = list(seen.values())
    print(f"[去重] 移除 {n_before - len(all_lines)} 条重复对话，剩 {len(all_lines)} 条")

# ---------- 超长截断/过滤 + 过短过滤（批量编码统计 token 数） ----------
kept = []                 # (来源文件, 行内容)
kept_count = {}           # 每文件最终保留条数
kept_tokens = {}          # 每文件最终保留 token 数
long_drop = {}            # 每文件超长丢弃数
short_drop = {}           # 每文件过短丢弃数
n_long = 0        # 丢弃的超长对话数
n_trunc = 0       # 截断的超长对话数
n_turn_trunc = 0  # 其中在 assistant 回合边界截断的条数
n_head_trunc = 0  # 其中头部截断的条数
n_split = 0       # SPLIT_REMAINDER 拆出的后续完整对话数
n_short = 0
for i in tqdm(range(0, len(all_lines), BATCH_SIZE), desc="token 统计/处理"):
    batch = all_lines[i : i + BATCH_SIZE]
    for (src, line), enc in zip(batch, tokenizer.encode_batch([l for _, l in batch])):
        n = len(enc.ids)
        if n > MAX_TOKENS:
            if TRUNCATE_LONG and (DROP_OVER_TOKENS is None or n <= DROP_OVER_TOKENS):
                results = truncate_line(line, enc)
                if results:
                    n_trunc += 1
                for text, kind in results:
                    kept.append((src, text))
                    kept_count[src] = kept_count.get(src, 0) + 1
                    kept_tokens[src] = kept_tokens.get(src, 0) + len(tokenizer.encode(text).ids)
                    if kind == "turn":
                        n_turn_trunc += 1
                    elif kind == "head":
                        n_head_trunc += 1
                    else:
                        n_split += 1
            else:
                # 丢弃：LONG_MODE="drop"，或超过 DROP_OVER_TOKENS 过长阈值
                long_drop[src] = long_drop.get(src, 0) + 1
                n_long += 1
        elif n < MIN_TOKENS:
            short_drop[src] = short_drop.get(src, 0) + 1
            n_short += 1
        else:
            kept.append((src, line))
            kept_count[src] = kept_count.get(src, 0) + 1
            kept_tokens[src] = kept_tokens.get(src, 0) + n
if TRUNCATE_LONG:
    over = f"，过长(>{DROP_OVER_TOKENS} tok) 丢弃 {n_long} 条" if DROP_OVER_TOKENS is not None else ""
    split = f"，拆分续行 {n_split} 条" if SPLIT_REMAINDER else ""
    print(f"[处理] 超长(>{MAX_TOKENS} tok) 截断 {n_trunc} 条（回合边界 {n_turn_trunc} / 头部 {n_head_trunc}）{over}{split}，过短(<{MIN_TOKENS} tok) 丢弃 {n_short} 条，保留 {len(kept)} 条")
else:
    print(f"[处理] 超长(>{MAX_TOKENS} tok) 丢弃 {n_long} 条，过短(<{MIN_TOKENS} tok) 丢弃 {n_short} 条，保留 {len(kept)} 条")

# ---------- 处理后最终文件的来源占比展示 ----------
print("=" * 135)
print(f"{'文件路径':<42} {'采样后':<7} {'去重drop':<8} {'超长drop':<8} {'过短drop':<8} "
      f"{'最终保留':<7} {'条数占比':<8} {'token占比':<9} {'保留率':<7}")
print("-" * 135)
total_tokens = sum(kept_tokens.values())
for INPUT_PATH in SFT_PATH_DICT.keys():
    orig = line_counts.get(INPUT_PATH, 0)
    final = kept_count.get(INPUT_PATH, 0)
    tok = kept_tokens.get(INPUT_PATH, 0)
    dup = dup_removed.get(INPUT_PATH, 0)
    ld = long_drop.get(INPUT_PATH, 0)
    sd = short_drop.get(INPUT_PATH, 0)
    pct = final / len(kept) * 100 if kept else 0
    pct_tok = tok / total_tokens * 100 if total_tokens else 0
    keep_rate = final / orig * 100 if orig else 0
    print(
        f"{INPUT_PATH[:27]+'...'+INPUT_PATH[-15:]:<42} {orig:<7} {dup:<8} {ld:<8} {sd:<8} "
        f"{final:<7} {f'{pct:.1f}%':<8} {f'{pct_tok:.1f}%':<9} {f'{keep_rate:.1f}%':<7}"
    )
print("-" * 135)
print(f"合计保留 {len(kept)} 条对话 / {total_tokens:,} tokens（100%）")
print("=" * 135)

# ---------- 全局 shuffle ----------
if SHUFFLE:
    random.Random(SEED).shuffle(kept)
    print(f"[shuffle] seed={SEED}")

# ---------- 写输出 ----------
with open(OUTPUT_PATH, "w", encoding="UTF-8") as f:
    f.write("\n".join(l for _, l in kept) + "\n")
print(f"完成。输出 {OUTPUT_PATH}：{len(kept)} 条对话，"
      f"{os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f}MB")
