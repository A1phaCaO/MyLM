"""
自定义 BPE 训练器 - 不使用 tokenizers.trainers.BpeTrainer
从头实现 BPE 合并算法，仅使用 tokenizers 库进行预处理和格式转换
严格对齐 train_tokenizer.py 的所有设置
"""
from collections import defaultdict
from tokenizers import (
    normalizers,
    models,
    pre_tokenizers,
    processors,
    decoders,
    Tokenizer,
    Regex,
)
from tqdm import tqdm

# ========== 配置（与 train_tokenizer.py 完全一致）==========
BBPE = False
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|beginoftext|>",
    "<|pad|>",
    "<|unk|>",
    "<|im_end|>",
    "<|im_start|>",
]
PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
VOCAB_SIZE = 7168
MIN_FREQUENCY = 2
LIMIT_ALPHABET = 65535  # 初始字符词表大小上限（不含特殊 token），保证留有 BPE 合并空间
FILE_PATH = r"train_text\merged.txt"
OUTPUT_PATH = r"tokenizer/bpe_tokenizer_7k_260724_xl.json"


def build_preprocessor():
    """构建归一化器和预分词器（按 BBPE 分支，与 train_tokenizer.py 一致）"""
    if BBPE:
        normalizer = normalizers.NFC()
        pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                Regex(PRETOKENIZE_REGEX),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    else:
        normalizer = normalizers.NFKD()
        pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    return normalizer, pre_tokenizer


def count_word_frequencies(normalizer, pre_tokenizer, file_path):
    """逐行流式读取文件，统计预分词后的词频（内存高效）"""
    import os
    import sys

    word_freq = defaultdict(int)
    file_size = os.path.getsize(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        with tqdm(total=file_size, desc="统计词频", unit="B", unit_scale=True,
                  file=sys.stdout, mininterval=2.0) as pbar:
            for line in f:
                line_bytes = len(line.encode("utf-8"))
                pbar.update(line_bytes)
                line = line.strip()
                if not line:
                    continue
                normalized = normalizer.normalize_str(line)
                pre_tokenized = pre_tokenizer.pre_tokenize_str(normalized)
                for token, _ in pre_tokenized:
                    if token:
                        word_freq[token] += 1
    return word_freq


def build_initial_vocab(word_freq, special_tokens, limit_alphabet):
    """构建初始词表：特殊 token + initial_alphabet + 高频字符（受 limit_alphabet 限制）

    limit_alphabet 控制初始字符词表大小上限（不含特殊 token），确保留有 BPE 合并空间。
    initial_alphabet 中的字符始终保留，语料字符按频率降序填充剩余配额。
    与 tokenizers.trainers.BpeTrainer 的 limit_alphabet 行为对齐。
    """
    import sys
    import string
    from collections import defaultdict

    vocab = {}
    for token in special_tokens:
        vocab[token] = len(vocab)

    # BBPE=False 时强制加入初始字母表（与 train_tokenizer.py 一致），即使语料中未出现
    initial_alphabet = set()
    if not BBPE:
        initial_alphabet = set(list(string.ascii_letters) + list(string.digits) + ["\n"])

    # 统计语料中每个字符的频率（按词频加权）
    char_freq = defaultdict(int)
    words = list(word_freq.keys())
    with tqdm(total=len(words), desc="统计字符频率", file=sys.stdout, mininterval=2.0) as pbar:
        for word in words:
            freq = word_freq[word]
            for char in word:
                char_freq[char] += freq
            pbar.update(1)

    # 构建 alphabet：initial_alphabet 优先，语料字符按频率降序填充至 limit_alphabet
    alphabet = set(initial_alphabet)
    corpus_chars = sorted(
        set(char_freq.keys()) - initial_alphabet,
        key=lambda c: char_freq[c],
        reverse=True,
    )
    remaining_slots = max(0, limit_alphabet - len(alphabet))
    for char in corpus_chars[:remaining_slots]:
        alphabet.add(char)

    corpus_char_set = set(char_freq.keys())
    n_total_corpus = len(corpus_char_set)
    n_kept_corpus = len(alphabet & corpus_char_set)
    n_filtered = n_total_corpus - n_kept_corpus
    if n_filtered > 0:
        print(f"  [limit_alphabet={limit_alphabet}] 语料字符 {n_total_corpus} 个，"
              f"保留 {n_kept_corpus} 个，过滤 {n_filtered} 个低频字符", flush=True)

    # 按 Unicode 码点排序加入词表（与原实现一致）
    for char in sorted(alphabet, key=ord):
        if char not in vocab:
            vocab[char] = len(vocab)

    return vocab


def train_bpe(word_freq, vocab, vocab_size, min_frequency):
    """执行 BPE 合并训练（固定 pair_table + 链表 + 增量更新，内存固定不增长）"""
    import numpy as np
    import sys

    def build_links(lens, total):
        """由每词长度构建同词内前后指针链表；用 arange + 词边界置 -1，
        避免 np.where 产生 ~total 大小的 int64 临时数组，消除初始化阶段内存尖峰"""
        nxt = np.arange(1, total + 1, dtype=np.int32)
        prv = np.arange(-1, total - 1, dtype=np.int32)
        ends = np.cumsum(lens, dtype=np.int64) - 1
        nxt[ends] = -1
        prv[ends[:-1] + 1] = -1
        return nxt, prv

    # 预计算 word_lens 和 freqs（用 fromiter 避免临时 list）
    words = list(word_freq.keys())
    n_words = len(words)
    word_lens = np.fromiter((len(w) for w in words), dtype=np.int32, count=n_words)
    freqs = np.fromiter((word_freq[w] for w in words), dtype=np.int64, count=n_words)
    word_freq.clear()

    token_to_id = dict(vocab)
    id_to_token = {v: k for k, v in vocab.items()}
    next_id = len(vocab)

    # 注意：不在此处添加初始词表之外的字符。limit_alphabet 限制掉的稀有字符
    # 将在构建 flat_symbols 时被过滤（id == -1），不参与 BPE 训练。

    # 若初始词表已达到/超过目标大小，则无需任何合并（字母表始终完整保留，
    # 与 HF BpeTrainer 行为一致）。BBPE=False 且语料字符数很多时可能触发。
    if vocab_size - len(vocab) <= 0:
        print(f"初始词表大小 {len(vocab)} 已达到或超过目标 {vocab_size}，跳过合并", flush=True)
        return []

    # 构建 码点 -> int16 token_id 查表
    # 字节级(BBPE)：字符均在 BMP，utf-16-le 每字符 2 bytes，表 65536 entries=128KB
    # 字符级：字符可能落在补充平面(码点≥65536)，用 utf-32-le + 全 Unicode 码点表(2.2MB)
    if BBPE:
        table_size, char_encoding, code_dtype = 65536, "utf-16-le", np.uint16
    else:
        table_size, char_encoding, code_dtype = 0x110000, "utf-32-le", np.uint32
    char_id_table = np.full(table_size, -1, dtype=np.int16)
    for c, tid in token_to_id.items():
        if len(c) == 1:
            char_id_table[ord(c)] = tid
    del token_to_id

    # 分批构建 all_symbols（int16，避免 all_text/text_bytes 大临时对象）
    # 不在词表中的字符 id 为 -1（由 limit_alphabet 限制），后续过滤
    total_len = int(word_lens.sum())
    all_symbols = np.empty(total_len, dtype=np.int16)
    offset = 0
    batch_size = 50000
    for i in range(0, n_words, batch_size):
        batch = words[i:i+batch_size]
        batch_text = ''.join(batch)
        batch_bytes = batch_text.encode(char_encoding)
        batch_codes = np.frombuffer(batch_bytes, dtype=code_dtype)
        batch_ids = char_id_table[batch_codes]
        n = len(batch_ids)
        all_symbols[offset:offset+n] = batch_ids
        offset += n
        del batch, batch_text, batch_bytes, batch_codes, batch_ids
    del char_id_table, words

    # 过滤被 limit_alphabet 排除的稀有字符（id == -1）
    valid_mask = all_symbols != -1
    n_filtered = int(total_len - int(valid_mask.sum()))
    if n_filtered > 0:
        print(f"  [limit_alphabet] 过滤掉 {n_filtered} 个稀有字符位置，"
              f"剩余 {int(valid_mask.sum())} 个", flush=True)

    # 计算过滤后每词长度（用 reduceat 避免构建 total_len 大小的 word_id_full 数组）
    word_starts = np.cumsum(word_lens, dtype=np.int64) - word_lens
    filtered_lens = np.add.reduceat(valid_mask, word_starts).astype(np.int32)
    del word_starts

    flat_symbols = all_symbols[valid_mask]
    del all_symbols, valid_mask

    word_id = np.repeat(np.arange(n_words, dtype=np.int32), filtered_lens)
    word_lens = filtered_lens

    n_positions = len(flat_symbols)
    # 不存 alive 数组，用 prev_idx == -2 (DEAD) 标记死亡位置，省 559MB
    # 初始化链表（仅同词内相邻位置建立前后指针）
    next_idx, prev_idx = build_links(word_lens, n_positions)

    # pair_table: 固定大小 1D 数组，vocab_size² × 4 bytes (int32) = 205MB
    # 索引: pair_table[id1 * vocab_size + id2]
    pair_table = np.zeros(vocab_size * vocab_size, dtype=np.int32)

    # 初始统计（分块累加，避免 ~n_positions 的 int64/float64 临时数组导致内存尖峰）
    # max_id_init 小（字节级字符 ≤ 256），pair_counts_init 表本身很小
    max_id_init = next_id
    pair_counts_init = np.zeros(max_id_init * max_id_init, dtype=np.float64)
    count_chunk = 4_000_000
    for c_start in range(0, n_positions, count_chunk):
        c_end = min(c_start + count_chunk, n_positions)
        nxt = next_idx[c_start:c_end]
        local_first = np.flatnonzero(nxt != -1)
        if len(local_first) == 0:
            continue
        second_pos = nxt[local_first]
        pf = flat_symbols[c_start:c_end][local_first].astype(np.int64)
        ps = flat_symbols[second_pos].astype(np.int64)
        pfreq = freqs[word_id[c_start:c_end][local_first]].astype(np.float64)
        pair_encoded = pf * max_id_init + ps
        pair_counts_init += np.bincount(pair_encoded, weights=pfreq,
                                        minlength=max_id_init * max_id_init)
        del nxt, local_first, second_pos, pf, ps, pfreq, pair_encoded

    # 填充 pair_table（向量化）
    nonzero_idx = np.flatnonzero(pair_counts_init > 0)
    id1_arr = (nonzero_idx // max_id_init).astype(np.int64)
    id2_arr = (nonzero_idx % max_id_init).astype(np.int64)
    indices = id1_arr * vocab_size + id2_arr
    pair_table[indices] = pair_counts_init[nonzero_idx].astype(np.int32)
    del pair_counts_init, nonzero_idx, id1_arr, id2_arr, indices

    merges = []
    target_merges = vocab_size - len(vocab)
    n_dead = 0
    compaction_threshold = max(n_positions // 5, 100000)

    with tqdm(total=target_merges, desc="BPE 合并", file=sys.stdout, mininterval=2.0) as pbar:
        while len(merges) < target_merges:
            # 找最大 pair（np.argmax 扫描固定 205MB 表，约 50ms）
            best_idx = int(np.argmax(pair_table))
            best_count = int(pair_table[best_idx])
            if best_count < min_frequency:
                print(f"\n最高频率 {best_count} < min_frequency {min_frequency}，停止训练", flush=True)
                break

            A = best_idx // vocab_size
            B = best_idx % vocab_size

            merged_token = id_to_token[A] + id_to_token[B]
            # 若 merged_token 已存在则复用已有 ID，避免 next_id 越界
            if merged_token in vocadb:
                merged_id = vocab[merged_token]
            else:
                merged_id = next_id
                next_id += 1
                id_to_token[merged_id] = merged_token
                vocab[merged_token] = merged_id
            merges.append((id_to_token[A], id_to_token[B]))

            # 查找合并位置：flat_symbols == A 且 next == B
            a_positions = np.flatnonzero(flat_symbols == A)
            if len(a_positions) == 0:
                pair_table[best_idx] = 0
                pbar.update(1)
                continue

            a_next = next_idx[a_positions]
            valid_mask = a_next >= 0
            a_positions = a_positions[valid_mask]
            a_next = a_next[valid_mask]
            next_syms = flat_symbols[a_next]
            merge_positions = a_positions[next_syms == B]

            if len(merge_positions) == 0:
                pair_table[best_idx] = 0
                pbar.update(1)
                continue

            b_positions = next_idx[merge_positions]
            prev_positions = prev_idx[merge_positions]
            after_b = next_idx[b_positions]
            merge_weights = freqs[word_id[merge_positions]].astype(np.int64)

            # 从 pair_table[A, B] 中减去已合并的权重
            pair_table[best_idx] = 0

            # 计算增量（考虑重叠合并：A B A B 的情况）
            # --- 后邻居（B → after_b）---
            has_after = after_b >= 0
            if has_after.any():
                after_pos_v = after_b[has_after]
                after_syms = flat_symbols[after_pos_v]
                after_w = merge_weights[has_after]

                # 检测 after_b 是否是 merge_position（重叠）
                after_is_merge = np.isin(after_pos_v, merge_positions)

                # 不重叠的：after_b 不是 merge_position
                not_overlap = ~after_is_merge
                if not_overlap.any():
                    syms_no = after_syms[not_overlap]
                    w_no = after_w[not_overlap]
                    uai, ua_inv = np.unique(syms_no, return_inverse=True)
                    ua_sums = np.bincount(ua_inv, weights=w_no.astype(np.float64))
                    # 向量化增量更新
                    ua_sums_i32 = ua_sums.astype(np.int32)
                    idx_sub = B * vocab_size + uai.astype(np.int64)
                    idx_add = merged_id * vocab_size + uai.astype(np.int64)
                    pair_table[idx_sub] -= ua_sums_i32
                    pair_table[idx_add] += ua_sums_i32

                # 重叠的：after_b 是 merge_position → 新 pair 是 (merged, merged)
                if after_is_merge.any():
                    w_ov = after_w[after_is_merge]
                    total_w_ov = int(w_ov.sum())
                    pair_table[B * vocab_size + A] -= total_w_ov
                    pair_table[merged_id * vocab_size + merged_id] += total_w_ov

            # --- 前邻居（prev → A）---
            has_prev = prev_positions >= 0
            if has_prev.any():
                prev_pos_v = prev_positions[has_prev]
                prev_syms = flat_symbols[prev_pos_v]
                prev_w = merge_weights[has_prev]

                # 检测 prev 是否是 b_position（重叠）
                prev_is_b = np.isin(prev_pos_v, b_positions)

                # 只处理不重叠的：prev 不是 b_position
                not_overlap = ~prev_is_b
                if not_overlap.any():
                    syms_no = prev_syms[not_overlap]
                    w_no = prev_w[not_overlap]
                    upi, up_inv = np.unique(syms_no, return_inverse=True)
                    up_sums = np.bincount(up_inv, weights=w_no.astype(np.float64))
                    # 向量化增量更新
                    up_sums_i32 = up_sums.astype(np.int32)
                    idx_sub = upi.astype(np.int64) * vocab_size + A
                    idx_add = upi.astype(np.int64) * vocab_size + merged_id
                    pair_table[idx_sub] -= up_sums_i32
                    pair_table[idx_add] += up_sums_i32

            # 更新 flat_symbols：A 位置 -> merged_id
            flat_symbols[merge_positions] = merged_id

            # 更新链表
            has_prev_lk = prev_positions >= 0
            if has_prev_lk.any():
                next_idx[prev_positions[has_prev_lk]] = merge_positions[has_prev_lk]

            has_after_lk = after_b >= 0
            if has_after_lk.any():
                prev_idx[after_b[has_after_lk]] = merge_positions[has_after_lk]

            next_idx[merge_positions] = after_b
            next_idx[b_positions] = -1
            prev_idx[b_positions] = -2  # DEAD 标记（替代 alive 数组）
            n_dead += len(b_positions)

            # 定期压缩
            if n_dead > compaction_threshold:
                alive_idx = np.flatnonzero(prev_idx != -2)
                flat_symbols = flat_symbols[alive_idx]
                word_id = word_id[alive_idx]
                n_positions = len(flat_symbols)
                n_dead = 0
                del alive_idx
                # 从压缩后的 word_id 重建链表（bincount 求每词长度，内存高效）
                comp_lens = np.bincount(word_id, minlength=n_words)
                next_idx, prev_idx = build_links(comp_lens, n_positions)
                del comp_lens
                compaction_threshold = max(n_positions // 5, 100000)

            pbar.update(1)
            if len(merges) % 50 == 0:
                print(f"  [进度] 已合并 {len(merges)}/{target_merges}, 当前频率={best_count}, "
                      f"合并位置数={len(merge_positions)}, n_positions={n_positions}", flush=True)

    return merges


def build_and_save_tokenizer(vocab, merges, special_tokens, output_path):
    """构建并保存最终 Tokenizer"""
    model = models.BPE(
        vocab=vocab,
        merges=merges,
        unk_token="<|unk|>",
    )

    tokenizer = Tokenizer(model)
    if BBPE:
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                Regex(PRETOKENIZE_REGEX),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tokenizer.post_processor = processors.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
    else:
        tokenizer.normalizer = normalizers.NFKD()
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # 注册特殊 token 为 added_tokens
    tokenizer.add_special_tokens(special_tokens)

    tokenizer.save(output_path)
    print(f"分词器已保存到 {output_path}")
    print(f"词表大小: {len(vocab)}, 合并数: {len(merges)}")


def main():
    print("=" * 60)
    print("自定义 BPE 训练器（不使用 tokenizers.trainers）")
    print("=" * 60)

    normalizer, pre_tokenizer = build_preprocessor()

    print(f"正在读取文件 {FILE_PATH} 并统计词频...")
    word_freq = count_word_frequencies(normalizer, pre_tokenizer, FILE_PATH)
    print(f"唯一预分词数量: {len(word_freq)}")

    print("正在构建初始词表...")
    vocab = build_initial_vocab(word_freq, SPECIAL_TOKENS, LIMIT_ALPHABET)
    print(f"初始词表大小: {len(vocab)} (特殊token: {len(SPECIAL_TOKENS)}, 初始字母: {len(vocab) - len(SPECIAL_TOKENS)}, limit_alphabet: {LIMIT_ALPHABET})")

    print(f"开始 BPE 合并训练 (目标词表大小: {VOCAB_SIZE}, min_frequency: {MIN_FREQUENCY})...")
    merges = train_bpe(word_freq, vocab, VOCAB_SIZE, MIN_FREQUENCY)
    print(f"训练完成，最终词表大小: {len(vocab)}, 合并数: {len(merges)}")

    build_and_save_tokenizer(vocab, merges, SPECIAL_TOKENS, OUTPUT_PATH)

    print("\n验证中...")
    test_tokenizer = Tokenizer.from_file(OUTPUT_PATH)
    test_text = "你好，世界！Hello world!"
    encoded = test_tokenizer.encode(test_text)
    decoded = test_tokenizer.decode(encoded.ids)
    print(f"测试文本: {test_text}")
    print(f"Token 数: {len(encoded.ids)}")
    print(f"解码结果: {decoded}")
    print("验证通过！")


if __name__ == "__main__":
    main()
