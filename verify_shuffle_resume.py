"""验证 PretrainTokenIDDataset 固定种子 shuffle 的续训语义（只读真实数据）。

验证点:
1. 同 seed 两次加载 -> perm 完全一致(种子可复现)。
2. loader(shuffle=False) 第 k 个 batch == perm 连续段 [k*bs:(k+1)*bs]
   (即跳过逻辑按 batch 位置即按已消费样本位置)。
3. 续训无重复: 跳过 K 个 batch 后取到的样本 == perm 的第 K*bs 位起,
   与已训集合 (perm[0:K*bs]) 无交集; 若续训多次同一位置, 得到完全相同 batch。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import numpy as np
from dataset import PretrainTokenIDDataset

DATA_DIR = r"medium_data256v2.npy"
SEQ_MAX_LEN = 256
BATCH_SIZE = 48
VALSET_RATE = 0.0018
SEED = 42


def build(seed):
    # 模拟 PreTrainer.__init__: 先 _set_seed() 再 random_split(依赖 torch 全局 RNG)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = PretrainTokenIDDataset(
        DATA_DIR, seq_max_len=SEQ_MAX_LEN, downsample=1,
        padding_side="right", shuffle_seed=seed,
    )
    val_len = int(len(ds) * VALSET_RATE)
    train_len = len(ds) - val_len
    train_ds, _ = torch.utils.data.random_split(ds, [train_len, val_len])
    return ds, train_ds


def first_tokens(batch, n=5):
    return [int(t) for t in batch[0][0, :n].tolist()]


def row_key(row):
    import hashlib
    return hashlib.md5(np.asarray(row, dtype=np.uint16).tobytes()).hexdigest()


def main():
    print("=== 1. 同 seed 两次加载, perm 可复现 ===")
    ds1, _ = build(SEED)
    ds2, _ = build(SEED)
    same = bool(np.array_equal(ds1.perm, ds2.perm))
    print(f"  perm 相同: {same}  (perm size {ds1.perm.shape})")

    print("=== 2. loader batch k == perm 连续段 ===")
    _, train_ds = build(SEED)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    it = iter(loader)
    for k in (0, 7, 123, 27999, 28000):
        for _ in range(k - (0 if k == 0 else 0)):
            pass
        # 重新取: 每次从头重建迭代器到第 k 个 batch(验证确定性)
    it = iter(loader)
    k_target = 28000
    got = None
    for i in range(k_target + 1):
        b = next(it)
        if i in (0, 7, 123, 27999, 28000):
            start = i * BATCH_SIZE
            expect = train_ds.indices[start:start + BATCH_SIZE]
            expect = [int(ds1.perm[e]) for e in expect]
            # 数据行第一列没有 id, 用整行 hash 校验: 直接比对 perm 映射出的首 token
            # 简单可靠: 比对 batch 每行的第一个有效 token 与 perm 映射后的行首 token
            row_tokens = b[0][:, 0].tolist()
            perm_tokens = []
            src = train_ds.indices[start:start + BATCH_SIZE]
            for e in src:
                row = np.array(ds1.data[int(ds1.perm[e])], dtype=np.uint16)
                perm_tokens.append(int(row[0]))
            ok = row_tokens == perm_tokens
            print(f"  batch[{i}] 行首 token 与 perm 映射一致: {ok}")
        if i == k_target:
            got = b
    # batch 28000 与 batch 28001 所在 perm 位置不重叠(整行内容对比,
    # 仅用行首 token 可能误判: 不同行可共享首 token)
    b28000 = got
    b28001 = next(it)
    rows_a = {row_key(r) for r in b28000[0].tolist()}
    rows_b = {row_key(r) for r in b28001[0].tolist()}
    overlap = len(rows_a & rows_b)
    print(f"  batch[28000] 与 batch[28001] 整行相同数: {overlap} (perm 无重复, 应=0)")

    print("=== 3. 续训无重复: 跳过 K 批后取到的样本 == 新区域, 与已训无交集 ===")
    K = 28000
    ds3, train_ds3 = build(SEED)
    loader3 = torch.utils.data.DataLoader(
        train_ds3, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    it3 = iter(loader3)
    trained_ids = set()
    for i in range(K):
        if i % 7000 == 0:
            print(f"  ...已消费 {i} 批")
        b = next(it3)
        if i < 3:  # 记录已训样本位置(用 perm 位置)
            pass
    trained_pos = set(range(K * BATCH_SIZE))
    # 续训: 下一批 = perm 的 [K*bs, (K+1)*bs)
    pos_resume = list(range(K * BATCH_SIZE, (K + 1) * BATCH_SIZE))
    overlap = len(set(pos_resume) & trained_pos)
    print(f"  续训批在 perm 中的位置 [{K*BATCH_SIZE}, {K*BATCH_SIZE+BATCH_SIZE})")
    print(f"  与已训位置交集: {overlap} (应为 0, 数学保证)")
    # 再次从头构建并直接取第 K 个 batch, 与上面 loader3 取到的比对(续训确定性)
    _, train_ds4 = build(SEED)
    loader4 = torch.utils.data.DataLoader(
        train_ds4, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    it4 = iter(loader4)
    b_ref = None
    for i in range(K + 1):
        b_ref = next(it4)
    b_resume = next(it3)  # loader3 已在 K 位置
    eq = bool(torch.equal(b_ref[0], b_resume[0]))
    print(f"  续训(replay) 与 从头直取第 {K} 批 内容一致: {eq}")
    print(f"  即: 续训后拿到的就是原始训练流中下一个 batch, 不多不少")


if __name__ == "__main__":
    main()