"""断点续训数据重复性实验 v2 (只读, 不动 ckpt / pre_train.py)。

Part 1 (toy, 精确可复现): 模拟训练-中断-续训, 计算续训后取到的样本与
   中断前已训样本的"重复命中率"。若续训开启全新一轮 shuffle, 重复率应
   约等于(已训样本数 / 训练集大小) -> 证明重复训练真实存在。
Part 2 (真实数据): 恢复真实 ckpt CPU RNG 后, loader 产出的 batch 首个样本索引
   与 seed42 从头 loader 的首批索引比较(是否全新排列 / 是否从头重来)。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import random
import time
import numpy as np
import torch
import torch.utils.data
from dataset import PretrainTokenIDDataset

SEED = 42
DATA_DIR = r"medium_data256v2.npy"
SEQ_MAX_LEN = 256
BATCH_SIZE = 48
VALSET_RATE = 0.0018
PADDING_SIDE = "right"
REAL_CKPT_RNG = torch.load(
    r"ckpt\ckpt_epoch_0_step_28000.pth",
    map_location="cpu",
    weights_only=False,
)["rng_states"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ToyDS(torch.utils.data.Dataset):
    """id 可追踪的玩具数据集: 每个样本带唯一 id(第一列), 便于交集检测。"""

    def __init__(self, n, L):
        self.n, self.L = n, L
        self.data = np.random.randint(1, 2**31, size=(n, L + 1)).astype(np.int64)
        self.data[:, 0] = np.arange(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        row = self.data[i]
        return row[:-1], row[1:], torch.ones(self.L)


def batch_ids(batch):
    """取 batch 中每行首列(唯一 id)。"""
    return set(batch[0][:, 0].tolist())


def run_repeat_experiment(n, trained_batches, resumed_batches, bs=20, verbose=True):
    """训练 trained_batches 个 batch 后中断;
    续训 resumed_batches 个 batch, 统计与已训集合的重复。
    """
    torch.manual_seed(123)
    ds = ToyDS(n, 32)
    train_len = int(n * 0.95)
    a, _ = torch.utils.data.random_split(ds, [train_len, n - train_len])

    # run1: 连续训练到中断点(保存 RNG 快照), 记录已训样本集合
    set_seed(7)
    dl1 = torch.utils.data.DataLoader(a, batch_size=bs, shuffle=True)
    it1 = iter(dl1)
    trained = set()
    rng_snapshot = None
    for i in range(trained_batches + resumed_batches):
        batch = next(it1)
        if i < trained_batches:
            trained |= batch_ids(batch)
        if i == trained_batches - 1:
            rng_snapshot = torch.get_rng_state()

    # resume: 重建 loader -> 恢复 RNG -> 取 resumed_batches 个 batch
    set_seed(7)
    dl2 = torch.utils.data.DataLoader(a, batch_size=20, shuffle=True)
    torch.set_rng_state(rng_snapshot)
    it2 = iter(dl2)
    resumed = batch_ids(next(it2))
    for _ in range(1, resumed_batches):
        resumed |= batch_ids(next(it2))

    dup = len(resumed & trained)
    ratio = dup / max(len(resumed), 1)
    if verbose:
        print(f"  n={n}, 训练集={train_len}, 已训 {trained_batches}批, 续训 {resumed_batches}批")
        print(f"    续训 {len(resumed)} 样本中, {dup} 个曾训过 ({ratio*100:.1f}%)")
        print(f"    理论无重复时应为 0%, 全新一轮 shuffle 时约为 "
              f"{trained_batches*20/train_len*100:.1f}%")
    return ratio


def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("Part 1: 玩具数据 - 中断后续训的重复率")
    ratio1 = run_repeat_experiment(n=2000, trained_batches=30, resumed_batches=10)
    ratio2 = run_repeat_experiment(n=10000, trained_batches=120, resumed_batches=20)
    print(f"  Part 1 结论: 续训数据重复率 ~ {ratio1*100:.1f}% / {ratio2*100:.1f}%")

    # ============ Part 2: 真实数据 ============
    print("=" * 70)
    print("Part 2: 真实数据 - 恢复 ckpt RNG 后新 loader 的 perm 与从头是否一致")
    set_seed(SEED)
    dataset = PretrainTokenIDDataset(
        DATA_DIR, seq_max_len=SEQ_MAX_LEN, downsample=1, padding_side=PADDING_SIDE
    )
    val_len = int(len(dataset) * VALSET_RATE)
    train_len = len(dataset) - val_len
    train_ds, _ = torch.utils.data.random_split(dataset, [train_len, val_len])

    def first_batch_ids(dl):
        return next(iter(dl))[0][:, 0].tolist()

    set_seed(SEED)
    loader_fresh = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    ids_fresh = first_batch_ids(loader_fresh)

    set_seed(SEED)
    loader_resume = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    torch.set_rng_state(REAL_CKPT_RNG["torch"])
    ids_resume = first_batch_ids(loader_resume)

    same = set(ids_fresh) == set(ids_resume)
    overlap = len(set(ids_fresh) & set(ids_resume))
    print(f"  数据集行数: {len(dataset)}, train: {train_len}")
    print(f"  从头 loader 首批样本索引: {ids_fresh[:6]} ...")
    print(f"  续训 loader 首批样本索引: {ids_resume[:6]} ...")
    print(f"  28 个索引比较: 相同批={same}, 交集={overlap}")
    print(f"  (相同 => 续训从新排列第一页开始, 等同于从头重来)")
    print(f"  (不同 => 新 shuffle 中的随机位置起步, 已训数据会被再次抽中)")

    print(f"总耗时 {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()