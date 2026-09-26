# 将多个txt文件按配置的比例打包合并成一个txt文件
from tqdm import tqdm
import os
import random

# 文件路径与对应采样比例（sample_rate）
# sample_rate 含义：
#   - (0, 1) : 降采样，随机抽取该百分比的行。例如 0.1 表示随机抽取 10% 的行
#   - 1      : 不采样，使用全部数据
#   - > 1    : 重复（repeat），将数据重复若干次。例如 3 表示将数据重复 3 倍
# FILE_RATIO = {
#     r"train_text\WanJuan1.0part-000036-a894b46e-downsample30x-processed.txt": 0.27,
#     r"train_text\ultrafineweb-zh-part-001-of-256-downsample8x.txt": 0.9,
#     r"train_text\SkyPile2023-14_zh_middle_0010_processed.txt": 1,
#     r"train_text\时政文章.txt": 1,
#     r"train_text\斗罗大陆4终极斗罗.txt": 0.04,
#     r"train_text\高三议论文-作文网20220310-20200806.txt": 1,
#     r"train_text\SFT\Infinity-Instruct-Gen-00000-of-00015-processed.txt": 0.8,
#     r"train_text\ultrafineweb-l3-mutistyle-en-part0.txt": 0.2,
#     r"train_text\ultrafineweb-l3-mutistyle-cn-part0.txt": 0.45,
# }

FILE_RATIO = {
    r"data/data_sft256.txt": 1,
    r"data/data_sft384.txt": 1,
    # r"data/data_sft512.txt": 1
}

# 输出文件路径
OUTPUT_PATH = r"data/data_sft-256-384-merge.txt"

# 随机种子，保证可复现；设为 None 则每次随机
SEED = 42


def count_lines(path):
    """统计文件行数"""
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def get_file_size_bytes(path):
    """返回文件大小（字节）"""
    return os.path.getsize(path)


def get_sampled_size_mb(original_size_mb, ratio):
    """根据采样比率计算预期输出大小（MB）"""
    if ratio <= 0:
        return 0
    elif ratio <= 1:
        return original_size_mb * ratio
    else:
        return original_size_mb * int(ratio)


def get_sample_label(ratio):
    """返回可读的采样比率标签"""
    if ratio < 1:
        return f"downsample {ratio*100:.2f}%"
    elif ratio == 1:
        return "1x (全量)"
    else:
        return f"repeat {int(ratio)}x"


def pack_files(file_ratio, output_path, seed=SEED):
    """按配置比例从各文件采样并合并写入输出文件"""
    if seed is not None:
        random.seed(seed)

    # 统计每个文件需要采样的行数
    plan = {}
    for path, ratio in file_ratio.items():
        total = count_lines(path)
        if ratio <= 1:
            take = int(total * ratio)
        else:
            take = total * int(ratio)
        plan[path] = take

    total_take = sum(plan.values())

    # 打印采样前统计表格
    print(
        f"{'文件路径':<50} {'原始大小(MB)':<9} {'采样后大小(MB)':<10} {'占比':<10} {'采样方式':<20} "
    )
    print("-" * 120)
    total_size_mb = 0
    total_sampled_size_mb = 0
    for path, ratio in file_ratio.items():
        original_size_mb = get_file_size_bytes(path) / (1024 * 1024)
        sampled_size_mb = get_sampled_size_mb(original_size_mb, ratio)
        total_size_mb += original_size_mb
        total_sampled_size_mb += sampled_size_mb
    for path, ratio in file_ratio.items():
        original_size_mb = get_file_size_bytes(path) / (1024 * 1024)
        sampled_size_mb = get_sampled_size_mb(original_size_mb, ratio)
        percentage = (
            (sampled_size_mb / total_sampled_size_mb) * 100
            if total_sampled_size_mb > 0
            else 0
        )
        print(
            f"{path[:31]+'...'+path[-20:]:<50} {original_size_mb:<15.2f} {sampled_size_mb:<15.2f} {f'{percentage:.1f}%':<10} {get_sample_label(ratio):<20}"
        )
    print("-" * 120)
    print(
        f"共计{len(file_ratio)}个文件".ljust(50)
        + f"{total_size_mb:.1f}MB".ljust(16)
        + f"{total_sampled_size_mb:.1f}MB".ljust(16)
        + f"100%".ljust(16)
    )
    print("=" * 120)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for path, take in plan.items():
            total = count_lines(path)
            ratio = file_ratio[path]
            if ratio <= 1:
                # 需要采样的行索引集合
                indices = set(random.sample(range(total), take)) if take < total else set(range(total))
                with open(path, 'r', encoding='utf-8') as infile:
                    for i, line in enumerate(tqdm(infile, desc=os.path.basename(path))):
                        if i in indices:
                            outfile.write(line if line.endswith('\n') else line + '\n')
            else:
                # 重复（repeat）：将全部数据重复 int(ratio) 次
                repeat_times = int(ratio)
                with open(path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()
                for _ in range(repeat_times):
                    for line in tqdm(lines, desc=f"{os.path.basename(path)} x{repeat_times}"):
                        outfile.write(line if line.endswith('\n') else line + '\n')

    # 打印最终文件大小与各子文件实际占比
    final_bytes = get_file_size_bytes(output_path)
    final_size_mb = final_bytes / (1024 * 1024)
    print(f"\n合并完成，共写入 {total_take} 行 -> {output_path}")
    print(
        f"{'文件路径':<50} {'写入行数':<10} {'写入大小(MB)':<12} {'实际占比':<10}"
    )
    print("-" * 90)
    # 重新计算各文件实际写入字节（按行数近似：实际占比以行数为准）
    for path, take in plan.items():
        pct = (take / total_take * 100) if total_take else 0
        print(
            f"{os.path.basename(path)[:40]:<50} {take:<10} {f'{pct/100*final_size_mb:.2f}':<12} {f'{pct:.2f}%':<10}"
        )
    print("-" * 90)
    print(f"最终文件大小：{final_bytes} 字节 ({final_size_mb:.2f} MB)")


if __name__ == "__main__":
    pack_files(FILE_RATIO, OUTPUT_PATH)
