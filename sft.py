import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils
import json
from models import *
from utils import TextGenerator
import numpy as np
from tqdm import tqdm
import time
import random
import matplotlib.pyplot as plt
from utils import model_structure
import collections
import gc
import os


# ---------------------------------------------------#
#   设置种子
# ---------------------------------------------------#
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# INPUT_DATA_DIR = r'/kaggle/input/skypile-part/input_data.txt'
# OUTPUT_DATA_DIR = r'/kaggle/input/skypile-part/output_data.txt'
MODEL_SAVE_DIR = r"model_sft.pth"
DATA_DIR = r"sft_data.txt"

EPOCHS = 40
BATCH = 16

DATASET_DOWNSAMPLE = int(1 / 0.002)
LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 1e-4
USE_AMP = False

# 导入模型文件
MODEL_DIR = "model_kaggle_126M-tok.pth"
VOCAB_DIR = "vocab_kaggle_126M-tok.json"
model = torch.load(MODEL_DIR).module


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DebugTimer:
    def __init__(self, name=None):
        self.start_time = None
        self.name = name

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.timer_start(self.name)
            result = func(*args, **kwargs)
            self.timer_stop()
            return result

        return wrapper

    def timer_start(self, name=None):
        self.start_time = time.perf_counter()
        if name is not None:
            self.name = name
        print(f"{self.name}:", end="")

    def timer_stop(self):
        elapsed_time = round(time.perf_counter() - self.start_time, 4)
        print(f"{elapsed_time}s")


t = DebugTimer()


class Vocab:
    def __init__(self, word_counts=None, specials=["<PAD>", "<UNK>"]):
        if word_counts is None:
            self.stoi = {}
            self.itos = []
            return None

        # 先添加特殊token到词典
        self.stoi = {}
        self.itos = []
        for special in specials:
            self.stoi[special] = len(self.stoi)

        # 添加普通单词
        for word, _ in word_counts:
            self.stoi.setdefault(word, len(self.stoi))

        # 设置默认索引为`<UNK>`的索引
        self.itos = list(self.stoi.keys())
        self.default_index = self.stoi["<UNK>"]

    def __call__(self, word):
        """
        调用这个方法比直接调用
        self.stoi.get(word, self.default_index)
        慢很多...
        """
        # 如果单词不在词典中，返回`<UNK>`的索引
        return self.stoi.get(word, self.default_index)

    def __len__(self):
        return len(self.stoi)

    def init_from_dict(self, word_dict):
        self.stoi = word_dict
        self.itos = list(self.stoi.keys())
        self.default_index = self.stoi["<UNK>"]


class TextDatasetForSFT(torch.utils.data.Dataset):
    def __init__(self, data_dir, vocab: Vocab, downsample: int):
        super().__init__()
        self.tokenizer = lambda x: x.split(" ")
        self.load_and_preprocess_data(data_dir, downsample)
        self.load_vocab(vocab)
        self.encode_data()
        self.seq_max_len = len(self.input_data[0])  # padding后元素一样长

    @DebugTimer("加载并生成训练数据")
    def load_and_preprocess_data(self, data_dir, downsample):
        with open(data_dir, encoding="utf-8") as f:
            lines = f.read().splitlines()[::downsample]
        self.input_data = [self.tokenizer(line)[:-1] for line in lines]
        self.output_data = [self.tokenizer(line)[1:] for line in lines]

    @DebugTimer("导入词典")
    def load_vocab(self, vocab):
        self.vocab = vocab
        self.word2index = self.vocab.stoi
        self.index2word = self.vocab.itos
        self.word_nums = len(self.vocab)

    @DebugTimer("编码数据")
    def encode_data(self):
        word2index = self.word2index
        self.input_data = [
            torch.tensor(
                [word2index.get(word, self.vocab.default_index) for word in sentence]
            )
            for sentence in tqdm(self.input_data)
        ]
        self.output_data = [
            torch.tensor(
                [word2index.get(word, self.vocab.default_index) for word in sentence]
            )
            for sentence in tqdm(self.output_data)
        ]

        self.input_data = torch.nn.utils.rnn.pad_sequence(
            self.input_data, batch_first=True
        )
        self.output_data = torch.nn.utils.rnn.pad_sequence(
            self.output_data, batch_first=True
        )

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        """
        根据索引获取数据样本
        参数:
            index (int): 索引值
        返回值:
            tuple: 输入数据和输出数据的元组，以tensor形式表示
        """
        return (
            self.input_data[index].clone().detach(),
            self.output_data[index].clone().detach().long(),
        )


class TextGenerator:
    def __init__(self, model: nn.Module, vocab, device) -> None:
        self.model = model
        self.vocab = vocab
        self.device = device

    def generate(self, start_token: list, seq_len=10, temperature=1, print_out=True):
        with torch.no_grad():
            self.model.eval()
            tokens = start_token.copy()
            for _ in range(seq_len):
                seq = [self.vocab(j) for j in tokens]
                seq = nn.utils.rnn.pad_sequence(
                    [
                        torch.tensor(seq).clone().detach().to(self.device),
                        torch.zeros(self.model.model_args.seq_max_len),
                    ],
                    batch_first=True,
                )[
                    0
                ]  # 加一项保证padding长度为seq_max_len
                out = self.model(seq.long().unsqueeze(0).to(self.device))
                out /= temperature

                probabilities = F.softmax(out, dim=-1)
                next_token = self.vocab.itos[
                    probabilities.squeeze(0)[len(tokens) - 1].multinomial(num_samples=1)
                ]
                if print_out:
                    print(next_token, end=" ")
                tokens.append(next_token)

        return tokens


# 加载词典
with open(VOCAB_DIR, "r", encoding="utf-8") as f:
    content = f.read()
    try:
        word_dict = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error at position {e.pos}: {e.msg}")
        print(f"Context: {content[e.pos-20:e.pos+20]}")

vocab = Vocab()
vocab.init_from_dict(word_dict)

dataset = TextDatasetForSFT(DATA_DIR, vocab=vocab, downsample=DATASET_DOWNSAMPLE)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH, shuffle=True, pin_memory=False
)
# with open("vocab.txt", "w", encoding="utf-8") as file:
#     file.write(str(dataset.word2index))
# 使用json保存vocab
with open("vocab.json", "w", encoding="utf-8") as file:
    json.dump(dataset.word2index, file)

# print(dataset.word2index)
# print([dataset[i] for i in range(len(dataset))])
print(f"词数: {dataset.word_nums}")
dataset_len = len(dataset)
print(f"数据集数量：{dataset_len}")

model_structure(model)

model.to(device)
print(dataset.index2word[:30])

# print([(i, item) for i, item in enumerate(dataset.index2word)])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=EPOCHS, T_mult=2, eta_min=MIN_LEARNING_RATE, last_epoch=-1
)
loss_log = []
# print(dataset.vocab('<OOV>'))


scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
print("~~~训练咯~~~")
gc.collect()

# 训练模型
for epoch in range(EPOCHS):
    #     loss_log = []
    loss_sum = 0
    bar = tqdm(dataloader, unit="batch")
    torch.cuda.empty_cache()
    for i, (inputs, targets) in enumerate(dataloader):
        # print(f'Step: {i}')
        model.train()
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            output = model(torch.as_tensor(inputs))
            # print(inputs.size(), output.size(), targets.size())
            loss = criterion(output.view(-1, dataset.word_nums), targets.view(-1))

        scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(
        #     parameters=model.parameters(), max_norm=1, norm_type=2
        # )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        loss_log.append(loss.item())
        loss_sum += loss.item()

        bar.update(1)
        bar.postfix = f"Loss: {round(loss.item(), 5)}"

    scheduler.step()
    bar.close()

    with torch.no_grad():
        print(scheduler.get_last_lr())
        print(
            f"Epoch: {epoch+1}/{EPOCHS}, Loss_sum: {loss_sum}, Loss_avg: {loss_sum/len(dataloader)}"
        )
        if epoch % 1 == 0:
            # start = [
            #     random.choice(dataset.index2word[2:20])
            # ]  # 取词表中词频最高的20词(不含OOV PAD)
            start = ["h", "u", "m", "a", "n", ":", " "]
            generator = TextGenerator(model, dataset.vocab, device)
            ans = generator.generate(
                start_token=start, seq_len=25, print_out=False, temperature=0.8
            )
            ans = ans[len(start) :]  # 截掉start_token
            print(f"Step: {i} (input){start}->", end="")
            for i in ans:
                print(i, end=" ")
            print("\n\n")
        if epoch % 1 == 0:
            # 保存ckpt
            torch.save(model, f"{''.join(MODEL_SAVE_DIR.split(r'/')[:-1])}ckpt_{1}.pth")


# 保存模型
torch.save(model, f"{MODEL_SAVE_DIR}")
torch.jit.trace(model, inputs).save(f"model_viz.pt")
# 可视化loss曲线
plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.cla()  # 清除当前轴
plt.plot(loss_log)
plt.yscale("log")  # 将 y 轴设置为对数坐标轴
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.cla()  # 清除当前轴
plt.plot(loss_log)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.pause(0.001)  # 暂停一小段时间以便图形更新
plt.show()


# 测试模型
test_generator = TextGenerator(model, dataset.vocab, device)
MAX_LEN = 100
T = 0.8
while True:
    start = input("In>>")
    if start[:2] == "T=":
        T = float(start[2:])
        print(f"T={T}")
    else:
        template = f"human: {start} gpt: "
        print(
            f"T={T}\n"
            + "".join(
                test_generator.generate(
                    start_token=list(template),
                    seq_len=MAX_LEN,
                    temperature=T,
                    print_out=False,
                )
            )
        )
