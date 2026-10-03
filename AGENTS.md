# AGENTS.md

紧凑版仓库指南。只记录 agent 容易踩坑或无法一眼看出的事实。

## Quick start

```powershell
uv run python pre_train.py            # 预训练（需 tokenizer/ + data/*.npy 存在）
uv run python generate_dataset_v3.py  # 由 train_text/ 生成 data/*.npy
uv run python train_tokenizer.py      # 训练 BBPE tokenizer → tokenizer/
```

- Python 3.14 via uv；`python` 不在 PATH，一律 `uv run python ...`。
- PyTorch 2.13.0+cu132，来自国内镜像（pyproject.toml 内 aliyun pypi + nju cu132 whl）。
- 无 pytest / CI / linter / formatter / typecheck。唯一验证手段：`uv run python models.py`（自带 padding-mask 隔离测试）；`tools/checks/model_architecture_test.py` 依赖 `data/nano_test_data180.txt`（当前缺失，需先生成）。

## Windows / torch.compile 坑

- 开 compile 训练前必须 `$env:PYTHONUTF8='1'`（GBK 下 inductor 模板读取崩溃）：`$env:PYTHONUTF8='1'; uv run python pre_train.py`。
- `KMP_DUPLICATE_LIB_OK="True"` 只在 `pre_train.py` 顶部设置；跑其他脚本需自行设置。
- compile 只认 inductor + `mode="max-autotune"`（实测 ~1.7x）；`backend="eager"` 实测更慢。MoE(FixedCap) 是 compile-safe，**不要**用 `exclude_moe_from_compile` 排除（实测慢 ~14%，接口仅遗留）。
- `PreTrainer.__init__` 设 `torch.set_float32_matmul_precision("high")`（TF32）。
- Windows spawn 模式下 DataLoader `num_workers>0` 会在 worker 里**重新执行主模块顶层代码**：任何（含临时探针/冒烟脚本）会创建 Trainer 或启动 loader 的脚本，主体必须包在 `if __name__ == "__main__":` 里，否则 worker 套娃启动新 worker 直至卡死/报错（本会话已踩两次）。

## 目录布局与运行约定

- **仓库根 = 活跃管线**：`pre_train.py`（唯一维护的训练管线）、`continue_training_sft.py`、`models.py`、`dataset.py`、`utils.py`、`generate_dataset_v3.py`、`generate_dataset_sft.py`、`train_tokenizer*.py`、`debug_moe.py`。
- `data/` 生成后数据集、`tokenizer/` 被引用的分词器 json（均从仓库根按相对路径引用）、`model/` 权重+推理配置 json、`ckpt/` 训练 checkpoint、`logs/` TensorBoard。这些全在 gitignore（`data/`、`train_text/`、`*.txt`、`*.npy`、`*.log` 等），fresh clone 后需重新生成数据。
- `tools/checks/`（验证/复现脚本）与 `tools/model_tools/`（推理检查工具）：**从仓库根运行**；引用根模块的脚本靠顶部 `sys.path.insert(0, parents[2])` 引导，移动脚本需保持该深度约定。
- `debug_moe.py` 不可移动：内部子进程 `python -c "from debug_moe import ..."` 依赖 cwd=仓库根。
- 归档区（复现旧 run 才碰）：`legacy/`（scripts: continue_training.py+models_250830.py、generate_dataset_v2.py、hyperparameter_search.py——后者 import 已不存在的 `StreamingTextDataset`，本就 broken；models: models_2507*.py）、`legacy_model_configuration/`（含 legacy `config.json`，**任何脚本都不读它，其中 vocab_size/scaling 字段已过期**）、`tokenizer_archive/`、`notebooks/`、`autoresearch/`、`experiments/`。

## 训练管线（`pre_train.py`）

- `TrainingConfig` dataclass 是**唯一配置事实源**（含 `data_dir`/`tokenizer_dir` 路径、`resume_from`）。当前默认：`batch_acceleration=2`、`learning_rate=5e-3`、`min_learning_rate=5e-4`（峰值 10%）、`lr_decay_start_rate=0.75`、`warmup_steps=5`、`dataset_shuffle_seed=42`。
- **vocab size 由 tokenizer 自动注入**（`PreTrainer.__init__`），绝不硬编码。
- 双优化器（都在梯度累积边界 step）：`torch.optim.Muon` 管 2D 权重（embedding/lm_head 除外），`adjust_lr_fn="match_rms_adamw"`；`bnb.optim.adamw.AdamW8bit` 管其余，`betas=(0.85, 0.999)`。
- LR：WSD（`utils.WarmUpStableDecayLR`），75% 步数后线性衰减至峰值 10%。AMP bf16 autocast（train 和 validate 都显式 `dtype=torch.bfloat16`）+ GradScaler。
- 梯度累积：loss÷N、clip 1.0；`zero_grad(set_to_none=False)` + 预分配 `.grad` 缓冲保 CUDAGraph 稳定。多卡 `nn.DataParallel` 在 `train()` 内、**compile 之后**包裹。
- 确定性续训：loader `shuffle=False`，排列来自固定 seed；每个 epoch 开头对 base dataset 调 `set_permute_seed(seed+epoch)`（穿透 random_split Subset）重排。`drop_last=True`。resume 用 `resume_from` 指向 `ckpt\ckpt_*_step_*.pth`，恢复模型+双优化器+双调度器+RNG。ckpt 自动清理：留最近 `ckpt_keep_recent=3` 个，更早每 `ckpt_keep_stride=3` 个留 1 个。
- 预训练 loss 有 mask：`CrossEntropyLoss(reduction="none")` × dataset mask 归一，pad 位不贡献 loss。
- 数据格式：`PretrainTokenIDDataset` mmap 读 uint16 `.npy`，行宽 = `seq_max_len+1`（`generate_dataset_v3.py` 的 `SENTENCE_MAXLEN=256+1`；数据切片 `[:-1]`/`[1:]`）。

## 模型与 tokenizer 关键设计

- `models.py MyLM`：RoPE 预计算 cos/sin buffer、RMSNorm（**variance 用 fp32 计算**再回原 dtype）、SiLU-gated FFN、可选 MoE。奇偶层交替：偶数层 sigmoid 门控注意力，奇数层标准注意力+V sigmoid 激活。
- DeepNet 残差缩放（o_proj/down_proj/latent_up × `1/sqrt(2*n_layers)`）**烧在 `_reset_parameters` 里**，`MyLMArgs` 没有对应字段。
- MoE `MoEFFN`：FixedCap 分桶（纯 tensor 操作，compile 友好）、DeepSeek 无辅助 loss 均衡（`expert_bias` buffer）、可选 `latent_moe`；κ=`moe_capacity=1.25`（实测最优）。
- **显式 padding mask**：`MyLMArgs.pad_id`（默认 0）→ `MyLM.forward` 用 `(ids != pad_id)` 构 4D mask 逐层下传，与 causal mask 相交。旧的 `(x.sum(-1)!=0)` 启发式已废弃（`attn_bias` 下会失效）。`padding_mask=None` = 纯 causal（推理）。
- Tokenizer：HF BBPE ByteLevel，训练目标 `vocab_size=7168`，**实际保存 7160**（含 6 special）。EOS=`<|endoftext|>`= id 0，同时是 pad token。现役：`tokenizer/bbpe_tokenizer_7k_260723_xl.json`。`train_tokenizer.py` 输入硬编码 `train_text\merged.txt`。
- SFT（`continue_training_sft.py`，继承 `PreTrainer`）：ChatML，**右 pad + 左截断**（保尾部回答），`SFTTextDataset` 把 id=0 位置 loss mask 置 0；`<|pad|>`(id=2) 仅是保留位。

## Checkpoint 加载陷阱（高频翻车点）

- ckpt 是裸 `torch.save` dict：`torch.load(..., weights_only=False)`；键可能带 `module.` / `_orig_mod.` 前缀，load 前先剥离。
- `_build_model` 内已 `torch.compile`，`self.model` 是 `OptimizedModule`，`state_dict()` 键带 `_orig_mod.` 前缀——对已剥前缀的 ckpt 做前缀过滤必须作用在裸模型上（`getattr(self.model, "_orig_mod", self.model)`），否则**所有键被过滤掉、权重静默不加载**（症状：loss ≈ ln(vocab)）。同理处理 seq_max_len 扩展时 RoPE buffer 的形状失配过滤。

## 杂项

- TensorBoard 看 `logs/`：`launch_tensorboard.bat`。架构测试实验在 `logs/exp/<name>_<时间戳>/`（`arch_test.py` 产出，可多 run 并排：`tensorboard --logdir logs/exp`）。
- **新架构验证用 `arch_test.py`**（pre_train.py 精简版：双优化器/WSD/梯度累积/确定性续训同口径，砍掉 MoE 热力图/权重统计/文本生成/ckpt pruning）；换模型只改顶部 `build_model`+`MODEL_ARGS`，契约 `forward(ids, padding_mask=...) -> (B,S,V)`；调试可传 override JSON 路径缩短 run（max_steps/val_interval_step）。
- `tools/checks/bench_module.py`：任意 nn.Module 的 FP/BP/STEP 吞吐+显存 benchmark（纯常量配置，无 CLI 参数；`uv run python tools/checks/bench_module.py`）。
- 实验产物在 `experiments/results/`（初始化 sweep、sdpa bench 等）；一次性临时脚本约定命名 `bench_*_tmp.py`（已 gitignore）。
