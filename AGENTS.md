# AGENTS.md

## Quick start

```powershell
uv run python pre_train.py          # pretrain
uv run python generate_dataset_v3.py # build .npy dataset
uv run python train_tokenizer.py     # train BPE tokenizer
```

Python 3.14, managed via uv (`uv.lock`, `pyproject.toml`).  
PyTorch 2.13.0+cu132 from `mirrors.aliyun.com` / `mirrors.nju.edu.cn` (Chinese mirrors in `pyproject.toml`).

## Training quirks

- **Env fix**: `os.environ["KMP_DUPLICATE_LIB_OK"] = "True"` must be set early (line 3 of `pre_train.py`).
- **torch.compile**: uses `mode="max-autotune"` with `backend="eager"` (not inductor).
- **Two optimizers**: Muon (`torch.optim.Muon`) for 2D weight matrices, AdamW for everything else. Both in one step loop.
- **LR schedule**: WSD (Warmup Stable Decay) via `WarmUpStableDecayLR` in `utils.py`.
- **AMP**: `bfloat16` autocast, enabled by default (`use_amp=True`).
- **Vocab size**: set dynamically from tokenizer vocab at `PreTrainer.__init__` time.
- **Data**: `PretrainTokenIDDataset` loads pre-tokenized `.npy` files (uint16 dtype, seq_len=192+1).
- **Grad accumulation**: controlled by `batch_acceleration` (default 2) — loss divided, gradient clipped at 1.0, step taken every N batches.
- **Checkpoint**: saves multi-optimizer/scheduler states + RNG states. Resume via `resume_from` in `TrainingConfig`.
- **Multi-GPU**: `nn.DataParallel` wrapper when `torch.cuda.device_count() > 1`.

## Model architecture (`models.py`)

- `MyLM` — Transformer decoder with RoPE, RMSNorm (bfloat16 internal), SiLU-gated FFN, optional MoE.
- **Gated attention**: even-numbered layers use sigmoid-gated attention (`use_gate=True`), odd layers use standard attention with sigmoid-activated V.
- DeepNet residual scaling (`use_deepnet_scaling` in `MyLMArgs`).
- Config lives in `config.json` (root), overridden by `TrainingConfig` in `pre_train.py`.

## Tokenizer

- HuggingFace `tokenizers` library, BBPE (ByteLevel BPE), vocab_size=7168 + 6 special tokens.
- Training: `train_tokenizer.py` (std) or `train_tokenizer_custom.py` (pure Python BPE).
- Pre-tokenized dataset generation: `generate_dataset_v3.py` splits on Chinese punctuation, pads to 193 tokens, saves as `.npy`.

## Entrypoints

| Script | Purpose |
|---|---|
| `pre_train.py` | Pretraining (main entry) |
| `continue_training.py` | Resume / continue training (SFT path) |
| `continue_training_sft.py` | SFT variant with ChatML tokenizer |
| `generate_dataset_v3.py` | Build `.npy` from raw text |
| `train_tokenizer.py` | Train BPE tokenizer |
| `models.py` | Model definitions |
| `dataset.py` | Dataset loaders |
| `utils.py` | TextGenerator, LR schedulers, DebugTimer, model_structure |

## No tests

No CI, no linter, no formatter, no typecheck. Only quick smoke checks via `model_architecture_test.py` or standalone `python models.py`.

## Directories ignored by git

`train_text/`, `logs/`, `model/`, `ckpt/`, `*.txt` — these are runtime artifacts. The `logs/` dir contains TensorBoard event files.

## Legacy

`legacy/` and `legacy_model_configuration/` contain older code variants. `legacy_model_configuration/models_2507*.py` are previous model versions.
