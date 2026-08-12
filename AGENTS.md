# AGENTS.md

## Quick start

```powershell
uv run python pre_train.py            # pretrain (needs .npy dataset + tokenizer json present)
uv run python generate_dataset_v3.py  # build .npy token dataset
uv run python train_tokenizer.py      # train BBPE tokenizer
```

- Python 3.14 managed via uv (`pyproject.toml`, `uv.lock`). `python` is **not** on PATH — always use `uv run python ...`.
- PyTorch 2.13.0+cu132 from Chinese mirrors (aliyun pypi / nju cu132 whl, configured in `pyproject.toml`).

## Windows / torch.compile gotchas

- Set `$env:PYTHONUTF8='1'` before training with compile on — inductor template reads crash under GBK on Windows: `$env:PYTHONUTF8='1'; uv run python pre_train.py`.
- `os.environ["KMP_DUPLICATE_LIB_OK"] = "True"` is set inside `pre_train.py` (line 35) but **not** in other scripts; set it manually when running those.
- `torch.compile`: inductor backend with `mode="max-autotune"` only (measured ~1.7x). Do NOT use `backend="eager"` (no fusion, measured slower). MoE (FixedCap) is compile-safe — do not exclude it; `exclude_moe_from_compile` is legacy and excluding measures ~14% slower.
- `PreTrainer.__init__` sets `torch.set_float32_matmul_precision("high")` (TF32).

## Training pipeline — `pre_train.py` is the only maintained path

- **Config**: `TrainingConfig` dataclass in `pre_train.py` is the single source of truth (model args, data/tokenizer paths, resume). Root `config.json` is a legacy artifact NOT read by any script (only `run_model_for_state.py` reads `model\config.json`).
- **Vocab size** is set automatically from the tokenizer at `PreTrainer.__init__` — never hardcode; `config.json`'s `vocab_size` (7186) is stale.
- **Two optimizers** (both stepped at grad-accumulation boundaries): `torch.optim.Muon` (built into pinned torch 2.13) for 2D weights excluding embedding/lm_head, with `adjust_lr_fn="match_rms_adamw"`; `bnb.optim.adamw.AdamW8bit` (bitsandbytes) for the rest with `betas=(0.85, 0.999)`.
- **LR schedule**: WSD via `WarmUpStableDecayLR` (`utils.py`) — `warmup_steps`, `lr_decay_start_rate=0.75` (% of steps before decay), `min_lr = 10%` of peak, linear decay.
- **AMP**: bf16 autocast + `torch.GradScaler` (no-op under bf16 but wired in).
- **Grad accumulation**: `batch_acceleration` (currently 4) — loss ÷ N, clip at 1.0, optim/sched step every Nth batch; `zero_grad(set_to_none=False)` + pre-allocated `.grad` buffers keep CUDAGraph/compile stable.
- **Data**: `PretrainTokenIDDataset` mmap-reads uint16 `.npy`; stored row length = `seq_max_len + 1` (dataset slices `[:-1]`/`[1:]`), loader truncates/pads anyway. `.npy`/`*.txt` are gitignored, so datasets must be regenerated after a fresh clone (`generate_dataset_v3.py` → `medium_data256v2.npy`, `SENTENCE_MAXLEN=257`).
- **Deterministic resume**: DataLoader runs `shuffle=False`; permutation comes from `dataset_shuffle_seed` (fixed seed), so a resumed run continues from the consumed position without repeated data. `drop_last=True` keeps batch shapes constant.
- **Resume**: set `resume_from` to a `ckpt\ckpt_*_step_*.pth` path (restores model + both optimizers + both schedulers + RNG states). Step checkpoints are auto-pruned (keep `ckpt_keep_recent`; older every `ckpt_keep_stride`-th).
- **Multi-GPU**: `nn.DataParallel` wraps inside `train()`, i.e. AFTER `torch.compile`.
- **Logs**: TensorBoard in `logs/`; view with `launch_tensorboard.bat` (`uv run tensorboard --logdir logs`).

## Model architecture (`models.py`)

- `MyLM` — Transformer decoder: RoPE (precomputed cos/sin buffers), RMSNorm (internal bf16 cast, see `models.py:50`), SiLU-gated FFN, optional MoE.
- **Gated attention**: even-numbered layers use sigmoid-gated attention (`use_gate=True`), odd layers standard attention with sigmoid-activated V (`models.py:537`).
- **`MoEFFN`** — FixedCap bucketing (pure tensor ops: bincount/index_add/bmm, compile-friendly), DeepSeek loss-free load balancing via `expert_bias` buffer (no aux loss), optional `latent_moe` bottleneck; κ = `moe_capacity` (1.25 measured best).
- DeepNet-style residual scaling (o_proj/down_proj/latent_up × `1/sqrt(2*n_layers)`) is baked into `_reset_parameters` — `MyLMArgs` has no `use_deepnet_scaling`/`resid_scale`/`layer_scale` fields (those `config.json` keys are stale).
- Checkpoints are plain `torch.save` dicts — load with `weights_only=False`; strip `module.` / `_orig_mod.` prefixes before `load_state_dict`. **Gotcha**: `torch.compile` is applied inside `_build_model` (pre_train.py:207), so `self.model` is an `OptimizedModule` whose `state_dict()` keys carry the `_orig_mod.` prefix — prefix-matching against a stripped checkpoint must run on the bare model (`getattr(self.model, "_orig_mod", self.model)`), otherwise every key is filtered out and weights silently never load (loss ≈ ln(vocab)). Same rule for shape-mismatch filtering (RoPE cos/sin buffers when extending seq_max_len).

## Tokenizer

- HF `tokenizers` BBPE (ByteLevel). Trainer targets `vocab_size=7168`; actual saved vocab = 7160 + 6 special tokens. EOS = id 0 (`<|endoftext|>`).
- **SFT pad 复用 EOS (id=0)，对齐 Qwen**（EOS=pad 同一 token）：`SFTTextDataset` 默认 `pad_value=0`（左 pad），`<|pad|>`(id=2) 是 tokenizer 保留字段、不用于训练；mask 对 id=0 位置（含数据行尾的 `<|endoftext|>`）强制不计 loss；`SFTTrainer._zero_pad_embedding` 把 id=0 的 embedding 行置零，让 `models.py` 的 `seq_mask`（`x.sum(-1)!=0`）对左 pad 真正生效。
- Live tokenizer: `bbpe_tokenizer_7k_260723_xl.json` (hardcoded in `pre_train.py` and `generate_dataset_v3.py`).
- `train_tokenizer.py` hardcodes input `train_text\merged.txt` and saves to legacy filename `bbpe_tokenizer_6k_260715.json` — rename after training. `train_tokenizer_custom.py` = pure-Python BPE alternative; `test_tokenizer.py` = smoke check.

## Entrypoints

| Script | Status |
|---|---|
| `pre_train.py` | Main pretraining; the only actively maintained pipeline |
| `continue_training.py` | Legacy SFT chain — imports `models_250830` and old 6k ChatML tokenizers, NOT `models.py`; partly stale config |
| `continue_training_sft.py` | SFT (ChatML 指令微调) — inherits `PreTrainer` (syncs all latest pretrain mechanisms: WSD, Muon+AdamW8bit, bf16 AMP, compile, ckpt prune/resume); uses `models.py` + `SFTTextDataset` (dataset.py, integrates SFT loss mask) + new 7k tokenizer; `train_from` = pretrain weights |
| `hyperparameter_search.py` | **Broken** — imports `StreamingTextDataset`, which no longer exists in `dataset.py` |
| `generate_dataset_v3.py` | Build uint16 `.npy` from `train_text/` (hardcoded paths/sample rates) |
| `train_tokenizer.py` | Train BBPE tokenizer |
| `models.py` | Model definitions; self-tests when run standalone |
| `dataset.py` / `utils.py` | Loaders / TextGenerator, LR schedulers, DebugTimer, model_structure |
| `model_architecture_test.py` | Smoke test — tiny model on `nano_test_data180.txt` |
| `debug_moe.py`, `visualize_logs.py`, `find_similar_words.py`, `run_model_for_state.py` | One-off utilities |

## No tests / CI

No pytest, no CI, no linter, no formatter, no typecheck. Only smoke checks: `uv run python models.py` or `model_architecture_test.py`.

## Directories ignored by git

`train_text/`, `logs/`, `model/`, `ckpt/`, `*.txt`, `*.npy` — runtime artifacts / generated data.

## Legacy / experimental code

`legacy/`, `legacy_model_configuration/` (incl. `models_260808.py`), root `models_2507*.py` / `models_250830.py`, `notebooks/`, `autoresearch/`, `experiments/` — old or experimental variants; don't edit unless replicating an old run.