# tools

根目录整理出来的小型辅助工具，按用途分两类。**统一约定：从仓库根目录运行**
（脚本内数据/模型路径多为仓库根相对路径，如 `model\...`、`ckpt\...`、`*.npy`）。

引用了根目录 Python 模块（`models` / `utils` / `dataset` / `pre_train`）的脚本，
顶部均有 3 行 `sys.path.insert(... parents[2])` 引导，无需设置 PYTHONPATH。

## checks/ — 验证与复现脚本

| 脚本 | 用途 |
|---|---|
| `model_architecture_test.py` | 冒烟测试：tiny 模型跑 `data/nano_test_data180.txt`（配套库 `model_baseline.py`，勿单独移动；**注意该数据文件当前缺失，需先生成**） |
| `test_tokenizer.py` | 分词器 smoke 检查（unk 词统计） |
| `validate_notebook.py` | 校验 `notebooks/pre_train_notebook.ipynb` 的 JSON 格式与代码单元语法 |
| `verify_shuffle_resume.py` | 验证 `PretrainTokenIDDataset` 固定种子 shuffle 的续训语义（只读） |
| `resume_replay_check.py` | 断点续训数据重复性实验（依赖 `ckpt\ckpt_epoch_0_step_28000.pth`） |
| `test_accum_equiv.py` | 梯度累积等价性复现测试（batch64x1 vs batch32x2） |

## model_tools/ — 推理与模型检查

| 脚本 | 用途 |
|---|---|
| `run_model_for_state.py` | 加载 ckpt 做模型状态检查 / 续写采样 |
| `run_model_for_chat.py` | SFT 模型交互式对话 |
| `find_similar_words.py` | 基于 embedding 查找近义词 |
| `visualize_logs.py` | 训练日志 JSON 可视化 |
| `reset_ckpt_config.py` | 修改 checkpoint 内保存的训练配置 |

运行示例：`uv run python tools/checks/validate_notebook.py`
