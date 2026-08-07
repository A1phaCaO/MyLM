# AutoResearch: Attention 算子吞吐优化

## 任务目标

优化 `autoresearch/models.py` 中 `Attention` 类的 `forward` 方法，在**保持数学正确性**的前提下，最大化**前向+反向传播的吞吐量**（tokens/second）。

## 工作目录

所有修改**只限于 `autoresearch/` 文件夹内**，禁止修改项目根目录的任何文件。

## 可修改范围

- **只修改** `autoresearch/models.py:Attention` 类
  - `forward` 方法的实现
  - 可添加辅助方法
  - 可修改 `__init__`（保持接口兼容）
- **不修改** `autoresearch/` 下的其他文件（`benchmark.py`、`evolution_env.py`、`__init__.py`）
- **不修改**项目根目录的任何文件

## 约束条件

1. **接口兼容**：`forward(self, x, token_ids=None, mask=None, causal=True) -> Tensor` 签名不可变
2. **输出形状**：`(batch_size, seq_len, d_model)` 必须与输入一致
3. **无 NaN / Inf**：输出必须不包含 NaN 或 Inf
4. **因果性**：`causal=True` 时，位置 i 的输出只能依赖位置 ≤ i 的输入
5. **门控兼容**：偶数层 `use_gate=True` 与奇数层 `use_gate=False` 均需正确工作（`benchmark.py` 默认测试 `use_gate=False` 模式）
6. **仅限 PyTorch 原生**：不可引入额外的编译依赖（如 Triton、CUDA 自定义 kernel）。可以使用 `torch.*` 内的任何 API（含 `torch.compile`）
7. **可训练**：必须支持 `loss.backward()`，梯度必须有效

## 评估方法

由 `autoresearch/benchmark.py` 执行自动评估：

1. **正确性检测**（`verify_correctness`）
   - 检查输出形状
   - 检查 NaN / Inf
   - 验证 backward 可运行且梯度有效
   - 失败则 score = 0
2. **吞吐量基准**（`benchmark_attention`）
   - 固定参数：`d_model=512, d_head=64, n_heads=8, seq_len=256, batch_size=16`
   - 20 轮 warmup + 100 轮 forward+backward 计时
   - 返回 tokens/second
3. **综合分**：`score = throughput if correct else 0.0`

## 进化环境

由 `autoresearch/evolution_env.py` 提供：

- `EvolutionEnv.evaluate()` — 运行完整评测，返回 (throughput, score, is_correct, message)
- `EvolutionEnv.submit(agent_name, ...)` — 记录一次尝试到 `leaderboard.json`
- `EvolutionEnv.leaderboard()` — 查看所有历史记录，按分数降序
- `EvolutionEnv.status()` — 查看当前最佳、总尝试次数、当前代次

历史记录持久化在 `autoresearch/leaderboard.json`，每次 submit 追加一条记录，包含：时间戳、Agent 名、代次、分数、吞吐量、正确性、代码描述、benchmark 参数。

## 使用流程

```python
from autoresearch import EvolutionEnv

env = EvolutionEnv()
throughput, score, is_correct, msg = env.evaluate()

# 查看结果
print(f"Throughput: {throughput:.0f} tokens/sec")
print(f"Score: {score:.0f}")
print(f"Correct: {is_correct}")

# 记录尝试
env.submit(
    agent_name="my_agent_v1",
    score=score,
    throughput=throughput,
    is_correct=is_correct,
    code_description="使用F.scaled_dot_product_attention替代手动实现",
)

# 看排行
for entry in env.leaderboard(top_k=5):
    print(f"#{entry['generation']} {entry['agent_name']}: {entry['score']:.0f}")
```

## 快速验证

```bash
# 运行基准测试
uv run python -m autoresearch.benchmark

# 运行进化环境
uv run python -m autoresearch.evolution_env
```
