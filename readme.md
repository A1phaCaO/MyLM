# MyLM - 从零实现的语言模型训练全流程

**以下内容由AI临时生成，可能有幻觉、错误，请注意甄别。**

---

一个从零开始实现的完整语言模型（LM）训练全流程代码库，既是个人学习项目，也为初学者提供参考。

## 🎯 项目目标与价值定位

本项目旨在通过从头实现的方式，帮助学习者深入理解现代语言模型的底层原理和训练流程。我们致力于创建一个轻量化的教学型代码库，让初学者能够在较低硬件门槛下探索LM训练的核心技术。

### 核心亮点

- **从头实现**: 最小化依赖第三方库，注重底层原理展示，避免过度抽象，展现底层实现细节
- **轻量化设计**: 采用小型模型架构，降低硬件门槛，适合个人开发者和学生使用
- **实用架构实现**: 基于经典Transformer架构，包含RoPE位置编码、门控注意力（Gated Attention）、MoE等现代LM训练技术
- **完整训练流程**: 覆盖数据预处理、模型构建、训练循环等基础环节
- **教育导向**: 代码结构相对清晰，注释较详尽，便于理解学习

## 🏗️ 技术架构

### 模型设计
- 基于Transformer架构的GPT模型
- 实现了RoPE位置编码、门控注意力（Gated Attention）、MoE等现代化改进技术
- 包含多种模型变体以供学习比较
- 支持可配置的模型尺寸和层数

### 训练流程
- 完整的数据预处理管道
- 模型训练循环实现
- 权重初始化策略对比研究
- 超参数搜索与优化工具

### 工具组件
- BPE分词器实现
- 模型验证与测试工具
- 训练日志可视化工具
- 检查点管理与恢复

## 🔧 快速开始

```bash
# 克隆项目
git clone <repository_url>

# 安装依赖
pip install torch

# 运行预训练
python pre_train.py

# 数据集生成
python generate_dataset_v2.py

# 训练分词器
python train_tokenizer.py
```

## 📁 项目结构

```
Pytorch_GPT/
├── models.py                    # 模型定义与架构实现
├── dataset.py                   # 数据集处理与加载
├── pre_train.py                 # 预训练脚本主入口
├── utils.py                     # 通用工具函数（包括学习率调度器等）
├── train_tokenizer.py           # BPE分词器训练
├── generate_dataset_v*.py       # 多版本数据集生成脚本
├── continue_training.py         # 继续训练脚本
├── continue_training_sft.py     # 监督微调训练脚本
├── hyperparameter_search.py     # 超参数搜索工具
├── validate_notebook.ipynb      # 模型验证Jupyter笔记本
├── visualize_logs.py            # 训练日志可视化
├── test_tokenizer.py            # 分词器测试
├── reset_ckpt_config.py         # 检查点配置重置
├── run_model.py                 # 模型推理运行
├── run_model_for_state.py       # 模型状态检查
├── config.json                  # 模型配置文件
├── model_architecture_test.py   # 模型架构测试
├── model_baseline.py            # 基线模型实现
├── legacy_model.py              # 旧版模型实现
├── legacy_module.py             # 旧版模块实现
├── models_*.py                  # 不同版本的模型文件
├── data_process/                # 数据处理工具目录
│   ├── process_parquet.py       # Parquet数据处理
│   ├── extracting_json.py       # JSON数据提取
│   ├── downsample_data.py       # 数据降采样
│   ├── add_postfix.py           # 添加后缀
│   └── strip_return.py          # 移除回车符
├── hyperparameter_search_logs/  # 超参数搜索日志目录
│   ├── best_params.json         # 最优参数配置
│   └── search_results.json      # 搜索结果记录
├── bpe_tokenizer_*.json         # 预训练分词器文件
└── weight_initialization_comparison.png  # 权重初始化对比图
```

## 🚀 功能特性

### 模型架构
- 基础Transformer解码器结构实现
- 支持多头自注意力机制
- 包含现代化改进如RoPE位置编码、门控注意力（Gated Attention）
- 可配置的模型尺寸、层数和注意力头数
- 支持MoE（混合专家系统）架构
- 多种模型变体供学习对比

### 训练优化
- 梯度裁剪防止梯度爆炸
- 支持多种学习率调度策略（包括WarmUpCosine和WarmUpStableDecay）
- 检查点保存与恢复机制（支持断点续训）
- 训练过程可视化监控（TensorBoard）
- 损失函数平滑与指标跟踪
- 支持多种优化器（包括Muon、AdamW）

### 数据处理
- BPE分词器实现
- 基础数据格式支持（JSON、Parquet等）
- 数据清洗与预处理工具
- 基础批处理功能
- 简单数据处理功能

## 📊 开发进度

- [x] 基础Transformer架构实现
- [x] 各类模型变体开发
- [x] 模型训练循环实现
- [x] 数据预处理管道
- [x] BPE分词器实现
- [ ] SFT（监督微调）代码完善
- [ ] 完整模型训练流程
- [ ] 完整文档与教程
- [ ] 高级训练技巧集成
- [ ] FP8性能优化与加速

## 📚 教育价值

### 对于初学者
- 相对清晰的代码结构，易于跟随学习
- 较详细的注释解释关键概念和实现细节
- 尝试避免过度抽象，展现底层实现原理
- 逐步引导理解LM训练基础流程
- 提供多个模型版本用于对比学习

### 对于进阶学习者
- 探索不同架构变体的效果对比
- 尝试不同的训练策略和优化方法
- 进行超参数调优实验
- 实现和集成新的改进技术
- 研究权重初始化策略的影响

## ⚠️ 项目状态

该项目**正在积极开发迭代中**，持续完善功能特性和文档支持。目前仅作为个人轻量级学习项目，尚未完全开发完毕。

**注意**: 本项目主要用于教育目的，不适用于生产环境。

## 🤝 贡献

欢迎任何形式的贡献，包括但不限于：
- Bug修复
- 文档改进
- 新功能建议
- 学习心得分享
- 代码优化

## 📄 许可证

本项目遵循 [LICENSE](./LICENSE) 许可证。

## 💡 未来规划
- 完善文档和教程
- 可能会尝试更多模型架构（如RetNet等）
- 性能优化与训练加速
- 更多示例应用和用例
- 社区支持和交流

