"""
超参数搜索脚本，用于寻找最佳的模型初始化参数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from itertools import product
import matplotlib.pyplot as plt
import logging

from utils import model_structure, TextGenerator, WarmUpCosineLR, DebugTimer
from dataset import StreamingTextDataset
from models import MyLMArgs, MyLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparameter_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    """搜索配置参数"""
    # 数据配置
    data_dir: str = r"data_large_ChatML.txt"
    tokenizer_dir: str = r"bpe_tokenizer_6k_0724_ChatML.json"
    log_dir: str = r"hyperparameter_search_logs"
    padding_side = "left"
    
    # 搜索参数
    init_std_range: List[float] = None  # 初始化标准差范围
    resid_scale_range: List[float] = None # 残差缩放范围
    layer_scale_range: List[float] = None  # 层缩放范围
    use_deepnet_scaling_range: List[bool] = None  # DeepNet缩放开关
    
    # 训练参数
    seed: int = 42
    epochs: int = 1  # 减少epoch数以加快搜索
    batch_size: int = 16 # 减小批次大小以适应内存
    batch_acceleration: int = 4
    dataset_downsample: float = 0.001  # 减小数据集大小以加快搜索
    valset_rate: float = 0.001
    seq_max_len: int = 192
    
    # 优化参数
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    warmup_steps: int = 1
    use_amp: bool = False
    
    # 模型参数 (使用较小的模型以加快搜索)
    model_args = MyLMArgs(
        d_model=64,
        d_inner=256,
        d_head=32,
        n_heads=None,
        n_layers=1,
        vocab_size=None,  # 稍后设置
        seq_max_len=seq_max_len,
        use_moe=False,
        n_experts=None,
        n_experts_per_tok=None,
        d_conv=None,
        conv_bias=None,
        ffn_bias=False,
        attn_bias=True,
        dropout=0.1,
        init_std=0.02,  # 基础初始化标准差
        resid_pdrop=0.1,  # 残差连接dropout
        resid_scale=1.0,  # 残差流缩放参数
        layer_scale=1.0,  # 层缩放参数
        use_deepnet_scaling=True  # 是否使用DeepNet缩放策略
    )
    
    def __post_init__(self):
        # 设置默认搜索范围
        if self.init_std_range is None:
            self.init_std_range = [0.02, 0.1, 0.3, 0.7]
        if self.resid_scale_range is None:
            self.resid_scale_range = [1.0]
        if self.layer_scale_range is None:
            self.layer_scale_range = [1.0]
        if self.use_deepnet_scaling_range is None:
            self.use_deepnet_scaling_range = [False, True]

class HyperparameterSearcher:
    def __init__(self, config: SearchConfig):
        self.config = config
        self._set_seed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载tokenizer
        import tokenizers
        self.tokenizer = tokenizers.Tokenizer.from_file(config.tokenizer_dir)
        self.config.model_args.vocab_size = len(self.tokenizer.get_vocab())
        
        # 创建日志目录
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 存储搜索结果
        self.search_results = []
        
    def _set_seed(self):
        """设置随机种子"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _build_dataloader(self, downsample_rate=None):
        """构建数据加载器"""
        if downsample_rate is None:
            downsample_rate = self.config.dataset_downsample
            
        dataset = StreamingTextDataset(
            self.config.data_dir,
            downsample=downsample_rate,
            seq_max_len=self.config.seq_max_len,
            tokenizer=self.tokenizer,
            re_tokenize=False,
            batch=False,
            padding_side=self.config.padding_side,
        )
        
        val_dataset_len = int(len(dataset) * self.config.valset_rate)
        train_dataset_len = len(dataset) - val_dataset_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_dataset_len, val_dataset_len]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=1  # 减少worker数以节省内存
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=1
        )

        return train_loader, val_loader

    def _build_model_with_params(self, init_std, resid_scale, layer_scale, use_deepnet_scaling):
        """构建具有指定初始化参数的模型"""
        # 创建模型参数的副本并修改初始化参数
        model_args = MyLMArgs(
            d_model=self.config.model_args.d_model,
            d_inner=self.config.model_args.d_inner,
            d_head=self.config.model_args.d_head,
            n_heads=self.config.model_args.n_heads,
            n_layers=self.config.model_args.n_layers,
            vocab_size=self.config.model_args.vocab_size,
            seq_max_len=self.config.model_args.seq_max_len,
            use_moe=self.config.model_args.use_moe,
            n_experts=self.config.model_args.n_experts,
            n_experts_per_tok=self.config.model_args.n_experts_per_tok,
            d_conv=self.config.model_args.d_conv,
            conv_bias=self.config.model_args.conv_bias,
            ffn_bias=self.config.model_args.ffn_bias,
            attn_bias=self.config.model_args.attn_bias,
            dropout=self.config.model_args.dropout,
            init_std=init_std,
            resid_pdrop=self.config.model_args.resid_pdrop,
            resid_scale=resid_scale,
            layer_scale=layer_scale,
            use_deepnet_scaling=use_deepnet_scaling
        )
        
        model = MyLM(model_args)
        return model

    def _train_single_epoch(self, model, train_loader, optimizer, criterion, scaler, use_amp):
        """训练单个epoch"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            
            with torch.autocast(str(self.device), enabled=use_amp):
                output = model(inputs)
                loss = criterion(
                    output.view(-1, self.config.model_args.vocab_size),
                    targets.view(-1)
                )

            loss = loss / self.config.batch_acceleration
            
            scaler.scale(loss).backward()
            
            if num_batches % self.config.batch_acceleration == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            
            total_loss += loss.item() * self.config.batch_acceleration
            num_batches += 1
            
            # 限制训练批次以加快搜索
            if num_batches >= 20:  # 限制每个epoch只训练20个批次
                break
                
        return total_loss / max(1, num_batches)

    def _validate(self, model, val_loader, criterion, use_amp):
        """验证模型"""
        model.eval()
        val_loss_sum = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                with torch.autocast(str(self.device), enabled=use_amp):
                    output = model(inputs)
                    loss = criterion(
                        output.view(-1, self.config.model_args.vocab_size),
                        targets.view(-1),
                    )
                val_loss_sum += loss.item()
                num_batches += 1
                
                # 限制验证批次以加快搜索
                if num_batches >= 10:  # 限制验证10个批次
                    break
                    
        return val_loss_sum / max(1, num_batches)

    def _evaluate_model_quality(self, model, val_loader, criterion, use_amp):
        """评估模型质量的额外指标"""
        model.eval()
        
        # 1. 计算验证损失
        val_loss = self._validate(model, val_loader, criterion, use_amp)
        
        # 2. 计算模型的梯度范数（评估训练稳定性）
        total_grad_norm = 0
        num_params = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                num_params += 1
        
        avg_grad_norm = total_grad_norm / max(1, num_params) if num_params > 0 else 0
        
        # 3. 计算输出的方差（评估梯度消失/爆炸）
        with torch.no_grad():
            # 从验证集中取一个批次
            inputs, targets = next(iter(val_loader))
            inputs = inputs.to(self.device)
            
            output = model(inputs)
            output_variance = output.var().item()
        
        # 4. 计算激活值统计（评估网络内部状态）
        activation_stats = {}
        def get_activation_stats(module, input, output):
            if isinstance(module, nn.Linear):
                key = f"{module.__class__.__name__}_{id(module)}"
                activation_stats[key] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'var': output.var().item()
                }
        
        # 注册钩子以捕获激活统计
        hooks = []
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(get_activation_stats))
        
        # 前向传播以捕获激活
        with torch.no_grad():
            dummy_input = torch.randint(0, self.config.model_args.vocab_size,
                                      (1, self.config.seq_max_len)).to(self.device)
            model(dummy_input)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 计算平均激活统计
        avg_activation_mean = np.mean([v['mean'] for v in activation_stats.values()]) if activation_stats else 0
        avg_activation_var = np.mean([v['var'] for v in activation_stats.values()]) if activation_stats else 0
        
        return {
            'val_loss': val_loss,
            'grad_norm': avg_grad_norm,
            'output_variance': output_variance,
            'activation_mean': avg_activation_mean,
            'activation_var': avg_activation_var
        }

    def _train_and_evaluate(self, init_std, resid_scale, layer_scale, use_deepnet_scaling):
        """训练并评估一组超参数"""
        logger.info(f"测试参数: init_std={init_std}, resid_scale={resid_scale}, "
              f"layer_scale={layer_scale}, use_deepnet_scaling={use_deepnet_scaling}")
        
        # 构建模型
        model = self._build_model_with_params(init_std, resid_scale, layer_scale, use_deepnet_scaling)
        model = model.to(self.device)
        
        # 构建数据加载器
        train_loader, val_loader = self._build_dataloader()
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            amsgrad=False,
            betas=(0.85, 0.999),
            eps=1e-6,
            weight_decay=0.005,
        )
        
        scaler = torch.GradScaler(self.device, enabled=self.config.use_amp)
        
        # 训练循环
        best_val_loss = float('inf')
        all_val_losses = []
        
        for epoch in range(self.config.epochs):
            # 训练
            train_loss = self._train_single_epoch(model, train_loader, optimizer, criterion, scaler, self.config.use_amp)
            
            # 验证
            val_loss = self._validate(model, val_loader, criterion, self.config.use_amp)
            all_val_losses.append(val_loss)
            
            logger.info(f"  Epoch {epoch+1}/{self.config.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        # 评估模型质量
        quality_metrics = self._evaluate_model_quality(
            model, val_loader, criterion, self.config.use_amp
        )
        
        logger.info(f" 最佳验证损失: {best_val_loss:.4f}")
        logger.info(f" 最终验证损失: {quality_metrics['val_loss']:.4f}")
        logger.info(f" 平均梯度范数: {quality_metrics['grad_norm']:.6f}")
        logger.info(f" 输出方差: {quality_metrics['output_variance']:.6f}")
        logger.info(f" 激活均值: {quality_metrics['activation_mean']:.6f}")
        logger.info(f" 激活方差: {quality_metrics['activation_var']:.6f}\n")
        
        return {
            'val_loss': quality_metrics['val_loss'],
            'grad_norm': quality_metrics['grad_norm'],
            'output_variance': quality_metrics['output_variance'],
            'activation_mean': quality_metrics['activation_mean'],
            'activation_var': quality_metrics['activation_var'],
            'all_val_losses': all_val_losses
        }

    def search(self):
        """执行超参数搜索"""
        logger.info("开始超参数搜索...")
        logger.info(f"搜索参数组合: {len(self.config.init_std_range)} x "
              f"{len(self.config.resid_scale_range)} x "
              f"{len(self.config.layer_scale_range)} x "
              f"{len(self.config.use_deepnet_scaling_range)} = "
              f"{len(list(self._get_param_combinations()))} 组")
        
        start_time = time.time()
        
        for i, (init_std, resid_scale, layer_scale, use_deepnet_scaling) in enumerate(self._get_param_combinations()):
            logger.info(f"进度: {i+1}/{len(list(self._get_param_combinations()))}")
            
            try:
                eval_results = self._train_and_evaluate(init_std, resid_scale, layer_scale, use_deepnet_scaling)
                
                result = {
                    'init_std': init_std,
                    'resid_scale': resid_scale,
                    'layer_scale': layer_scale,
                    'use_deepnet_scaling': use_deepnet_scaling,
                    'val_loss': eval_results['val_loss'],
                    'grad_norm': eval_results['grad_norm'],
                    'output_variance': eval_results['output_variance'],
                    'activation_mean': eval_results['activation_mean'],
                    'activation_var': eval_results['activation_var'],
                    'all_val_losses': eval_results['all_val_losses'],
                    'timestamp': time.time()
                }
                
                self.search_results.append(result)
                
                # 保存中间结果
                self._save_results()
                
            except Exception as e:
                logger.error(f"参数组合 {init_std}, {resid_scale}, {layer_scale}, {use_deepnet_scaling} 训练失败: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        total_time = time.time() - start_time
        logger.info(f"搜索完成! 总耗时: {total_time/60:.2f} 分钟")
        
        # 保存最终结果
        self._save_results()
        
        # 找出最佳参数
        self._find_best_params()
        
        # 可视化结果
        self._visualize_results()
        
    def _get_param_combinations(self):
        """获取参数组合"""
        return product(
            self.config.init_std_range,
            self.config.resid_scale_range,
            self.config.layer_scale_range,
            self.config.use_deepnet_scaling_range
        )
    
    def _save_results(self):
        """保存搜索结果"""
        results_path = os.path.join(self.config.log_dir, "search_results.json")
        # 保存结果的副本，不包含all_val_losses以减小文件大小
        results_to_save = []
        for result in self.search_results:
            result_copy = result.copy()
            if 'all_val_losses' in result_copy:
                del result_copy['all_val_losses']  # 避免保存过大的训练历史
            results_to_save.append(result_copy)
            
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    def _find_best_params(self):
        """找出最佳参数"""
        if not self.search_results:
            logger.warning("没有搜索结果!")
            return
            
        # 按验证损失排序
        best_result_by_loss = min(self.search_results, key=lambda x: x['val_loss'])
        
        logger.info("\n按验证损失排序的最佳参数组合:")
        logger.info(f" init_std: {best_result_by_loss['init_std']}")
        logger.info(f" resid_scale: {best_result_by_loss['resid_scale']}")
        logger.info(f"  layer_scale: {best_result_by_loss['layer_scale']}")
        logger.info(f"  use_deepnet_scaling: {best_result_by_loss['use_deepnet_scaling']}")
        logger.info(f" 验证损失: {best_result_by_loss['val_loss']:.4f}")
        logger.info(f" 梯度范数: {best_result_by_loss['grad_norm']:.6f}")
        logger.info(f" 输出方差: {best_result_by_loss['output_variance']:.6f}")
        logger.info(f" 激活均值: {best_result_by_loss['activation_mean']:.6f}")
        logger.info(f" 激活方差: {best_result_by_loss['activation_var']:.6f}")
        
        # 按梯度范数排序（评估训练稳定性）
        best_result_by_grad = min(self.search_results, key=lambda x: abs(x['grad_norm'] - 1.0))  # 目标梯度范数接近1
        
        logger.info("\n按梯度稳定性排序的最佳参数组合:")
        logger.info(f" init_std: {best_result_by_grad['init_std']}")
        logger.info(f"  resid_scale: {best_result_by_grad['resid_scale']}")
        logger.info(f"  layer_scale: {best_result_by_grad['layer_scale']}")
        logger.info(f"  use_deepnet_scaling: {best_result_by_grad['use_deepnet_scaling']}")
        logger.info(f" 验证损失: {best_result_by_grad['val_loss']:.4f}")
        logger.info(f" 梯度范数: {best_result_by_grad['grad_norm']:.6f}")
        logger.info(f" 输出方差: {best_result_by_grad['output_variance']:.6f}")
        logger.info(f" 激活均值: {best_result_by_grad['activation_mean']:.6f}")
        logger.info(f" 激活方差: {best_result_by_grad['activation_var']:.6f}")
        
        # 综合评估（平衡验证损失和梯度稳定性）
        def combined_score(result):
            # 标准化验证损失（越小越好）
            norm_val_loss = result['val_loss'] / max(r['val_loss'] for r in self.search_results)
            # 标准化梯度范数（接近1越好）
            norm_grad_norm = abs(result['grad_norm'] - 1.0) / max(abs(r['grad_norm'] - 1.0) for r in self.search_results)
            # 综合得分（越小越好）
            return 0.7 * norm_val_loss + 0.3 * norm_grad_norm
        
        best_result_combined = min(self.search_results, key=combined_score)
        
        logger.info("\n综合评估的最佳参数组合:")
        logger.info(f"  init_std: {best_result_combined['init_std']}")
        logger.info(f"  resid_scale: {best_result_combined['resid_scale']}")
        logger.info(f"  layer_scale: {best_result_combined['layer_scale']}")
        logger.info(f"  use_deepnet_scaling: {best_result_combined['use_deepnet_scaling']}")
        logger.info(f" 验证损失: {best_result_combined['val_loss']:.4f}")
        logger.info(f" 梯度范数: {best_result_combined['grad_norm']:.6f}")
        logger.info(f" 输出方差: {best_result_combined['output_variance']:.6f}")
        logger.info(f" 激活均值: {best_result_combined['activation_mean']:.6f}")
        logger.info(f" 激活方差: {best_result_combined['activation_var']:.6f}")
        
        # 保存最佳参数
        best_params_path = os.path.join(self.config.log_dir, "best_params.json")
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump({
                'by_loss': best_result_by_loss,
                'by_stability': best_result_by_grad,
                'combined': best_result_combined
            }, f, indent=2, ensure_ascii=False)
    
    def _visualize_results(self):
        """可视化搜索结果"""
        if not self.search_results:
            logger.warning("没有结果可可视化!")
            return
            
        # 提取结果数据
        init_stds = [r['init_std'] for r in self.search_results]
        resid_scales = [r['resid_scale'] for r in self.search_results]
        layer_scales = [r['layer_scale'] for r in self.search_results]
        use_deepnet_scaling_vals = [r['use_deepnet_scaling'] for r in self.search_results]
        val_losses = [r['val_loss'] for r in self.search_results]
        grad_norms = [r['grad_norm'] for r in self.search_results]
        output_variances = [r['output_variance'] for r in self.search_results]
        activation_means = [r['activation_mean'] for r in self.search_results]
        activation_vars = [r['activation_var'] for r in self.search_results]
        
        # 创建图表
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('超参数搜索结果可视化', fontsize=16)
        
        # 1. init_std vs val_loss
        scatter1 = axes[0, 0].scatter(init_stds, val_losses, c=grad_norms, cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('初始化标准差 (init_std)')
        axes[0, 0].set_ylabel('验证损失')
        axes[0, 0].set_title('初始化标准差 vs 验证损失 (颜色=梯度范数)')
        axes[0, 0].grid(True)
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # 2. resid_scale vs val_loss
        scatter2 = axes[0, 1].scatter(resid_scales, val_losses, c=output_variances, cmap='plasma', alpha=0.6)
        axes[0, 1].set_xlabel('残差缩放 (resid_scale)')
        axes[0, 1].set_ylabel('验证损失')
        axes[0, 1].set_title('残差缩放 vs 验证损失 (颜色=输出方差)')
        axes[0, 1].grid(True)
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # 3. layer_scale vs val_loss
        scatter3 = axes[0, 2].scatter(layer_scales, val_losses, c=init_stds, cmap='coolwarm', alpha=0.6)
        axes[0, 2].set_xlabel('层缩放 (layer_scale)')
        axes[0, 2].set_ylabel('验证损失')
        axes[0, 2].set_title('层缩放 vs 验证损失 (颜色=初始化标准差)')
        axes[0, 2].grid(True)
        plt.colorbar(scatter3, ax=axes[0, 2])
        
        # 4. 按use_deepnet_scaling分组的箱线图
        deepnet_true_losses = [r['val_loss'] for r in self.search_results if r['use_deepnet_scaling']]
        deepnet_false_losses = [r['val_loss'] for r in self.search_results if not r['use_deepnet_scaling']]
        
        axes[1, 0].boxplot([deepnet_false_losses, deepnet_true_losses], labels=['False', 'True'])
        axes[1, 0].set_ylabel('验证损失')
        axes[1, 0].set_title('DeepNet缩放对验证损失的影响')
        axes[1, 0].grid(True)
        
        # 5. 梯度范数 vs 输出方差
        axes[1, 1].scatter(grad_norms, output_variances, c=val_losses, cmap='Reds', alpha=0.6)
        axes[1, 1].set_xlabel('梯度范数')
        axes[1, 1].set_ylabel('输出方差')
        axes[1, 1].set_title('梯度稳定性 vs 输出方差 (颜色=验证损失)')
        axes[1, 1].grid(True)
        plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        
        # 6. 激活统计分析
        axes[1, 2].scatter(activation_means, activation_vars, c=val_losses, cmap='viridis', alpha=0.6)
        axes[1, 2].set_xlabel('激活均值')
        axes[1, 2].set_ylabel('激活方差')
        axes[1, 2].set_title('激活统计分析 (颜色=验证损失)')
        axes[1, 2].grid(True)
        plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
        
        # 7. 最佳结果对比 (Top 10 by validation loss)
        sorted_results = sorted(self.search_results, key=lambda x: x['val_loss'])[:10] # 取前10个最佳结果
        top_losses = [r['val_loss'] for r in sorted_results]
        top_labels = [f"std:{r['init_std']:.3f}\nres:{r['resid_scale']:.2f}" for r in sorted_results]
        
        axes[2, 0].barh(range(len(top_losses)), top_losses)
        axes[2, 0].set_xlabel('验证损失')
        axes[2, 0].set_ylabel('参数组合 (top 10)')
        axes[2, 0].set_title('Top 10 参数组合性能对比')
        axes[2, 0].set_yticks(range(len(top_labels)))
        axes[2, 0].set_yticklabels(top_labels)
        axes[2, 0].grid(True, axis='x')
        
        # 8. 参数组合热力图 (init_std vs resid_scale)
        unique_init_stds = sorted(set(init_stds))
        unique_resid_scales = sorted(set(resid_scales))
        
        # 创建热力图数据
        heatmap_data = np.full((len(unique_resid_scales), len(unique_init_stds)), np.nan)
        for i, resid_scale in enumerate(unique_resid_scales):
            for j, init_std in enumerate(unique_init_stds):
                matching_results = [r['val_loss'] for r in self.search_results
                                  if r['init_std'] == init_std and r['resid_scale'] == resid_scale]
                if matching_results:
                    heatmap_data[i, j] = np.mean(matching_results)
        
        im = axes[2, 1].imshow(heatmap_data, cmap='viridis', aspect='auto', origin='upper')
        axes[2, 1].set_xticks(range(len(unique_init_stds)))
        axes[2, 1].set_yticks(range(len(unique_resid_scales)))
        axes[2, 1].set_xticklabels([f'{x:.3f}' for x in unique_init_stds])
        axes[2, 1].set_yticklabels([f'{x:.2f}' for x in unique_resid_scales])
        axes[2, 1].set_xlabel('初始化标准差')
        axes[2, 1].set_ylabel('残差缩放')
        axes[2, 1].set_title('验证损失热力图 (init_std vs resid_scale)')
        plt.colorbar(im, ax=axes[2, 1])
        
        # 9. 收敛性分析
        # 显示不同参数组合的训练收敛趋势
        if len(self.search_results) > 0:
            # 选择前几个最佳结果绘制训练曲线
            top_3_results = sorted(self.search_results, key=lambda x: x['val_loss'])[:3]
            for i, result in enumerate(top_3_results):
                if 'all_val_losses' in result and result['all_val_losses']:
                    axes[2, 2].plot(range(len(result['all_val_losses'])),
                                   result['all_val_losses'],
                                   label=f"std:{result['init_std']},res:{result['resid_scale']}",
                                   marker='o')
            
            axes[2, 2].set_xlabel('Epoch')
            axes[2, 2].set_ylabel('验证损失')
            axes[2, 2].set_title('Top 3 最佳参数的训练收敛曲线')
            axes[2, 2].legend()
            axes[2, 2].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.config.log_dir, "hyperparameter_search_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"结果图表已保存至: {plot_path}")
        plt.show()

def main():
    """主函数"""
    config = SearchConfig()
    searcher = HyperparameterSearcher(config)
    searcher.search()

if __name__ == "__main__":
    main()