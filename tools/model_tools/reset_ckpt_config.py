import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import asdict
import json
import argparse
from typing import Optional, Dict, Any
# 注释掉不需要的导入，因为我们只是修改checkpoint而不创建模型
# from utils import WarmUpCosineLR, WarmUpStableDecayLR
# from models import MyLMArgs, MyLM
# from pre_train import TrainingConfig


def load_checkpoint(checkpoint_path: str):
    """加载checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    print(f"加载checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"成功加载checkpoint，包含以下键: {list(checkpoint.keys())}")
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"加载checkpoint失败: {str(e)}")


def print_checkpoint_info(checkpoint: Dict):
    """打印checkpoint的详细信息"""
    print("\n=== Checkpoint 信息 ===")
    for key, value in checkpoint.items():
        if key == 'model_state_dict':
            print(f"- {key}: 包含 {len(value)} 个参数张量")
        elif key == 'optimizer_states':
            print(f"- {key}: 包含 {len(value)} 个优化器状态")
            for i, opt_state in enumerate(value):
                if 'param_groups' in opt_state:
                    print(f"  - 优化器 {i}: {len(opt_state['param_groups'])} 个参数组")
                    for j, param_group in enumerate(opt_state['param_groups']):
                        print(f"    - 参数组 {j}: lr={param_group.get('lr', 'N/A')}, "
                              f"weight_decay={param_group.get('weight_decay', 'N/A')}")
                # 打印优化器状态的关键信息，确保动量等状态被保留
                if 'state' in opt_state:
                    print(f"    - 优化器状态: 包含 {len(opt_state['state'])} 个参数的状态")
                    # 显示前几个参数的状态信息（如动量等）
                    state_keys = list(opt_state['state'].keys())
                    if state_keys:
                        sample_param_id = state_keys[0]
                        sample_state = opt_state['state'][sample_param_id]
                        print(f"      示例参数状态: {list(sample_state.keys())}")
        elif key == 'scheduler_states':
            print(f"- {key}: 包含 {len(value)} 个调度器状态")
        elif key == 'rng_states':
            print(f"- {key}: 随机数状态")
        else:
            print(f"- {key}: {value}")
    print("========================\n")


def save_checkpoint(checkpoint: Dict, save_path: str):
    """保存checkpoint"""
    # 确保输出目录存在
    output_dir = os.path.dirname(save_path)
    if output_dir:  # 如果路径包含目录，则创建目录
        os.makedirs(output_dir, exist_ok=True)
    print(f"保存checkpoint到: {save_path}")
    try:
        torch.save(checkpoint, save_path)
        print("保存成功")
    except Exception as e:
        raise RuntimeError(f"保存checkpoint失败: {str(e)}")


def update_optimizer_config(checkpoint: Dict, new_lr: Optional[float] = None,
                           new_weight_decay: Optional[float] = None,
                           new_optimizer_params: Optional[Dict] = None):
    """更新优化器配置，确保保留所有状态信息（如动量、历史梯度等）"""
    if "optimizer_states" in checkpoint and checkpoint["optimizer_states"]:
        # 处理多个优化器
        for i, opt_state in enumerate(checkpoint["optimizer_states"]):
            # 保留原始的state状态（包括动量、历史梯度等）
            original_state = opt_state.get('state', {}).copy()
            
            if new_lr is not None:
                # 更新学习率
                for param_group in opt_state['param_groups']:
                    param_group['lr'] = new_lr
                    if new_lr != param_group.get('initial_lr', param_group['lr']):
                        param_group['initial_lr'] = new_lr
            
            if new_weight_decay is not None:
                # 更新权重衰减
                for param_group in opt_state['param_groups']:
                    param_group['weight_decay'] = new_weight_decay
            
            if new_optimizer_params:
                # 更新其他优化器参数
                for param_group in opt_state['param_groups']:
                    for key, value in new_optimizer_params.items():
                        if key in param_group:
                            param_group[key] = value
            
            # 确保原始的state状态信息得到保留
            opt_state['state'] = original_state
            checkpoint["optimizer_states"][i] = opt_state
    else:
        # 兼容旧版本checkpoint
        if "optimizer_state_dict" in checkpoint:
            opt_state = checkpoint["optimizer_state_dict"]
            # 保留原始的state状态（包括动量、历史梯度等）
            original_state = opt_state.get('state', {}).copy()
            
            if new_lr is not None:
                for param_group in opt_state['param_groups']:
                    param_group['lr'] = new_lr
                    if new_lr != param_group.get('initial_lr', param_group['lr']):
                        param_group['initial_lr'] = new_lr
        
            if new_weight_decay is not None:
                for param_group in opt_state['param_groups']:
                    param_group['weight_decay'] = new_weight_decay
        
            if new_optimizer_params:
                for param_group in opt_state['param_groups']:
                    for key, value in new_optimizer_params.items():
                        if key in param_group:
                            param_group[key] = value
            
            # 确保原始的state状态信息得到保留
            opt_state['state'] = original_state
            checkpoint["optimizer_state_dict"] = opt_state
        else:
            print("警告: checkpoint中未找到optimizer_states或optimizer_state_dict")
    
    return checkpoint


def update_scheduler_config(checkpoint: Dict, new_total_steps: Optional[int] = None,
                           new_warmup_steps: Optional[int] = None,
                           new_min_lr: Optional[float] = None,
                           new_stable_steps: Optional[int] = None,
                           new_decay_mode: Optional[str] = None):
    """更新学习率调度器配置"""
    if "scheduler_states" in checkpoint and checkpoint["scheduler_states"]:
        # 处理多个调度器
        for i, sched_state in enumerate(checkpoint["scheduler_states"]):
            # 直接更新调度器参数
            if new_total_steps is not None:
                # 对于WarmUpStableDecayLR等调度器，可能需要更新total_steps
                sched_state['last_epoch'] = min(sched_state.get('last_epoch', 0), new_total_steps)
                # 如果调度器有total_steps等参数，需要根据实际调度器类型进行更新
                if 'total_steps' in sched_state:
                    sched_state['total_steps'] = new_total_steps
                # 更新base_lrs以反映新的参数
                if 'base_lrs' in sched_state and '_step_count' in sched_state:
                    sched_state['_step_count'] = min(sched_state['_step_count'], new_total_steps)
            
            if new_warmup_steps is not None and 'warmup_steps' in sched_state:
                sched_state['warmup_steps'] = new_warmup_steps
            if new_min_lr is not None and 'min_lr' in sched_state:
                sched_state['min_lr'] = new_min_lr
            if new_stable_steps is not None and 'stable_steps' in sched_state:
                sched_state['stable_steps'] = new_stable_steps
            if new_decay_mode is not None and 'decay_mode' in sched_state:
                sched_state['decay_mode'] = new_decay_mode
            
            checkpoint["scheduler_states"][i] = sched_state
    else:
        # 兼容旧版本checkpoint
        if "scheduler_state_dict" in checkpoint:
            sched_state = checkpoint["scheduler_state_dict"]
            # 直接更新调度器参数
            if new_total_steps is not None:
                sched_state['last_epoch'] = min(sched_state.get('last_epoch', 0), new_total_steps)
                if 'total_steps' in sched_state:
                    sched_state['total_steps'] = new_total_steps
            
            if new_warmup_steps is not None and 'warmup_steps' in sched_state:
                sched_state['warmup_steps'] = new_warmup_steps
            if new_min_lr is not None and 'min_lr' in sched_state:
                sched_state['min_lr'] = new_min_lr
            if new_stable_steps is not None and 'stable_steps' in sched_state:
                sched_state['stable_steps'] = new_stable_steps
            if new_decay_mode is not None and 'decay_mode' in sched_state:
                sched_state['decay_mode'] = new_decay_mode
            
            checkpoint["scheduler_state_dict"] = sched_state
        else:
            print("警告: checkpoint中未找到scheduler_states或scheduler_state_dict")
    
    # 更新checkpoint中的训练配置参数（如果存在）
    if 'training_config' in checkpoint:
        config = checkpoint['training_config']
        if new_total_steps is not None:
            config['total_steps'] = new_total_steps
        if new_warmup_steps is not None:
            config['warmup_steps'] = new_warmup_steps
        if new_min_lr is not None:
            config['min_lr'] = new_min_lr
        if new_stable_steps is not None:
            config['stable_steps'] = new_stable_steps
        if new_decay_mode is not None:
            config['decay_mode'] = new_decay_mode
        checkpoint['training_config'] = config
    else:
        # 如果不存在training_config，只创建提供了非None值的参数
        config = {}
        if new_total_steps is not None:
            config['total_steps'] = new_total_steps
        if new_warmup_steps is not None:
            config['warmup_steps'] = new_warmup_steps
        if new_min_lr is not None:
            config['min_lr'] = new_min_lr
        if new_stable_steps is not None:
            config['stable_steps'] = new_stable_steps
        if new_decay_mode is not None:
            config['decay_mode'] = new_decay_mode
        if config:  # 只有当至少有一个参数被设置时才添加training_config
            checkpoint['training_config'] = config
    
    return checkpoint

def reset_ckpt_config(ckpt_path: str, output_path: str,
                     learning_rate: Optional[float] = None,
                     weight_decay: Optional[float] = None,
                     total_steps: Optional[int] = None,
                     warmup_steps: Optional[int] = None,
                     min_lr: Optional[float] = None,
                     stable_steps: Optional[int] = None,
                     decay_mode: Optional[str] = None,
                     optimizer_params: Optional[Dict] = None,
                     show_info: bool = False):
    """
    重置checkpoint的配置参数
    
    Args:
        ckpt_path: 输入checkpoint路径
        output_path: 输出checkpoint路径
        learning_rate: 新的学习率
        weight_decay: 新的权重衰减
        total_steps: 新的总步数
        warmup_steps: 新的预热步数
        min_lr: 新的最小学习率
        stable_steps: 新的稳定步数
        decay_mode: 新的衰减模式
        optimizer_params: 其他优化器参数
        show_info: 是否显示checkpoint的详细信息
    """
    # 加载checkpoint
    checkpoint = load_checkpoint(ckpt_path)
    
    if show_info:
        print_checkpoint_info(checkpoint)
    
    print("开始更新checkpoint配置...")
    
    # 更新优化器配置
    if learning_rate is not None or weight_decay is not None or optimizer_params is not None:
        checkpoint = update_optimizer_config(
            checkpoint,
            new_lr=learning_rate,
            new_weight_decay=weight_decay,
            new_optimizer_params=optimizer_params
        )
        print(f"已更新优化器配置: lr={learning_rate}, weight_decay={weight_decay}")
    
    # 更新学习率调度器配置
    if any(param is not None for param in [total_steps, warmup_steps, min_lr, stable_steps, decay_mode]):
        checkpoint = update_scheduler_config(
            checkpoint,
            new_total_steps=total_steps,
            new_warmup_steps=warmup_steps,
            new_min_lr=min_lr,
            new_stable_steps=stable_steps,
            new_decay_mode=decay_mode
        )
        print(f"已更新调度器配置: total_steps={total_steps}, warmup_steps={warmup_steps}, "
              f"min_lr={min_lr}, stable_steps={stable_steps}, decay_mode={decay_mode}")
    
    # 保存修改后的checkpoint
    save_checkpoint(checkpoint, output_path)
    print(f"修改后的checkpoint已保存到: {output_path}")
    
    if show_info:
        print("修改后的checkpoint信息:")
        print_checkpoint_info(checkpoint)
    
    return checkpoint


def main():
    # 使用代码中的配置参数
    optimizer_params = {}
    if NEW_LEARNING_RATE or NEW_WEIGHT_DECAY:
        if NEW_LEARNING_RATE:
            optimizer_params['lr'] = NEW_LEARNING_RATE
        if NEW_WEIGHT_DECAY:
            optimizer_params['weight_decay'] = NEW_WEIGHT_DECAY
    
    # 重置配置
    reset_ckpt_config(
        ckpt_path=CKPT_PATH,
        output_path=OUTPUT_PATH,
        learning_rate=NEW_LEARNING_RATE,
        weight_decay=NEW_WEIGHT_DECAY,
        total_steps=NEW_TOTAL_STEPS,
        warmup_steps=NEW_WARMUP_STEPS,
        min_lr=NEW_MIN_LR,
        stable_steps=NEW_STABLE_STEPS,
        decay_mode=NEW_DECAY_MODE,
        optimizer_params=optimizer_params if optimizer_params else None,
        show_info=SHOW_INFO
    )

# 配置参数 - 在此处修改需要的训练配置参数
CKPT_PATH = "ckpt\ckpt_epoch_0_step_11000 origin.pth"  # 输入checkpoint路径
OUTPUT_PATH = "ckpt\ckpt_epoch_0_step_11000 modified_standerd.pth"  # 输出checkpoint路径
NEW_LEARNING_RATE = None # 新的学习率 (设为None则不修改)
NEW_WEIGHT_DECAY = None # 新的权重衰减 (设为None则不修改)
NEW_TOTAL_STEPS = None # 新的总训练步数 (设为None则不修改)
NEW_WARMUP_STEPS = None  # 新的预热步数 (设为None则不修改)
NEW_MIN_LR = None  # 新的最小学习率 (设为None则不修改)
NEW_STABLE_STEPS = 3991  # 新的稳定步数 (设为None则不修改)
NEW_DECAY_MODE = "exp" # 新的衰减模式 ("linear" or "exp") (设为None则不修改)
SHOW_INFO = True  # 是否显示checkpoint的详细信息


if __name__ == "__main__":
    main()