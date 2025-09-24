import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import json
import time
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, Union, Type

from utils import model_structure
from dataset import StreamingTextDataset
from models import MyLM, MyLMArgs
from model_baseline import LLaMABaseline, GPT2Baseline
import tokenizers


@dataclass
class ModelTestConfig:
    """模型测试配置参数"""

    # 数据配置
    data_dir: str = r"data_large_ChatML.txt"
    tokenizer_dir: str = r"bpe_tokenizer_6k_0724_ChatML.json"
    log_dir: str = r"logs"
    dataset_downsample: float = 0.006
    seq_max_len: int = 192
    valset_rate: float = 0.007

    # 训练参数
    seed: int = 43
    epochs: int = 1
    batch_size: int = 32
    batch_acceleration: int = 2
    val_interval: int = 200  # 每N步进行一次验证

    # 优化参数
    learning_rate: float = 3e-4
    use_amp: bool = False

    # 模型参数
    model_args: MyLMArgs = field(
        default_factory=lambda: MyLMArgs(
            d_model=128,
            d_inner=int(((128 * (8 / 3)) // 64) * 64),
            d_head=64,
            n_heads=None,
            n_layers=2,
            vocab_size=None,
            seq_max_len=192,
            use_moe=False,
            n_experts=None,
            n_experts_per_tok=None,
            d_conv=None,
            conv_bias=None,
            ffn_bias=False,
            attn_bias=True,
            dropout=0.1,
        )
    )

    # 新增：模型类
    model_class: Type = MyLM

    # 新增：备注信息
    notes: str = "Large" 


class ModelArchitectureTester:
    """
    模型架构性能测试器
    用于测试不同模型架构的性能，记录参数量、超参数设置、训练损失等信息
    """

    def __init__(self, config: ModelTestConfig):
        self.config = config
        self._set_seed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizers.Tokenizer.from_file(config.tokenizer_dir)
        self.config.model_args.vocab_size = len(self.tokenizer.get_vocab())
        self.train_loader, self.val_loader = self._build_dataloader()
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = self._build_optimizer()

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.train_loss_log = []
        self.val_loss_log = []
        self.lr_log = []

        # 测试日志
        self.test_log = {
            "test_start_time": None,
            "test_end_time": None,
            "model_architecture": "",
            "model_parameters_count": 0,
            "hyperparameters": {},
            "training_logs": [],
            "test_results": {},
            "notes": "",  # 新增备注信息
        }

        # 训练时间记录
        self.training_start_time = None
        self.training_end_time = None
        self.total_training_time = 0

    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_model(self):
        """构建模型"""
        # 直接使用配置中的模型类
        model = self.config.model_class(self.config.model_args)
        return model

    def _build_dataloader(self):
        """构建数据加载器"""
        dataset = StreamingTextDataset(
            self.config.data_dir,
            downsample=self.config.dataset_downsample,
            seq_max_len=self.config.seq_max_len,
            tokenizer=self.tokenizer,
            re_tokenize=False,
            batch=False,
            padding_side="left",
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
            num_workers=3,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=3,
        )

        return train_loader, val_loader

    def _build_optimizer(self):
        """构建优化器"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        return optimizer

    def _train_step(self, inputs, targets):
        """单步训练"""
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.autocast(str(self.device), enabled=self.config.use_amp):
            output = self.model(inputs)
            loss = self.criterion(
                output.view(-1, self.config.model_args.vocab_size), targets.view(-1)
            )

        loss = loss / self.config.batch_acceleration
        loss.backward()

        if ((self.global_step + 1) % self.config.batch_acceleration == 0) or (
            self.global_step + 1 == len(self.train_loader)
        ):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return loss.item() * self.config.batch_acceleration

    def _validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.autocast(str(self.device), enabled=self.config.use_amp):
                    output = self.model(inputs)
                    loss = self.criterion(
                        output.view(-1, self.config.model_args.vocab_size),
                        targets.view(-1),
                    )

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    def log_model_info(self):
        """记录模型信息"""
        # 获取模型参数量
        total_params = model_structure(self.model)

        # 记录模型架构说明
        model_architecture_str = str(self.model)

        self.test_log["model_parameters_count"] = total_params
        self.test_log["model_architecture"] = model_architecture_str

        # 创建可序列化的配置字典
        config_dict = asdict(self.config)
        # 将模型类替换为类名字符串，因为类型对象无法JSON序列化
        config_dict["model_class"] = self.config.model_class.__name__
        self.test_log["hyperparameters"] = config_dict
        self.test_log["notes"] = self.config.notes  # 记录备注信息

        print(f"模型参数量: {total_params:,}")
        print(f"模型超参数: {json.dumps(config_dict, indent=2, ensure_ascii=False)}")
        if self.config.notes:
            print(f"备注信息: {self.config.notes}")

    def train_and_evaluate(self):
        """训练并评估模型"""
        print("开始模型架构性能测试...")

        self.test_log["test_start_time"] = datetime.now().isoformat()
        self.training_start_time = time.time()  # 记录训练开始时间
        # 记录模型信息
        self.log_model_info()
        self.validation_time = 0  # 累积验证时间
        self.epoch_overhead_time = 0  # 累积epoch切换时间
        # 训练过程
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch

            print(f"开始第 {epoch+1} 轮训练...")
            
            # 训练阶段
            for step, (inputs, targets) in enumerate(self.train_loader):
                if step == 0:
                    self.epoch_overhead_time += time.time() - epoch_start_time
                loss = self._train_step(inputs, targets)
                self.global_step += 1

                # 记录训练日志，排除验证时间和epoch切换时间
                train_log_entry = {
                    "epoch": epoch + 1,
                    "step": self.global_step,
                    "train_loss": round(float(loss), 6),
                    "timestamp": time.time()
                    - self.training_start_time
                    - self.validation_time
                    - self.epoch_overhead_time,
                }
                self.train_loss_log.append((self.global_step, loss))
                self.test_log["training_logs"].append(train_log_entry)

                # 每N步进行验证
                if (
                    self.config.val_interval > 0
                    and self.global_step % self.config.val_interval == 0
                ):
                    val_start_time = time.time()  # 记录验证开始时间
                    val_loss = self._validate()
                    self.val_loss_log.append((self.global_step, val_loss))
                    val_duration = time.time() - val_start_time
                    self.validation_time += val_duration
                    # 记录验证日志
                    val_log_entry = {
                        "epoch": epoch + 1,
                        "step": self.global_step,
                        "val_loss": round(float(val_loss), 6),
                        "timestamp": time.time()
                        - self.training_start_time
                        - self.validation_time
                        - self.epoch_overhead_time,
                        "type": "step_validation",
                    }
                    self.test_log["training_logs"].append(val_log_entry)

                    # 累积验证所花费的时间
                    
                    print(f"Step [{self.global_step}] 验证损失: {val_loss:.6f}")

                if step % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{self.config.epochs}], Step [{step}], Loss: {loss:.6f}"
                    )
            # 验证阶段
            epoch_end_time = time.time()  # 计算并累积epoch切换时间（包括日志打印等非训练时间）
            val_loss = self._validate()
            self.val_loss_log.append((epoch + 1, val_loss))
            epoch_time = epoch_end_time - epoch_start_time
            print(
                f"第 {epoch+1} 轮完成 - 验证损失: {val_loss:.6f}, 耗时: {epoch_time:.2f}秒"
            )
            self.epoch_overhead_time += time.time() - epoch_end_time 
            # 记录验证日志，确保包含val_loss信息
            val_log_entry = {
                "epoch": epoch + 1,
                "val_loss": round(float(val_loss), 6),
                "epoch_time": round(epoch_time, 2),
                "timestamp": time.time()
                - self.training_start_time
                - self.validation_time
                - self.epoch_overhead_time,
                "type": "epoch_validation",
            }
            self.test_log["training_logs"].append(val_log_entry)
            
            

        # 记录训练结束时间和总训练时间
        self.training_end_time = time.time()
        self.total_training_time = (
            self.training_end_time
            - self.training_start_time
            - self.validation_time
            - self.epoch_overhead_time
        )
        # 最终测试结果
        final_train_loss = self.train_loss_log[-1][1] if self.train_loss_log else 0
        final_val_loss = self.val_loss_log[-1][1] if self.val_loss_log else 0

        self.test_log["test_end_time"] = datetime.now().isoformat()
        self.test_log["test_results"] = {
            "final_train_loss": round(float(final_train_loss), 6),
            "final_val_loss": round(float(final_val_loss), 6),
            "total_training_steps": self.global_step,
            "total_epochs": self.config.epochs,
            "total_training_time": round(self.total_training_time, 2),  # 总训练时间
            "training_time_per_epoch": round(
                self.total_training_time / self.config.epochs, 2
            ),  # 每轮训练时间
        }

        print(f"训练完成!")
        print(f"最终训练损失: {final_train_loss:.6f}")
        print(f"最终验证损失: {final_val_loss:.6f}")
        print(f"总训练时间: {self.total_training_time:.2f}秒")
        print(
            f"平均每轮训练时间: {self.total_training_time / self.config.epochs:.2f}秒"
        )

    def save_test_log(self, filename: Optional[str] = None):
        """保存测试日志"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.model_class.__name__
            notes = self.config.notes if self.config.notes else ""
            if notes:
                filename = rf"{model_name}({notes.replace('/', '-')})_{timestamp}.json"
            else:
                filename = rf"{model_name}_{timestamp}.json"

        # 确保日志目录存在
        os.makedirs(self.config.log_dir, exist_ok=True)
        filepath = os.path.join(self.config.log_dir, filename)

        # 保存日志
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.test_log, f, indent=2, ensure_ascii=False)

        print(f"测试日志已保存至: {filepath}")
        return filepath

    def run_test(self, log_filename: Optional[str] = None):
        """运行完整的模型测试"""
        try:
            self.train_and_evaluate()
            log_path = self.save_test_log(log_filename)
            return log_path
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            raise


def test_model_alignment():
    """测试 MyLM 和 LLaMABaseline 实现的一致性"""
    print("开始测试 MyLM 和 LLaMABaseline 实现的一致性...")
    
    # 创建模型参数
    model_args = MyLMArgs(
        d_model=128,
        d_inner=int(((128 * (8 / 3)) // 64) * 64),
        d_head=64,
        n_heads=None,
        n_layers=2,
        vocab_size=1000,
        seq_max_len=192,
        use_moe=False,
        n_experts=None,
        n_experts_per_tok=None,
        d_conv=None,
        conv_bias=None,
        ffn_bias=False,
        attn_bias=True,
        dropout=0.1,
    )
    
    # 实例化两个模型
    my_lm_model = MyLM(model_args)
    baseline_model = LLaMABaseline(model_args)
    
    # 确保两个模型在相同的设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_lm_model = my_lm_model.to(device)
    baseline_model = baseline_model.to(device)
    
    # 设置为评估模式
    my_lm_model.eval()
    baseline_model.eval()
    
    # 创建测试输入
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, model_args.vocab_size, (batch_size, seq_length)).to(device)
    
    # 设置随机种子以确保一致性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 重新初始化 MyLM 模型的权重
    my_lm_model._reset_parameters()
    
    # 重新初始化 LLaMABaseline 模型的权重
    baseline_model._init_weights()
    
    # 强制两个模型使用相同的权重
    # 通过复制 MyLM 的权重到 LLaMABaseline
    my_lm_state_dict = my_lm_model.state_dict()
    baseline_model.load_state_dict(my_lm_state_dict)
    
    # 获取 MyLM 模型的输出
    with torch.no_grad():
        my_lm_output = my_lm_model(input_ids)
    
    # 获取 LLaMABaseline 模型的输出
    with torch.no_grad():
        baseline_output = baseline_model(input_ids)
    
    # 比较输出
    output_diff = torch.max(torch.abs(my_lm_output - baseline_output)).item()
    if torch.allclose(my_lm_output, baseline_output, atol=1e-6):
        print("[PASS] MyLM 和 LLaMABaseline 实现一致")
        print(f"  输出形状: {my_lm_output.shape}")
        print(f"  输出差异的最大值: {output_diff}")
        return True
    else:
        print("[FAIL] MyLM 和 LLaMABaseline 实现不一致")
        print(f"  MyLM 输出形状: {my_lm_output.shape}")
        print(f"  LLaMABaseline 输出形状: {baseline_output.shape}")
        print(f"  输出差异的最大值: {output_diff}")
        
        # 比较模型参数
        print("\n模型参数比较:")
        my_lm_params = list(my_lm_model.named_parameters())
        baseline_params = list(baseline_model.named_parameters())
        
        if len(my_lm_params) != len(baseline_params):
            print(f"  参数数量不一致: MyLM={len(my_lm_params)}, LLaMABaseline={len(baseline_params)}")
        else:
            print(f"  参数数量一致: {len(my_lm_params)}")
            # 比较前几个参数的名称和形状
            for i, ((name1, param1), (name2, param2)) in enumerate(zip(my_lm_params[:5], baseline_params[:5])):
                print(f"  参数 {i+1}: MyLM.{name1} {param1.shape} vs LLaMABaseline.{name2} {param2.shape}")
        
        # 检查是否有共享权重
        print("\n共享权重检查:")
        print(f"  MyLM 是否共享权重: {my_lm_model.head.weight is my_lm_model.token_embedding.weight}")
        print(f"  LLaMABaseline 是否共享权重: {baseline_model.head.weight is baseline_model.token_embedding.weight}")
        
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "align":
        # 测试模型对齐
        test_model_alignment()
    else:
        # 运行单一模型测试
        config = ModelTestConfig()
        tester = ModelArchitectureTester(config)
        tester.run_test()
