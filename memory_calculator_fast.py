import torch
import math
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Union

class Precision(Enum):
    FP32 = 4  # 4 bytes per parameter
    FP16 = 2  # 2 bytes per parameter
    BF16 = 2  # 2 bytes per parameter

@dataclass
class ModelConfig:
    """模型配置类"""
    model_size_in_billions: float  # 模型大小（以十亿参数为单位）
    sequence_length: int  # 序列长度
    hidden_size: int  # 隐藏层大小
    batch_size: int  # 批次大小
    num_layers: int  # 层数
    num_attention_heads: int  # 注意力头数
    optimizer_type: str = 'adam'  # 优化器类型
    precision: Precision = Precision.FP32  # 精度类型
    tensor_parallel_size: int = 1  # 张量并行大小
    pipeline_parallel_size: int = 1  # 流水线并行大小

class ModelMemoryCalculator:
    def __init__(
        self,
        model_size_in_billions: float,
        sequence_length: int,
        hidden_size: int,
        batch_size: int,
        num_layers: int,
        num_attention_heads: int,
        precision: Precision = Precision.FP32,
        optimizer_type: str = 'adam',
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1
    ):
        """
        初始化显存计算器
        :param model_size_in_billions: 模型大小（以十亿参数为单位）
        :param sequence_length: 序列长度
        :param hidden_size: 隐藏层大小
        :param batch_size: 批次大小
        :param num_layers: 层数
        :param num_attention_heads: 注意力头数
        :param precision: 精度类型，可选 FP32, FP16, BF16
        :param optimizer_type: 优化器类型 ('adam' 或 'sgd')
        :param tensor_parallel_size: 张量并行大小
        :param pipeline_parallel_size: 流水线并行大小
        """
        self.config = ModelConfig(
            model_size_in_billions=model_size_in_billions,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            batch_size=batch_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            optimizer_type=optimizer_type,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            precision=precision
        )
        self.bytes_per_param = precision.value

    def calculate_model_memory(self) -> float:
        """
        计算模型参数占用的显存
        :return: 显存占用（以GB为单位）
        """
        # 考虑张量并行和流水线并行的影响
        params_memory = (
            self.config.model_size_in_billions * 
            self.bytes_per_param / 
            (self.config.tensor_parallel_size * self.config.pipeline_parallel_size)
        )
        return params_memory

    def calculate_gradient_memory(self) -> float:
        """
        计算梯度占用的显存
        :return: 显存占用（以GB为单位）
        """
        # 梯度与参数大小相同，同样受并行策略影响
        return self.calculate_model_memory()

    def calculate_optimizer_memory(self) -> float:
        """
        计算优化器状态占用的显存
        :return: 显存占用（以GB为单位）
        """
        if self.config.optimizer_type == 'adam':
            # Adam优化器需要存储动量和方差，每个参数需要额外的8字节
            optimizer_memory = (
                (self.config.model_size_in_billions * 8 + self.config.model_size_in_billions * 4)/
                (self.config.tensor_parallel_size * self.config.pipeline_parallel_size)
            )
        else:  # SGD
            # SGD只需要存储动量，每个参数需要额外的4字节
            optimizer_memory = (
                self.config.model_size_in_billions * 4 / 
                (self.config.tensor_parallel_size * self.config.pipeline_parallel_size)
            )
        return optimizer_memory

    def calculate_activation_memory(self) -> float:
        """
        计算所有激活值占用的显存 https://zhuanlan.zhihu.com/p/648924115
        :return: 显存占用（以GB为单位）
        """
        # 基础激活值计算
        base_activation = (
            self.config.sequence_length * 
            self.config.batch_size * 
            self.config.hidden_size * 
            (34 + 5 * self.config.num_attention_heads * self.config.sequence_length / self.config.hidden_size)
        )
        
        # 考虑流水线并行的影响
        # 在流水线并行中，每个设备只需要存储部分层的激活值
        layers_per_device = self.config.num_layers / self.config.pipeline_parallel_size
        
        # 考虑张量并行的影响
        # 在张量并行中，每个设备只需要存储部分维度的激活值
        activation_per_device = base_activation / self.config.tensor_parallel_size
        
        # 计算总激活值
        total_activation = activation_per_device * layers_per_device / (1024**3)
        
        return total_activation

    def calculate_total_memory(self) -> Dict[str, float]:
        """
        计算总显存占用
        :return: 包含各部分显存占用的字典（以GB为单位）
        """
        model_memory = self.calculate_model_memory()
        gradient_memory = self.calculate_gradient_memory()
        optimizer_memory = self.calculate_optimizer_memory()
        activation_memory = self.calculate_activation_memory()
        
        total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
        
        return {
            'model_memory': model_memory,
            'gradient_memory': gradient_memory,
            'optimizer_memory': optimizer_memory,
            'activation_memory': activation_memory,
            'total_memory': total_memory
        }

    def print_memory_usage(self):
        """打印显存使用情况"""
        memory_usage = self.calculate_total_memory()
        print(f"\n{self.config.precision.name}精度下的显存占用 (TP={self.config.tensor_parallel_size}, PP={self.config.pipeline_parallel_size}):")
        print(f"模型大小: {self.config.model_size_in_billions}B")
        print(f"参数显存占用: {memory_usage['model_memory']:.2f}GB")
        print(f"梯度显存占用: {memory_usage['gradient_memory']:.2f}GB")
        print(f"优化器显存占用: {memory_usage['optimizer_memory']:.2f}GB")
        print(f"激活值显存占用: {memory_usage['activation_memory']:.2f}GB")
        print(f"总显存占用: {memory_usage['total_memory']:.2f}GB")

def main():

    model_config = {
        'model_size_in_billions': 7,
        'sequence_length': 4096,
        'hidden_size': 3584,
        'batch_size': 1,
        'num_layers': 28,
        'num_attention_heads': 28,
        'optimizer_type': 'adam',
        'precision': Precision.FP16,
        'tensor_parallel_size': 4,
        'pipeline_parallel_size': 2
    }

    # model_config = {
    #     'model_size_in_billions': 13,
    #     'sequence_length': 1024,
    #     'hidden_size': 5120,
    #     'batch_size': 1,
    #     'num_layers': 40,
    #     'num_attention_heads': 40,
    #     'optimizer_type': 'adam',
    #     'precision': Precision.FP16,
    #     'tensor_parallel_size': 1,
    #     'pipeline_parallel_size': 1
    # }

    calculator_fp16 = ModelMemoryCalculator(**model_config)
    calculator_fp16.print_memory_usage()

if __name__ == "__main__":
    main() 