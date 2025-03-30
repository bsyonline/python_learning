import torch
import torch.nn as nn
import numpy as np

# 创建示例数据
batch_size = 1
channels = 3
height = 32
width = 32
input_tensor = torch.randn(batch_size, channels, height, width)
print("输入张量形状:", input_tensor.shape)

# 1. 最大池化 (MaxPool2d)
max_pool = nn.MaxPool2d(
    kernel_size=2,        # 池化核大小
    stride=2,             # 步长
    padding=0,            # 填充
    dilation=1,           # 扩张率
    return_indices=False  # 是否返回最大值的索引
)
max_pooled = max_pool(input_tensor)
print("\n1. MaxPool2d输出形状:", max_pooled.shape)
print("参数说明: 2x2池化核，步长2，特征图尺寸减半")

# 2. 平均池化 (AvgPool2d)
avg_pool = nn.AvgPool2d(
    kernel_size=2,
    stride=2,
    padding=0
)
avg_pooled = avg_pool(input_tensor)
print("\n2. AvgPool2d输出形状:", avg_pooled.shape)
print("参数说明: 2x2池化核，步长2，计算区域平均值")

# 3. 带填充的最大池化
max_pool_padded = nn.MaxPool2d(
    kernel_size=3,
    stride=2,
    padding=1
)
max_pooled_padded = max_pool_padded(input_tensor)
print("\n3. 带填充的MaxPool2d输出形状:", max_pooled_padded.shape)
print("参数说明: 3x3池化核，步长2，填充1")

# 4. 自适应平均池化 (AdaptiveAvgPool2d)
adaptive_avg_pool = nn.AdaptiveAvgPool2d(
    output_size=(8, 8)    # 指定输出尺寸
)
adaptive_avg_pooled = adaptive_avg_pool(input_tensor)
print("\n4. AdaptiveAvgPool2d输出形状:", adaptive_avg_pooled.shape)
print("参数说明: 自动计算池化参数，输出固定大小8x8")

# 5. 自适应最大池化 (AdaptiveMaxPool2d)
adaptive_max_pool = nn.AdaptiveMaxPool2d(
    output_size=(8, 8)
)
adaptive_max_pooled = adaptive_max_pool(input_tensor)
print("\n5. AdaptiveMaxPool2d输出形状:", adaptive_max_pooled.shape)
print("参数说明: 自动计算池化参数，输出固定大小8x8")

# 6. 分数最大池化 (FractionalMaxPool2d)
fractional_pool = nn.FractionalMaxPool2d(
    kernel_size=2,
    output_size=(15, 15),    # 指定输出大小
    return_indices=False
)
fractional_pooled = fractional_pool(input_tensor)
print("\n6. FractionalMaxPool2d输出形状:", fractional_pooled.shape)
print("参数说明: 随机步长的最大池化，输出大小15x15")

# 7. 带返回索引的最大池化
max_pool_with_indices = nn.MaxPool2d(
    kernel_size=2,
    stride=2,
    return_indices=True   # 返回最大值的索引
)
pooled, indices = max_pool_with_indices(input_tensor)
print("\n7. 带索引的MaxPool2d输出形状:", pooled.shape)
print("索引张量形状:", indices.shape)
print("参数说明: 返回最大值位置的索引，可用于反池化")

if __name__ == "__main__":
    # 展示一个小型实例的具体效果
    print("\n小型实例演示:")
    small_input = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
    print("输入张量:\n", small_input.squeeze())
    
    # 最大池化
    small_max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    max_output = small_max_pool(small_input)
    print("\n2x2最大池化后:\n", max_output.squeeze())
    
    # 平均池化
    small_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    avg_output = small_avg_pool(small_input)
    print("\n2x2平均池化后:\n", avg_output.squeeze()) 