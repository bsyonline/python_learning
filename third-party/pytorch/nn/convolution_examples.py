import torch
import torch.nn as nn
import numpy as np

# 创建示例数据
batch_size = 1
in_channels = 3
height = 32
width = 32
input_tensor = torch.randn(batch_size, in_channels, height, width)
print("输入张量形状:", input_tensor.shape)

# 1. 基础的2D卷积层 (Conv2d)
basic_conv = nn.Conv2d(
    in_channels=3,        # 输入通道数
    out_channels=16,      # 输出通道数
    kernel_size=3,        # 卷积核大小
    stride=1,             # 步长
    padding=1             # 填充
)
basic_output = basic_conv(input_tensor)
print("\n1. 基础Conv2d输出形状:", basic_output.shape)
print("参数说明: 3通道输入，16通道输出，3x3卷积核，步长1，填充1")

# 2. 使用不同步长的卷积层
strided_conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=2,             # 步长为2，降采样
    padding=1
)
strided_output = strided_conv(input_tensor)
print("\n2. 步长为2的Conv2d输出形状:", strided_output.shape)
print("参数说明: 步长为2导致特征图尺寸减半")

# 3. 使用扩张卷积 (Dilated Convolution)
dilated_conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    dilation=2,           # 扩张率
    padding=2             # 需要更大的填充以保持输出尺寸
)
dilated_output = dilated_conv(input_tensor)
print("\n3. 扩张卷积输出形状:", dilated_output.shape)
print("参数说明: 扩张率2增加了感受野，不改变参数数量")

# 4. 分组卷积 (Grouped Convolution)
grouped_conv = nn.Conv2d(
    in_channels=3,
    out_channels=15,      # 修改为能被3整除的数
    kernel_size=3,
    groups=3,             # 分组数
    padding=1
)
grouped_output = grouped_conv(input_tensor)
print("\n4. 分组卷积输出形状:", grouped_output.shape)
print("参数说明: 3个分组，每组输入1通道，输出5通道")

# 5. 深度可分离卷积 (Depthwise Separable Convolution)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels  # 每个通道独立卷积
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

depthwise_sep_conv = DepthwiseSeparableConv(3, 16)
depthwise_output = depthwise_sep_conv(input_tensor)
print("\n5. 深度可分离卷积输出形状:", depthwise_output.shape)
print("参数说明: 先3x3深度卷积，再1x1逐点卷积")

# 6. 转置卷积 (Transposed Convolution)
transpose_conv = nn.ConvTranspose2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1      # 用于调整输出大小
)
transpose_output = transpose_conv(input_tensor)
print("\n6. 转置卷积输出形状:", transpose_output.shape)
print("参数说明: 步长2使特征图尺寸加倍")

# 7. 1x1 卷积 (Point-wise Convolution)
pointwise_conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=1        # 1x1卷积核
)
pointwise_output = pointwise_conv(input_tensor)
print("\n7. 1x1卷积输出形状:", pointwise_output.shape)
print("参数说明: 1x1卷积用于通道数调整和特征融合")

if __name__ == "__main__":
    # 展示一个小型实例的具体效果
    print("\n小型实例演示:")
    small_input = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
    print("输入张量:\n", small_input.squeeze())
    
    # 使用3x3卷积核
    small_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    # 设置固定的卷积核权重以便观察
    with torch.no_grad():
        small_conv.weight.data = torch.tensor([[[[1/9, 1/9, 1/9],
                                               [1/9, 1/9, 1/9],
                                               [1/9, 1/9, 1/9]]]])
        small_conv.bias.data.zero_()
    
    small_output = small_conv(small_input)
    print("\n3x3均值卷积后:\n", small_output.squeeze().detach().numpy()) 