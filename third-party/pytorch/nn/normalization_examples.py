import torch
import torch.nn as nn
import numpy as np

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 1. BatchNorm1d - 用于全连接层
print("1. BatchNorm1d 示例:")
batch_size = 4
features = 5
input_1d = torch.randn(batch_size, features)
print("输入张量形状:", input_1d.shape)
print("输入数据:\n", input_1d)

bn1d = nn.BatchNorm1d(features)
output_1d = bn1d(input_1d)
print("\n归一化后:\n", output_1d)
print("均值接近0:", output_1d.mean(dim=0))
print("方差接近1:", output_1d.var(dim=0, unbiased=False))

# 2. BatchNorm2d - 用于卷积层
print("\n2. BatchNorm2d 示例:")
batch_size = 2
channels = 3
height = 4
width = 4
input_2d = torch.randn(batch_size, channels, height, width)
print("输入张量形状:", input_2d.shape)

bn2d = nn.BatchNorm2d(channels)
output_2d = bn2d(input_2d)
print("归一化后张量形状:", output_2d.shape)
print("通道维度的均值接近0:", output_2d.mean(dim=(0, 2, 3)))
print("通道维度的方差接近1:", output_2d.var(dim=(0, 2, 3), unbiased=False))

# 3. LayerNorm - 层归一化
print("\n3. LayerNorm 示例:")
# 创建一个特征序列
seq_length = 6
hidden_size = 4
input_ln = torch.randn(batch_size, seq_length, hidden_size)
print("输入张量形状:", input_ln.shape)

ln = nn.LayerNorm(hidden_size)
output_ln = ln(input_ln)
print("归一化后张量形状:", output_ln.shape)
print("最后一个维度的均值接近0:", output_ln.mean(dim=-1))
print("最后一个维度的方差接近1:", output_ln.var(dim=-1, unbiased=False))

# 4. InstanceNorm2d - 实例归一化
print("\n4. InstanceNorm2d 示例:")
in2d = nn.InstanceNorm2d(channels)
output_in = in2d(input_2d)
print("归一化后张量形状:", output_in.shape)
print("每个实例的均值接近0:", output_in.mean(dim=(2, 3)))

# 5. GroupNorm - 组归一化
print("\n5. GroupNorm 示例:")
num_groups = 3
gn = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
output_gn = gn(input_2d)
print("归一化后张量形状:", output_gn.shape)
print("每个组的均值接近0:", output_gn.mean(dim=(2, 3)))

# 6. 在神经网络中使用归一化层的实例
class NormalizationNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        # 卷积层 + BatchNorm2d
        self.conv_bn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # 全连接层 + LayerNorm
        self.fc_ln = nn.Sequential(
            nn.Linear(16 * 4 * 4, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # 输出层
        self.output = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv_bn(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_ln(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    print("\n完整神经网络示例:")
    # 创建模型实例
    model = NormalizationNet()
    # 创建示例输入
    sample_input = torch.randn(2, 3, 4, 4)
    print("输入张量形状:", sample_input.shape)
    
    # 前向传播
    output = model(sample_input)
    print("输出张量形状:", output.shape)
    
    # 展示不同归一化方法的特点
    print("\n不同归一化方法的特点:")
    print("1. BatchNorm:")
    print("   - 在batch维度上归一化")
    print("   - 适用于固定大小的batch")
    print("   - 训练和推理行为不同")
    
    print("\n2. LayerNorm:")
    print("   - 在特征维度上归一化")
    print("   - 独立于batch size")
    print("   - 常用于Transformer")
    
    print("\n3. InstanceNorm:")
    print("   - 在单个样本上归一化")
    print("   - 适用于风格迁移")
    print("   - 不依赖于batch size")
    
    print("\n4. GroupNorm:")
    print("   - 在特征组上归一化")
    print("   - batch size独立")
    print("   - 平衡了BN和LN的优点")
    
    # 可选：显示每种归一化方法的数值示例
    test_input = torch.randn(2, 4, 3, 3)
    print("\n数值示例 (使用相同输入):")
    print("原始数据统计:")
    print(f"均值: {test_input.mean():.4f}")
    print(f"标准差: {test_input.std():.4f}")
    
    # 应用不同的归一化
    bn = nn.BatchNorm2d(4)
    ln = nn.LayerNorm([4, 3, 3])
    in_norm = nn.InstanceNorm2d(4)
    gn = nn.GroupNorm(2, 4)
    
    print("\n归一化后的统计:")
    print("BatchNorm2d -    均值: {:.4f}, 标准差: {:.4f}".format(
        bn(test_input).mean(), bn(test_input).std()))
    print("LayerNorm -     均值: {:.4f}, 标准差: {:.4f}".format(
        ln(test_input).mean(), ln(test_input).std()))
    print("InstanceNorm2d - 均值: {:.4f}, 标准差: {:.4f}".format(
        in_norm(test_input).mean(), in_norm(test_input).std()))
    print("GroupNorm -     均值: {:.4f}, 标准差: {:.4f}".format(
        gn(test_input).mean(), gn(test_input).std())) 