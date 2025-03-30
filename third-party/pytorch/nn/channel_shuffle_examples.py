import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def show_channels(tensor, title):
    """显示所有通道的特征图"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # 移除batch维度
    
    num_channels = tensor.size(0)
    num_cols = 8
    num_rows = (num_channels + num_cols - 1) // num_cols
    
    plt.figure(figsize=(20, 4 * num_rows))
    for i in range(num_channels):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(tensor[i].detach().numpy(), cmap='viridis')
        plt.title(f'Channel {i}')
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 1. 基础 ChannelShuffle 示例
print("1. 基础 ChannelShuffle 示例:")
# 创建输入张量
batch_size = 1
groups = 4
channels_per_group = 4
height = 8
width = 8
total_channels = groups * channels_per_group

# 创建有规律的输入以便观察效果
input_tensor = torch.zeros(batch_size, total_channels, height, width)
for i in range(total_channels):
    input_tensor[0, i] = i

# 创建 ChannelShuffle 层
channel_shuffle = nn.ChannelShuffle(groups)
output_tensor = channel_shuffle(input_tensor)

print("输入形状:", input_tensor.shape)
print("输出形状:", output_tensor.shape)

# 显示通道重排前后的对比
show_channels(input_tensor, "重排前的通道")
show_channels(output_tensor, "重排后的通道")

# 2. 展示重排过程
print("\n2. 通道重排过程:")
# 打印重排前后的通道顺序
input_channels = list(range(total_channels))
output_channels = output_tensor.squeeze(0).numpy()
print("输入通道顺序:", input_channels)
print("输出通道顺序:", [np.mean(output_channels[i]) for i in range(total_channels)])

# 3. ShuffleNet单元示例
class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.groups = groups
        channels_per_group = in_channels // groups
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.shuffle = nn.ChannelShuffle(groups)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.shuffle(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

# 4. 可视化不同groups的效果
def visualize_different_groups():
    input_tensor = torch.zeros(1, 16, 8, 8)
    for i in range(16):
        input_tensor[0, i] = i
    
    groups_to_test = [2, 4, 8]
    plt.figure(figsize=(20, 4))
    
    # 显示输入
    plt.subplot(1, len(groups_to_test) + 1, 1)
    plt.imshow(input_tensor[0, :4].mean(dim=0).detach().numpy(), cmap='viridis')
    plt.title('Input\n(输入)', fontsize=12)
    plt.axis('off')
    
    # 显示不同groups的结果
    for idx, g in enumerate(groups_to_test, 1):
        shuffle = nn.ChannelShuffle(g)
        output = shuffle(input_tensor)
        plt.subplot(1, len(groups_to_test) + 1, idx + 1)
        plt.imshow(output[0, :4].mean(dim=0).detach().numpy(), cmap='viridis')
        plt.title(f'Groups = {g}', fontsize=12)
        plt.axis('off')
    
    plt.suptitle('不同Groups数量的Channel Shuffle效果对比', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试ShuffleNet单元
    print("\nShuffleNet单元测试:")
    shuffle_unit = ShuffleNetUnit(16, 16)
    test_input = torch.randn(1, 16, 32, 32)
    test_output = shuffle_unit(test_input)
    print("ShuffleNet单元输出形状:", test_output.shape)
    
    # 可视化不同groups的效果
    print("\n不同groups数量的效果对比:")
    visualize_different_groups()
    
    # 性能测试
    print("\n性能测试:")
    large_input = torch.randn(1, 64, 112, 112)
    
    import time
    
    # 测试不同groups的性能
    for groups in [2, 4, 8]:
        shuffle = nn.ChannelShuffle(groups)
        start_time = time.time()
        _ = shuffle(large_input)
        end_time = time.time()
        print(f"Groups = {groups} 处理时间: {(end_time - start_time)*1000:.2f}ms")
    
    # 打印使用说明
    print("\nChannelShuffle使用注意事项:")
    print("1. 通道数必须能被groups整除")
    print("2. 常用于分组卷积后的特征融合")
    print("3. 是ShuffleNet的核心组件")
    print("4. 有助于跨组信息交流")
    
    # 绘制通道重排示意图
    def plot_channel_shuffle_diagram():
        plt.figure(figsize=(15, 5))
        
        # 原始通道排列
        plt.subplot(131)
        channels = np.arange(16).reshape(4, 4)
        plt.imshow(channels, cmap='viridis')
        plt.title('原始通道排列\n(4组，每组4通道)', fontsize=12)
        plt.axis('off')
        
        # 重塑过程
        plt.subplot(132)
        channels_reshaped = channels.T
        plt.imshow(channels_reshaped, cmap='viridis')
        plt.title('重塑后的排列\n(转置操作)', fontsize=12)
        plt.axis('off')
        
        # 最终结果
        plt.subplot(133)
        channels_shuffled = channels_reshaped.flatten().reshape(4, 4)
        plt.imshow(channels_shuffled, cmap='viridis')
        plt.title('最终通道排列\n(重新分组)', fontsize=12)
        plt.axis('off')
        
        plt.suptitle('Channel Shuffle 过程示意图', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    plot_channel_shuffle_diagram() 