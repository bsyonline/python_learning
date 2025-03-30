import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def show_feature_maps(tensor, title):
    """辅助函数：显示特征图"""
    if tensor.dim() == 4:  # [batch_size, channels, height, width]
        tensor = tensor.squeeze(0)  # 移除batch维度
    
    num_features = min(tensor.size(0), 16)  # 最多显示16个通道
    num_rows = int(np.ceil(np.sqrt(num_features)))
    
    plt.figure(figsize=(15, 15))
    for i in range(num_features):
        plt.subplot(num_rows, num_rows, i + 1)
        plt.imshow(tensor[i].detach().numpy(), cmap='viridis')
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def show_all_methods(input_tensor, outputs_dict):
    """显示所有上采样方法的结果对比"""
    num_methods = len(outputs_dict) + 1  # +1 for input
    fig = plt.figure(figsize=(20, 4))
    
    # 显示输入
    plt.subplot(1, num_methods, 1)
    plt.imshow(input_tensor[0, 0].detach().numpy(), cmap='viridis')
    plt.title('Input\n(输入)', fontsize=12)
    plt.axis('off')
    
    # 显示各种方法的结果
    for idx, (name, tensor) in enumerate(outputs_dict.items(), 1):
        plt.subplot(1, num_methods, idx + 1)
        plt.imshow(tensor[0, 0].detach().numpy(), cmap='viridis')
        plt.title(f'{name}', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 创建输入数据
batch_size = 1
input_channels = 3
height = 32
width = 32

# 1. 准备所有上采样方法
# PixelShuffle输入需要特殊处理
input_shuffle = torch.randn(batch_size, input_channels * 4, height, width)  # 4 = upscale_factor^2
input_tensor = torch.randn(batch_size, input_channels, height, width)

# 创建所有上采样层
pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
upsample_nearest = nn.Upsample(scale_factor=2, mode='nearest')
upsample_bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
upsample_bicubic = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
upsampling_nearest2d = nn.UpsamplingNearest2d(scale_factor=2)
upsampling_bilinear2d = nn.UpsamplingBilinear2d(scale_factor=2)

# 2. 计算所有方法的输出
outputs = {
    'PixelShuffle': pixel_shuffle(input_shuffle),
    'Nearest': upsample_nearest(input_tensor),
    'Bilinear': upsample_bilinear(input_tensor),
    'Bicubic': upsample_bicubic(input_tensor),
    'UpsamplingNearest2d': upsampling_nearest2d(input_tensor),
    'UpsamplingBilinear2d': upsampling_bilinear2d(input_tensor)
}

# 3. 显示结果
print("不同上采样方法的输出形状:")
for name, output in outputs.items():
    print(f"{name}: {output.shape}")

# 4. 可视化对比
show_all_methods(input_tensor, outputs)

# 5. 性能测试
print("\n性能测试:")
large_input = torch.randn(1, 3, 256, 256)
large_input_shuffle = torch.randn(1, 12, 256, 256)

methods = {
    'PixelShuffle': (nn.PixelShuffle(2), large_input_shuffle),
    'Upsample Nearest': (nn.Upsample(scale_factor=2, mode='nearest'), large_input),
    'Upsample Bilinear': (nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), large_input),
    'Upsample Bicubic': (nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True), large_input),
    'UpsamplingNearest2d': (nn.UpsamplingNearest2d(scale_factor=2), large_input),
    'UpsamplingBilinear2d': (nn.UpsamplingBilinear2d(scale_factor=2), large_input)
}

# 执行性能测试并收集结果
performance_results = {}
for name, (layer, test_input) in methods.items():
    start_time = time.time()
    _ = layer(test_input)
    end_time = time.time()
    performance_results[name] = (end_time - start_time) * 1000

# 绘制性能对比柱状图
plt.figure(figsize=(12, 6))
plt.bar(performance_results.keys(), performance_results.values())
plt.xticks(rotation=45, ha='right')
plt.ylabel('处理时间 (ms)')
plt.title('不同上采样方法的性能对比')
plt.tight_layout()
plt.show()

# 打印特点对比
print("\n不同上采样方法的特点对比:")
print("方法\t\t优点\t\t\t缺点")
print("-" * 60)
print("PixelShuffle\t信息完整，可学习\t需要特定通道数")
print("Nearest\t\t计算快速，简单\t可能产生锯齿")
print("Bilinear\t平滑过渡\t\t可能丢失细节")
print("Bicubic\t\t保持细节较好\t计算量较大")
print("UpsamplingNearest2d\t专用于2D，高效\t功能单一")
print("UpsamplingBilinear2d\t专用于2D，平滑\t功能单一")


if __name__ == "__main__":
    # 创建一个小图像
    sample_image = torch.randn(1, 3, 32, 32)
    
    # 比较不同上采样方法
    print("\n不同上采样方法的特点:")
    print("1. PixelShuffle:")
    print("   - 学习像素重排")
    print("   - 保持信息完整")
    print("   - 适用于超分辨率")
    
    print("\n2. PixelUnshuffle:")
    print("   - PixelShuffle的逆操作")
    print("   - 用于降采样")
    print("   - 保持信息完整")
    
    print("\n3. Upsample:")
    print("   - 支持多种插值模式")
    print("   - 可调整大小")
    print("   - 通用上采样操作")
    
    print("\n4. UpsamplingNearest2d:")
    print("   - 最近邻插值")
    print("   - 计算快速")
    print("   - 可能产生锯齿")
    
    print("\n5. UpsamplingBilinear2d:")
    print("   - 双线性插值")
    print("   - 平滑过渡")
    print("   - 可能丢失细节")
    
    # 性能测试
    print("\n性能测试:")
    large_input = torch.randn(1, 3, 256, 256)
    
    # 测试不同上采样方法的性能
    methods = {
        'PixelShuffle': (nn.PixelShuffle(2), torch.randn(1, 12, 256, 256)),
        'Upsample Nearest': (nn.Upsample(scale_factor=2, mode='nearest'), large_input),
        'Upsample Bilinear': (nn.Upsample(scale_factor=2, mode='bilinear'), large_input),
        'UpsamplingNearest2d': (nn.UpsamplingNearest2d(scale_factor=2), large_input),
        'UpsamplingBilinear2d': (nn.UpsamplingBilinear2d(scale_factor=2), large_input)
    }
    
    for name, (layer, test_input) in methods.items():
        start_time = time.time()
        _ = layer(test_input)
        end_time = time.time()
        print(f"{name}处理时间: {(end_time - start_time)*1000:.2f}ms") 