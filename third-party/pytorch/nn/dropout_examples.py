import torch
import torch.nn as nn
import numpy as np

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 1. 基础Dropout
print("1. 基础Dropout示例:")
dropout = nn.Dropout(p=0.5)  # 50%的概率将元素置为0
input_data = torch.ones(10)  # 创建全1张量
print("输入数据:", input_data)

# 训练模式
dropout.train()
train_output = dropout(input_data)
print("\n训练模式输出 (p=0.5):")
print(train_output)
print("非零元素数量:", torch.count_nonzero(train_output).item())

# 评估模式
dropout.eval()
eval_output = dropout(input_data)
print("\n评估模式输出:")
print(eval_output)
print("非零元素数量:", torch.count_nonzero(eval_output).item())

# 2. Dropout2d (通道dropout)
print("\n2. Dropout2d示例:")
dropout2d = nn.Dropout2d(p=0.5)
# 创建示例图像数据: [batch_size, channels, height, width]
input_2d = torch.ones(2, 4, 3, 3)
print("输入形状:", input_2d.shape)

# 训练模式
dropout2d.train()
train_output_2d = dropout2d(input_2d)
print("\n训练模式下每个通道的非零元素数量:")
for i in range(4):
    print(f"通道 {i}: {torch.count_nonzero(train_output_2d[:, i, :, :]).item()}")

# 3. Dropout3d
print("\n3. Dropout3d示例:")
dropout3d = nn.Dropout3d(p=0.5)
# 创建3D数据: [batch_size, channels, depth, height, width]
input_3d = torch.ones(2, 4, 3, 3, 3)
print("输入形状:", input_3d.shape)

# 训练模式
dropout3d.train()
train_output_3d = dropout3d(input_3d)
print("\n训练模式下每个通道的非零元素数量:")
for i in range(4):
    print(f"通道 {i}: {torch.count_nonzero(train_output_3d[:, i, :, :, :]).item()}")

# 4. AlphaDropout
print("\n4. AlphaDropout示例:")
alpha_dropout = nn.AlphaDropout(p=0.5)
input_alpha = torch.randn(10)  # 正态分布输入
print("输入数据:", input_alpha)

# 训练模式
alpha_dropout.train()
alpha_output = alpha_dropout(input_alpha)
print("\n训练模式输出:")
print(alpha_output)
print("输入均值:", input_alpha.mean().item())
print("输出均值:", alpha_output.mean().item())
print("输入标准差:", input_alpha.std().item())
print("输出标准差:", alpha_output.std().item())

# 5. 在神经网络中使用Dropout
class DropoutNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 6. 空间Dropout示例
class SpatialDropoutNet(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.dropout2d = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.conv2(x)
        return x

if __name__ == "__main__":
    print("\n5. 神经网络中的Dropout示例:")
    # 创建模型
    model = DropoutNet(input_size=20, hidden_size=10, num_classes=2)
    
    # 准备示例数据
    sample_data = torch.randn(4, 20)  # 4个样本，每个20个特征
    
    # 训练模式
    model.train()
    train_output = model(sample_data)
    print("\n训练模式输出形状:", train_output.shape)
    
    # 评估模式
    model.eval()
    eval_output = model(sample_data)
    print("评估模式输出形状:", eval_output.shape)
    
    print("\n6. 空间Dropout示例:")
    # 创建模型
    spatial_model = SpatialDropoutNet(in_channels=3)
    
    # 准备图像数据
    image_data = torch.randn(2, 3, 32, 32)  # 2张图片，3个通道，32x32大小
    
    # 训练模式
    spatial_model.train()
    spatial_output = spatial_model(image_data)
    print("空间Dropout输出形状:", spatial_output.shape)
    
    # Dropout的最佳实践
    print("\nDropout的最佳实践:")
    print("1. 使用时机")
    print("   - 训练时启用，测试时自动禁用")
    print("   - 通常在全连接层之间使用")
    print("   - 可以用在卷积层后（使用Dropout2d）")
    
    print("\n2. dropout率选择")
    print("   - 通常在0.2到0.5之间")
    print("   - 网络越大，可以使用越大的dropout率")
    print("   - 输入层通常使用较小的dropout率")
    
    print("\n3. 注意事项")
    print("   - 确保在训练时调用model.train()")
    print("   - 在验证/测试时调用model.eval()")
    print("   - Dropout会影响模型收敛速度")
    
    print("\n4. 不同类型的Dropout")
    print("   - Dropout: 用于全连接层")
    print("   - Dropout2d: 用于卷积层（通道级dropout）")
    print("   - Dropout3d: 用于3D卷积层")
    print("   - AlphaDropout: 用于SELU激活函数")
    
    # 性能对比
    print("\n性能测试:")
    large_input = torch.randn(1000, 100)
    dropout_layer = nn.Dropout(0.5)
    
    # 训练模式性能
    import time
    dropout_layer.train()
    start_time = time.time()
    _ = dropout_layer(large_input)
    train_time = time.time() - start_time
    
    # 评估模式性能
    dropout_layer.eval()
    start_time = time.time()
    _ = dropout_layer(large_input)
    eval_time = time.time() - start_time
    
    print(f"训练模式处理时间: {train_time*1000:.2f}ms")
    print(f"评估模式处理时间: {eval_time*1000:.2f}ms") 