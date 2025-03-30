import torch
import torch.nn as nn
import numpy as np

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 1. 基础线性层
print("1. 基础线性层示例:")
# 创建一个简单的线性层
linear = nn.Linear(
    in_features=10,    # 输入特征数
    out_features=5,    # 输出特征数
    bias=True          # 是否包含偏置项
)

# 创建输入数据
input_data = torch.randn(3, 10)  # 批次大小为3，每个样本10个特征
output = linear(input_data)

print("输入形状:", input_data.shape)
print("输出形状:", output.shape)
print("权重形状:", linear.weight.shape)
print("偏置形状:", linear.bias.shape)

# 2. 不带偏置的线性层
print("\n2. 不带偏置的线性层示例:")
linear_no_bias = nn.Linear(10, 5, bias=False)
output_no_bias = linear_no_bias(input_data)
print("输出形状:", output_no_bias.shape)
print("权重形状:", linear_no_bias.weight.shape)

# 3. 多层线性网络
class MultiLayerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 8)
        self.layer2 = nn.Linear(8, 6)
        self.layer3 = nn.Linear(6, 4)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

print("\n3. 多层线性网络示例:")
multi_layer = MultiLayerNetwork()
output_multi = multi_layer(input_data)
print("多层网络输出形状:", output_multi.shape)

# 4. 带有自定义初始化的线性层
print("\n4. 带有自定义初始化的线性层示例:")
linear_custom = nn.Linear(10, 5)
# 使用xavier初始化
nn.init.xavier_uniform_(linear_custom.weight)
# 将偏置初始化为0
nn.init.zeros_(linear_custom.bias)

output_custom = linear_custom(input_data)
print("输出形状:", output_custom.shape)
print("权重范围:", linear_custom.weight.min().item(), "到", linear_custom.weight.max().item())

# 5. 序列形式的线性层
print("\n5. 序列形式的线性层示例:")
sequential_model = nn.Sequential(
    nn.Linear(10, 8),
    nn.ReLU(),
    nn.Linear(8, 6),
    nn.ReLU(),
    nn.Linear(6, 4)
)

output_sequential = sequential_model(input_data)
print("序列模型输出形状:", output_sequential.shape)

# 6. 不同维度输入的线性层处理
print("\n6. 不同维度输入的线性层处理:")
# 2D输入 (batch_size, features)
input_2d = torch.randn(3, 10)
output_2d = linear(input_2d)
print("2D输入输出形状:", output_2d.shape)

# 3D输入 (batch_size, sequence_length, features)
input_3d = torch.randn(3, 4, 10)
output_3d = linear(input_3d)
print("3D输入输出形状:", output_3d.shape)

# 7. 实际应用示例
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 8. Identity层示例
print("\n8. Identity层示例:")
identity = nn.Identity()
identity_input = torch.randn(3, 5)
identity_output = identity(identity_input)
print("Identity层输入形状:", identity_input.shape)
print("Identity层输出形状:", identity_output.shape)
print("输入输出是否相同:", torch.equal(identity_input, identity_output))

# 9. Bilinear层示例
print("\n9. Bilinear层示例:")
# 创建双线性层
bilinear = nn.Bilinear(
    in1_features=10,    # 第一个输入的特征数
    in2_features=8,     # 第二个输入的特征数
    out_features=4      # 输出特征数
)

# 准备两个输入
input1 = torch.randn(3, 10)  # 第一个输入：3个样本，每个10个特征
input2 = torch.randn(3, 8)   # 第二个输入：3个样本，每个8个特征
bilinear_output = bilinear(input1, input2)

print("第一个输入形状:", input1.shape)
print("第二个输入形状:", input2.shape)
print("Bilinear层输出形状:", bilinear_output.shape)
print("Bilinear层权重形状:", bilinear.weight.shape)
print("Bilinear层偏置形状:", bilinear.bias.shape)

# 10. LazyLinear层示例
print("\n10. LazyLinear层示例:")
class LazyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # LazyLinear会在第一次前向传播时自动确定输入特征数
        self.lazy_linear = nn.LazyLinear(out_features=5)
    
    def forward(self, x):
        return self.lazy_linear(x)

lazy_model = LazyNetwork()
# 使用不同输入维度的数据
lazy_input1 = torch.randn(3, 10)
lazy_output1 = lazy_model(lazy_input1)
print("LazyLinear输入形状:", lazy_input1.shape)
print("LazyLinear输出形状:", lazy_output1.shape)
print("LazyLinear层权重形状:", lazy_model.lazy_linear.weight.shape)

# 尝试使用不同输入维度（这将引发错误，因为权重已经初始化）
print("\n尝试不同输入维度:")
try:
    lazy_input2 = torch.randn(3, 15)  # 不同的输入维度
    lazy_output2 = lazy_model(lazy_input2)
except RuntimeError as e:
    print("预期的错误：LazyLinear在第一次使用后就固定了输入维度")

if __name__ == "__main__":
    print("\n7. 分类器示例:")
    # 创建模型
    classifier = SimpleClassifier(input_size=10, hidden_size=8, num_classes=3)
    
    # 准备示例数据
    sample_data = torch.randn(5, 10)  # 5个样本，每个10个特征
    
    # 前向传播
    predictions = classifier(sample_data)
    print("分类器输出形状:", predictions.shape)
    print("分类器输出示例:\n", predictions[0])  # 显示第一个样本的预测概率
    
    # 线性层的常见应用
    print("\n线性层的常见应用场景:")
    print("1. 特征转换")
    print("   - 改变特征维度")
    print("   - 特征映射和投影")
    
    print("\n2. 分类任务")
    print("   - 最后一层用于输出类别概率")
    print("   - 通常接softmax激活函数")
    
    print("\n3. 回归任务")
    print("   - 预测连续值")
    print("   - 最后一层输出目标维度")
    
    print("\n4. 嵌入层")
    print("   - 将离散特征转换为连续向量")
    print("   - 降维或升维操作")
    
    # 线性层的注意事项
    print("\n使用线性层的注意事项:")
    print("1. 初始化方法")
    print("   - 合适的权重初始化对训练很重要")
    print("   - 常用xavier或kaiming初始化")
    
    print("\n2. 维度处理")
    print("   - 注意输入输出维度的匹配")
    print("   - 考虑批处理维度")
    
    print("\n3. 激活函数")
    print("   - 通常需要配合非线性激活函数")
    print("   - 避免多层线性层直接叠加")
    
    print("\n4. 正则化")
    print("   - 考虑使用dropout防止过拟合")
    print("   - 可以添加L1/L2正则化")
    
    # 性能测试
    print("\n性能测试:")
    large_input = torch.randn(1000, 10)
    large_linear = nn.Linear(10, 5)
    
    # 计算前向传播时间
    import time
    start_time = time.time()
    _ = large_linear(large_input)
    end_time = time.time()
    
    print(f"处理1000个样本的时间: {(end_time - start_time)*1000:.2f}ms")
    
    print("\n额外线性层的特点:")
    print("1. Identity层")
    print("   - 直接传递输入到输出")
    print("   - 在跳跃连接中很有用")
    print("   - 不改变输入的形状和值")
    
    print("\n2. Bilinear层")
    print("   - 处理两个输入之间的交互")
    print("   - 用于特征融合和多模态学习")
    print("   - 输出是两个输入的双线性变换")
    
    print("\n3. LazyLinear层")
    print("   - 输入维度在首次使用时自动确定")
    print("   - 适用于输入维度未知的情况")
    print("   - 初始化后维度固定")
    
    print("\n使用场景示例:")
    print("Identity层:")
    print("- ResNet中的跳跃连接")
    print("- 调试和可视化网络结构")
    
    print("\nBilinear层:")
    print("- 多模态特征融合")
    print("- 注意力机制的评分函数")
    print("- 特征交互建模")
    
    print("\nLazyLinear层:")
    print("- 动态网络结构")
    print("- 原型开发阶段")
    print("- 输入维度未知的场景") 