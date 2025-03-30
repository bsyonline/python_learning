import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 创建输入数据用于演示
x = torch.linspace(-5, 5, 200)
print("输入数据范围:", x.min().item(), "到", x.max().item())

# 1. ReLU (Rectified Linear Unit)
relu = nn.ReLU()
relu_out = relu(x)
print("\n1. ReLU特点:")
print("- 输出范围: [0, ∞)")
print("- 特点: x > 0时保持原值，x ≤ 0时输出0")

# 2. LeakyReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
leaky_relu_out = leaky_relu(x)
print("\n2. LeakyReLU特点:")
print("- 输出范围: (-∞, ∞)")
print("- 特点: x > 0时保持原值，x ≤ 0时输出negative_slope * x")

# 3. Sigmoid
sigmoid = nn.Sigmoid()
sigmoid_out = sigmoid(x)
print("\n3. Sigmoid特点:")
print("- 输出范围: (0, 1)")
print("- 特点: 将输入压缩到0-1之间，常用于二分类")

# 4. Tanh
tanh = nn.Tanh()
tanh_out = tanh(x)
print("\n4. Tanh特点:")
print("- 输出范围: (-1, 1)")
print("- 特点: 类似Sigmoid但输出范围是-1到1")

# 5. ELU (Exponential Linear Unit)
elu = nn.ELU(alpha=1.0)
elu_out = elu(x)
print("\n5. ELU特点:")
print("- 输出范围: (-α, ∞)")
print("- 特点: x > 0时保持原值，x ≤ 0时输出α * (exp(x) - 1)")

# 6. SELU (Scaled Exponential Linear Unit)
selu = nn.SELU()
selu_out = selu(x)
print("\n6. SELU特点:")
print("- 输出范围: (-λα, λ)")
print("- 特点: 自归一化特性，适合深层网络")

# 7. Softplus
softplus = nn.Softplus()
softplus_out = softplus(x)
print("\n7. Softplus特点:")
print("- 输出范围: (0, ∞)")
print("- 特点: ReLU的平滑版本")

# 8. PReLU (Parametric ReLU)
prelu = nn.PReLU()
prelu_out = prelu(x)
print("\n8. PReLU特点:")
print("- 输出范围: (-∞, ∞)")
print("- 特点: 可学习的负斜率参数")

# 9. GELU (Gaussian Error Linear Unit)
gelu = nn.GELU()
gelu_out = gelu(x)
print("\n9. GELU特点:")
print("- 输出范围: (-∞, ∞)")
print("- 特点: 在Transformer中常用，结合了ReLU和dropout的特性")

# 可视化所有激活函数
def plot_activations():
    plt.figure(figsize=(15, 10))
    plt.plot(x.numpy(), relu_out.numpy(), label='ReLU')
    plt.plot(x.numpy(), leaky_relu_out.numpy(), label='LeakyReLU')
    plt.plot(x.numpy(), sigmoid_out.numpy(), label='Sigmoid')
    plt.plot(x.numpy(), tanh_out.numpy(), label='Tanh')
    plt.plot(x.numpy(), elu_out.numpy(), label='ELU')
    plt.plot(x.numpy(), selu_out.numpy(), label='SELU')
    plt.plot(x.numpy(), softplus_out.numpy(), label='Softplus')
    plt.plot(x.numpy(), prelu_out.detach().numpy(), label='PReLU')
    plt.plot(x.numpy(), gelu_out.numpy(), label='GELU')
    
    plt.grid(True)
    plt.legend(loc='best', fontsize=10)
    plt.title('非线性激活函数对比', fontsize=14)
    plt.xlabel('输入 x', fontsize=12)
    plt.ylabel('输出 y', fontsize=12)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 保存图像到文件（可选）
    # plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 实际使用示例
    print("\n实际使用示例:")
    # 创建一个简单的输入张量
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("输入张量:", input_tensor)
    
    # 应用不同的激活函数
    print("\nReLU输出:", relu(input_tensor))
    print("LeakyReLU输出:", leaky_relu(input_tensor))
    print("Sigmoid输出:", sigmoid(input_tensor))
    print("Tanh输出:", tanh(input_tensor))
    
    # 在神经网络中使用激活函数
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(5, 3)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(3, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x
    
    # 创建模型实例
    model = SimpleNet()
    sample_input = torch.randn(5)
    output = model(sample_input)
    print("\n神经网络中的激活函数示例:")
    print("输入:", sample_input)
    print("输出:", output)
    
    # 绘制激活函数图像
    plot_activations() 