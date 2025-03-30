import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 1. 基本操作对比：以ReLU为例
print("1. 基本操作对比 - ReLU:")

# 创建测试数据
x = torch.randn(100, 100)

# nn.Module方式
class ReLUModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x)

# functional方式
def relu_functional(x):
    return F.relu(x)

# 测试两种方式
relu_module = ReLUModule()
output_module = relu_module(x)
output_functional = relu_functional(x)

print("nn.Module方式输出形状:", output_module.shape)
print("functional方式输出形状:", output_functional.shape)
print("两种方式结果是否相同:", torch.allclose(output_module, output_functional))

# 2. 带参数层对比：以卷积为例
print("\n2. 带参数层对比 - 卷积:")

# nn.Module方式
class ConvModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

# functional方式（需要手动管理参数）
class ConvFunctional(nn.Module):
    def __init__(self):
        super().__init__()
        # 手动创建并注册参数
        self.weight = nn.Parameter(torch.randn(64, 3, 3, 3))
        self.bias = nn.Parameter(torch.zeros(64))
    
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, padding=1)

# 创建测试数据
x = torch.randn(1, 3, 32, 32)

conv_module = ConvModule()
conv_functional = ConvFunctional()

output_module = conv_module(x)
output_functional = conv_functional(x)

print("nn.Module方式参数数量:", sum(p.numel() for p in conv_module.parameters()))
print("functional方式参数数量:", sum(p.numel() for p in conv_functional.parameters()))

# 3. 状态保持对比：以Dropout为例
print("\n3. 状态保持对比 - Dropout:")

class DropoutModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        return self.dropout(x)

class DropoutFunctional(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = 0.5
    
    def forward(self, x):
        return F.dropout(x, p=self.p, training=self.training)

# 测试训练和评估模式的区别
x = torch.ones(1000, 1000)
dropout_module = DropoutModule()
dropout_functional = DropoutFunctional()

# 训练模式
dropout_module.train()
dropout_functional.train()
out_train_module = dropout_module(x)
out_train_functional = dropout_functional(x)

# 评估模式
dropout_module.eval()
dropout_functional.eval()
out_eval_module = dropout_module(x)
out_eval_functional = dropout_functional(x)

print("训练模式 - Module输出均值:", out_train_module.mean().item())
print("训练模式 - Functional输出均值:", out_train_functional.mean().item())
print("评估模式 - Module输出均值:", out_eval_module.mean().item())
print("评估模式 - Functional输出均值:", out_eval_functional.mean().item())

# 4. 性能对比
print("\n4. 性能对比:")

def measure_time(func, input_tensor, num_runs=1000):
    start_time = time.time()
    for _ in range(num_runs):
        _ = func(input_tensor)
    end_time = time.time()
    return (end_time - start_time) / num_runs * 1000  # 转换为毫秒

# ReLU性能测试
x = torch.randn(1000, 1000)
relu_module = ReLUModule()
nn_time = measure_time(relu_module, x)
func_time = measure_time(relu_functional, x)

print(f"nn.ReLU 平均执行时间: {nn_time:.3f}ms")
print(f"F.relu 平均执行时间: {func_time:.3f}ms")

# 绘制性能对比图
plt.figure(figsize=(10, 6))
plt.bar(['nn.ReLU', 'F.relu'], [nn_time, func_time])
plt.title('torch.nn vs torch.nn.functional 性能对比')
plt.ylabel('平均执行时间 (ms)')
plt.grid(True, alpha=0.3)
plt.show()

if __name__ == "__main__":
    print("\n总结：torch.nn vs torch.nn.functional 主要区别")
    print("1. 形式区别：")
    print("   - torch.nn: 提供类的形式，继承自nn.Module")
    print("   - torch.nn.functional: 提供函数形式")
    
    print("\n2. 参数管理：")
    print("   - torch.nn: 自动管理参数")
    print("   - torch.nn.functional: 需要手动管理参数")
    
    print("\n3. 状态保持：")
    print("   - torch.nn: 可以保持状态（如BatchNorm的运行统计信息）")
    print("   - torch.nn.functional: 无状态")
    
    print("\n4. 使用建议：")
    print("   - 构建模型时优先使用torch.nn")
    print("   - 简单操作可以使用functional")
    print("   - 需要保存状态的层必须使用nn.Module")
    print("   - 函数式编程场景使用functional") 