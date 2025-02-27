# PyTorch基础 - 自动求导

import torch

# 1. 基本自动求导
def basic_autograd():
    print("基本自动求导示例：")
    
    # 创建需要梯度的张量
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    
    # 定义计算
    z = x**2 + y**3
    print(f"z = x^2 + y^3, 其中 x={x.item()}, y={y.item()}")
    print(f"z = {z.item()}")
    
    # 计算梯度
    z.backward()
    print(f"\n梯度:")
    print(f"dz/dx = {x.grad}")  # 应该是 2x = 4
    print(f"dz/dy = {y.grad}")  # 应该是 3y^2 = 27

# 2. 复杂计算的自动求导
def complex_autograd():
    print("\n复杂计算的自动求导示例：")
    
    # 创建输入张量
    x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    
    # 进行一系列操作
    y = x * 2
    z = y.mean()
    print(f"x:\n{x}")
    print(f"\ny = x * 2:\n{y}")
    print(f"\nz = mean(y): {z}")
    
    # 计算梯度
    z.backward()
    print(f"\nx的梯度:\n{x.grad}")  # 应该都是0.5

# 3. 使用with torch.no_grad()
def no_grad_example():
    print("\n使用no_grad示例：")
    
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    print(f"原始张量: {x}")
    
    # 正常情况下的计算
    y = x * 2
    print(f"需要梯度的计算结果: {y}")
    print(f"requires_grad: {y.requires_grad}")
    
    # 使用no_grad()
    with torch.no_grad():
        z = x * 2
        print(f"\n不需要梯度的计算结果: {z}")
        print(f"requires_grad: {z.requires_grad}")

# 4. 梯度累积和清零
def gradient_accumulation():
    print("\n梯度累积和清零示例：")
    
    # 创建参数
    w = torch.tensor([1.], requires_grad=True)
    
    # 多次计算梯度
    for t in range(3):
        # 前向传播
        y = w * 2
        
        # 反向传播
        y.backward()
        print(f"Step {t+1}, 梯度: {w.grad}")
    
    print("\n清零梯度后：")
    w.grad.zero_()
    y = w * 2
    y.backward()
    print(f"新的梯度: {w.grad}")

if __name__ == "__main__":
    basic_autograd()
    complex_autograd()
    no_grad_example()
    gradient_accumulation() 