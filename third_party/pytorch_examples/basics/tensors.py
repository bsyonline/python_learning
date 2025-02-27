# PyTorch基础 - 张量操作

import torch
import numpy as np

# 1. 创建张量
def create_tensors():
    print("创建张量示例：")
    
    # 从列表创建
    data = [[1, 2], [3, 4]]
    tensor1 = torch.tensor(data)
    print(f"从列表创建:\n{tensor1}")
    
    # 从NumPy数组创建
    np_array = np.array(data)
    tensor2 = torch.from_numpy(np_array)
    print(f"\n从NumPy数组创建:\n{tensor2}")
    
    # 创建特定形状的张量
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    rand = torch.rand(2, 3)
    
    print(f"\n零张量:\n{zeros}")
    print(f"\n全1张量:\n{ones}")
    print(f"\n随机张量:\n{rand}")

# 2. 张量操作
def tensor_operations():
    print("\n张量操作示例：")
    
    # 创建测试张量
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    
    # 基本运算
    print(f"加法:\n{x + y}")
    print(f"\n乘法:\n{x * y}")
    print(f"\n矩阵乘法:\n{torch.mm(x, y)}")
    
    # 形状操作
    print(f"\n原始形状: {x.shape}")
    reshaped = x.reshape(1, 4)
    print(f"重塑后: {reshaped.shape}")
    print(f"重塑的张量:\n{reshaped}")

# 3. 设备操作
def device_operations():
    print("\n设备操作示例：")
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建张量并移动到指定设备
    tensor = torch.rand(2, 3)
    tensor = tensor.to(device)
    print(f"\n在{device}上的张量:\n{tensor}")

# 4. 梯度计算准备
def gradient_preparation():
    print("\n梯度计算准备示例：")
    
    # 创建需要梯度的张量
    x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    print(f"原始张量:\n{x}")
    print(f"需要梯度: {x.requires_grad}")
    
    # 进行一些运算
    y = x * 2
    z = y.mean()
    print(f"\n计算结果: {z}")
    
    # 计算梯度
    z.backward()
    print(f"\n梯度:\n{x.grad}")

if __name__ == "__main__":
    create_tensors()
    tensor_operations()
    device_operations()
    gradient_preparation() 