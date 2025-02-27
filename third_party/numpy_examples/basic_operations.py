# NumPy基础操作示例

import numpy as np

# 1. 创建数组
def create_arrays():
    print("创建数组示例：")
    
    # 从列表创建数组
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"从列表创建: {arr1}")
    
    # 创建特定形状的数组
    arr2 = np.zeros((2, 3))
    print(f"\n零数组:\n{arr2}")
    
    arr3 = np.ones((2, 2))
    print(f"\n一数组:\n{arr3}")
    
    # 创建范围数组
    arr4 = np.arange(0, 10, 2)
    print(f"\n范围数组: {arr4}")
    
    # 创建线性空间
    arr5 = np.linspace(0, 1, 5)
    print(f"\n线性空间: {arr5}")

# 2. 数组操作
def array_operations():
    print("\n数组操作示例：")
    
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"原始数组:\n{arr}")
    
    # 形状操作
    print(f"\n形状: {arr.shape}")
    print(f"维度: {arr.ndim}")
    print(f"元素总数: {arr.size}")
    
    # 重塑数组
    reshaped = arr.reshape(3, 2)
    print(f"\n重塑后:\n{reshaped}")
    
    # 转置
    transposed = arr.T
    print(f"\n转置后:\n{transposed}")

# 3. 数学运算
def mathematical_operations():
    print("\n数学运算示例：")
    
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    
    # 基本运算
    print(f"加法: {arr1 + arr2}")
    print(f"乘法: {arr1 * arr2}")
    print(f"平方: {arr1 ** 2}")
    
    # 统计运算
    print(f"\n平均值: {arr1.mean()}")
    print(f"标准差: {arr1.std()}")
    print(f"最大值: {arr1.max()}")
    print(f"最小值: {arr1.min()}")

# 4. 索引和切片
def indexing_slicing():
    print("\n索引和切片示例：")
    
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"原始数组:\n{arr}")
    
    # 基本索引
    print(f"\n第一行: {arr[0]}")
    print(f"特定元素: {arr[1, 2]}")
    
    # 切片
    print(f"\n前两行:\n{arr[:2]}")
    print(f"\n所有行的前两列:\n{arr[:, :2]}")
    
    # 布尔索引
    mask = arr > 6
    print(f"\n大于6的元素:\n{arr[mask]}")

if __name__ == "__main__":
    create_arrays()
    array_operations()
    mathematical_operations()
    indexing_slicing() 