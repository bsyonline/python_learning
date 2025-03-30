# Python标准库 - 随机数生成

import random
import secrets

# 1. 基本随机数生成
def basic_random():
    print("基本随机数生成示例：")
    
    # 生成随机浮点数
    print(f"随机浮点数 (0-1): {random.random()}")
    print(f"随机浮点数 (1-10): {random.uniform(1, 10)}")
    
    # 生成随机整数
    print(f"随机整数 (1-100): {random.randint(1, 100)}")
    print(f"随机范围数 (0-100, 步长5): {random.randrange(0, 100, 5)}")

# 2. 序列操作
def sequence_operations():
    print("\n序列操作示例：")
    
    # 准备测试数据
    items = ["苹果", "香蕉", "橙子", "葡萄", "西瓜"]
    
    # 随机选择
    print(f"随机选择一个: {random.choice(items)}")
    print(f"随机选择多个: {random.choices(items, k=3)}")
    
    # 随机打乱
    items_copy = items.copy()
    random.shuffle(items_copy)
    print(f"打乱后的列表: {items_copy}")
    
    # 随机样本
    print(f"不重复的随机样本: {random.sample(items, k=3)}")

# 3. 概率分布
def probability_distributions():
    print("\n概率分布示例：")
    
    # 正态分布
    normal_numbers = [random.gauss(0, 1) for _ in range(5)]
    print(f"正态分布随机数: {normal_numbers}")
    
    # 指数分布
    exp_numbers = [random.expovariate(1.0) for _ in range(5)]
    print(f"指数分布随机数: {exp_numbers}")

# 4. 密码学安全的随机数
def cryptographic_random():
    print("\n密码学安全的随机数示例：")
    
    # 生成随机字节
    random_bytes = secrets.token_bytes(16)
    print(f"随机字节: {random_bytes}")
    
    # 生成随机十六进制标记
    random_hex = secrets.token_hex(16)
    print(f"随机十六进制: {random_hex}")
    
    # 生成URL安全的随机标记
    random_url = secrets.token_urlsafe(16)
    print(f"URL安全的随机标记: {random_url}")
    
    # 生成随机密码
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(12))
    print(f"随机生成的密码: {password}")

if __name__ == "__main__":
    basic_random()
    sequence_operations()
    probability_distributions()
    cryptographic_random() 