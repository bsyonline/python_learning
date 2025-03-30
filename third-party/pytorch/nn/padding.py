# PyTorch基础 - 填充

import torch
import torch.nn.functional as F

def padding_examples():
    print("PyTorch Padding 示例：")
    
    # 创建一个示例输入张量 (batch_size=1, channels=1, height=3, width=3)
    input_tensor = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]).float().unsqueeze(0).unsqueeze(0)
    
    print(f"原始张量形状: {input_tensor.shape}")
    print(f"原始张量内容:\n{input_tensor.squeeze()}\n")

    # 1. 零填充 (Zero Padding)
    pad_zero = F.pad(input_tensor, pad=(1, 1, 1, 1), mode='constant', value=0)
    print("零填充 (上下左右各填充1个单位):")
    print(f"形状: {pad_zero.shape}")
    print(f"内容:\n{pad_zero.squeeze()}\n")

    # 2. 复制填充 (Replicate Padding)
    pad_replicate = F.pad(input_tensor, pad=(1, 1, 1, 1), mode='replicate')
    print("复制填充 (使用边缘值填充):")
    print(f"形状: {pad_replicate.shape}")
    print(f"内容:\n{pad_replicate.squeeze()}\n")

    # 3. 反射填充 (Reflect Padding)
    pad_reflect = F.pad(input_tensor, pad=(1, 1, 1, 1), mode='reflect')
    print("反射填充:")
    print(f"形状: {pad_reflect.shape}")
    print(f"内容:\n{pad_reflect.squeeze()}\n")

    # 4. 循环填充 (Circular Padding)
    pad_circular = F.pad(input_tensor, pad=(1, 1, 1, 1), mode='circular')
    print("循环填充:")
    print(f"形状: {pad_circular.shape}")
    print(f"内容:\n{pad_circular.squeeze()}\n")

    # 5. 不对称填充示例
    pad_asymmetric = F.pad(input_tensor, pad=(1, 2, 2, 1), mode='constant', value=0)
    print("不对称填充 (左1右2上2下1):")
    print(f"形状: {pad_asymmetric.shape}")
    print(f"内容:\n{pad_asymmetric.squeeze()}")

if __name__ == "__main__":
    padding_examples()