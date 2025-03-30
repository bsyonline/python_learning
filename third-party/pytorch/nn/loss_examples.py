import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. MSE Loss (均方误差损失)
print("1. MSE Loss示例:")
mse_loss = nn.MSELoss()
input_mse = torch.randn(3, 5, requires_grad=True)
target_mse = torch.randn(3, 5)
output_mse = mse_loss(input_mse, target_mse)
print("MSE Loss:", output_mse.item())

# 2. Cross Entropy Loss (交叉熵损失)
print("\n2. Cross Entropy Loss示例:")
# 创建输入（logits）和目标
input_ce = torch.randn(3, 5, requires_grad=True)  # 3个样本，5个类别
target_ce = torch.empty(3, dtype=torch.long).random_(5)  # 随机目标类别
ce_loss = nn.CrossEntropyLoss()
output_ce = ce_loss(input_ce, target_ce)
print("Cross Entropy Loss:", output_ce.item())

# 3. Binary Cross Entropy Loss (二元交叉熵损失)
print("\n3. Binary Cross Entropy Loss示例:")
input_bce = torch.randn(3, requires_grad=True)
target_bce = torch.empty(3).random_(2)
bce_loss = nn.BCEWithLogitsLoss()  # 包含sigmoid层
output_bce = bce_loss(input_bce, target_bce)
print("Binary Cross Entropy Loss:", output_bce.item())

# 4. L1 Loss (绝对值误差损失)
print("\n4. L1 Loss示例:")
l1_loss = nn.L1Loss()
input_l1 = torch.randn(3, 5, requires_grad=True)
target_l1 = torch.randn(3, 5)
output_l1 = l1_loss(input_l1, target_l1)
print("L1 Loss:", output_l1.item())

# 5. Huber Loss (平滑的L1损失)
print("\n5. Huber Loss示例:")
huber_loss = nn.HuberLoss(delta=1.0)
input_huber = torch.randn(3, 5, requires_grad=True)
target_huber = torch.randn(3, 5)
output_huber = huber_loss(input_huber, target_huber)
print("Huber Loss:", output_huber.item())

# 6. KL Divergence Loss (KL散度损失)
print("\n6. KL Divergence Loss示例:")
kl_loss = nn.KLDivLoss(reduction='batchmean')
input_kl = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
target_kl = F.softmax(torch.randn(3, 5), dim=1)
output_kl = kl_loss(input_kl, target_kl)
print("KL Divergence Loss:", output_kl.item())

# 7. Margin Ranking Loss (边界排序损失)
print("\n7. Margin Ranking Loss示例:")
margin_loss = nn.MarginRankingLoss(margin=0.5)
input1 = torch.randn(3, requires_grad=True)
input2 = torch.randn(3, requires_grad=True)
target = torch.ones(3)  # 1表示input1应该比input2大
output_margin = margin_loss(input1, input2, target)
print("Margin Ranking Loss:", output_margin.item())

# 8. 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, input, target):
        return self.weight * torch.mean((input - target) ** 2 + torch.abs(input - target))

print("\n8. 自定义损失函数示例:")
custom_loss = CustomLoss(weight=2.0)
input_custom = torch.randn(3, 5, requires_grad=True)
target_custom = torch.randn(3, 5)
output_custom = custom_loss(input_custom, target_custom)
print("Custom Loss:", output_custom.item())

# 9. 可视化不同损失函数
def plot_loss_curves():
    # 创建预测值范围
    x = torch.linspace(-3, 3, 100)
    target = torch.zeros_like(x)
    
    # 计算不同损失
    mse = torch.tensor([nn.MSELoss()(torch.tensor([xi]), torch.tensor([0.0])).item() for xi in x])
    l1 = torch.tensor([nn.L1Loss()(torch.tensor([xi]), torch.tensor([0.0])).item() for xi in x])
    huber = torch.tensor([nn.HuberLoss(delta=1.0)(torch.tensor([xi]), torch.tensor([0.0])).item() for xi in x])
    bce = torch.tensor([nn.BCEWithLogitsLoss()(torch.tensor([xi]), torch.tensor([0.0])).item() for xi in x])
    custom = torch.tensor([CustomLoss(weight=1.0)(torch.tensor([xi]), torch.tensor([0.0])).item() for xi in x])
    
    # 计算KL散度损失
    kl_input = torch.tensor([F.log_softmax(torch.tensor([xi, 0.0]), dim=0)[0] for xi in x])
    kl_target = F.softmax(torch.tensor([0.0, 0.0]), dim=0)[0].expand(100)
    kl = torch.tensor([nn.KLDivLoss(reduction='sum')(torch.tensor([i]), torch.tensor([t])).item() 
                      for i, t in zip(kl_input, kl_target)])
    
    # 计算Margin Ranking损失
    margin = torch.tensor([nn.MarginRankingLoss(margin=0.5)(torch.tensor([xi]), 
                                                           torch.tensor([0.0]), 
                                                           torch.tensor([1.0])).item() for xi in x])
    
    # 绘图
    plt.figure(figsize=(20, 15))
    
    # MSE Loss
    plt.subplot(331)
    plt.plot(x.numpy(), mse.numpy(), label='MSE Loss', color='blue')
    plt.title('MSE Loss\n(均方误差损失)', fontsize=12)
    plt.xlabel('预测值', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.grid(True)
    plt.legend()
    
    # L1 Loss
    plt.subplot(332)
    plt.plot(x.numpy(), l1.numpy(), label='L1 Loss', color='red')
    plt.title('L1 Loss\n(绝对值误差损失)', fontsize=12)
    plt.xlabel('预测值', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.grid(True)
    plt.legend()
    
    # Huber Loss
    plt.subplot(333)
    plt.plot(x.numpy(), huber.numpy(), label='Huber Loss', color='green')
    plt.title('Huber Loss\n(平滑L1损失)', fontsize=12)
    plt.xlabel('预测值', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.grid(True)
    plt.legend()
    
    # BCE Loss
    plt.subplot(334)
    plt.plot(x.numpy(), bce.numpy(), label='BCE Loss', color='purple')
    plt.title('BCE Loss\n(二元交叉熵损失)', fontsize=12)
    plt.xlabel('预测值', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.grid(True)
    plt.legend()
    
    # KL Divergence Loss
    plt.subplot(335)
    plt.plot(x.numpy(), kl.numpy(), label='KL Div Loss', color='brown')
    plt.title('KL Divergence Loss\n(KL散度损失)', fontsize=12)
    plt.xlabel('预测值', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.grid(True)
    plt.legend()
    
    # Margin Ranking Loss
    plt.subplot(336)
    plt.plot(x.numpy(), margin.numpy(), label='Margin Loss', color='pink')
    plt.title('Margin Ranking Loss\n(边界排序损失)', fontsize=12)
    plt.xlabel('预测值', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.grid(True)
    plt.legend()
    
    # Custom Loss
    plt.subplot(337)
    plt.plot(x.numpy(), custom.numpy(), label='Custom Loss', color='orange')
    plt.title('Custom Loss\n(自定义损失)', fontsize=12)
    plt.xlabel('预测值', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.grid(True)
    plt.legend()
    
    # 所有损失函数对比
    plt.subplot(338)
    plt.plot(x.numpy(), mse.numpy(), label='MSE', alpha=0.6)
    plt.plot(x.numpy(), l1.numpy(), label='L1', alpha=0.6)
    plt.plot(x.numpy(), huber.numpy(), label='Huber', alpha=0.6)
    plt.plot(x.numpy(), bce.numpy(), label='BCE', alpha=0.6)
    plt.plot(x.numpy(), kl.numpy(), label='KL', alpha=0.6)
    plt.plot(x.numpy(), margin.numpy(), label='Margin', alpha=0.6)
    plt.plot(x.numpy(), custom.numpy(), label='Custom', alpha=0.6)
    plt.title('Loss Functions\n(损失函数对比)', fontsize=12)
    plt.xlabel('预测值', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 可视化损失函数
    plot_loss_curves()
    
    # 实际应用示例
    print("\n实际应用示例:")
    
    # 1. 回归问题
    print("\n1. 回归问题损失函数选择:")
    x = torch.randn(100, 1)  # 输入特征
    y = 2 * x + 1 + 0.1 * torch.randn(100, 1)  # 带噪声的目标值
    
    # 比较不同损失函数
    mse_loss = nn.MSELoss()(x, y)
    l1_loss = nn.L1Loss()(x, y)
    huber_loss = nn.HuberLoss()(x, y)
    
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"L1 Loss: {l1_loss.item():.4f}")
    print(f"Huber Loss: {huber_loss.item():.4f}")
    
    # 2. 分类问题
    print("\n2. 分类问题损失函数选择:")
    # 二分类
    binary_input = torch.randn(100, 1)
    binary_target = torch.randint(0, 2, (100, 1)).float()
    bce_loss = nn.BCEWithLogitsLoss()(binary_input, binary_target)
    print(f"Binary Cross Entropy Loss: {bce_loss.item():.4f}")
    
    # 多分类
    multi_input = torch.randn(100, 5)  # 5个类别
    multi_target = torch.randint(0, 5, (100,))
    ce_loss = nn.CrossEntropyLoss()(multi_input, multi_target)
    print(f"Cross Entropy Loss: {ce_loss.item():.4f}")
    
    # 损失函数的注意事项
    print("\n损失函数使用注意事项:")
    print("1. 回归问题:")
    print("   - MSE Loss: 对异常值敏感")
    print("   - L1 Loss: 对异常值较不敏感")
    print("   - Huber Loss: 结合MSE和L1的优点")
    
    print("\n2. 分类问题:")
    print("   - 二分类: BCEWithLogitsLoss")
    print("   - 多分类: CrossEntropyLoss")
    print("   - 需要注意标签格式要求")
    
    print("\n3. 自定义损失函数:")
    print("   - 继承nn.Module")
    print("   - 实现forward方法")
    print("   - 确保可导")
    
    # 性能测试
    print("\n性能测试:")
    large_input = torch.randn(1000, 50, requires_grad=True)
    large_target = torch.randn(1000, 50)
    
    import time
    
    # 测试MSE Loss
    start_time = time.time()
    _ = nn.MSELoss()(large_input, large_target)
    mse_time = time.time() - start_time
    
    # 测试L1 Loss
    start_time = time.time()
    _ = nn.L1Loss()(large_input, large_target)
    l1_time = time.time() - start_time
    
    print(f"MSE Loss计算时间: {mse_time*1000:.2f}ms")
    print(f"L1 Loss计算时间: {l1_time*1000:.2f}ms") 