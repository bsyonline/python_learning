# PyTorch模型 - 线性回归

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成示例数据
def generate_data(n_samples=100):
    # 生成随机输入数据
    X = np.linspace(-10, 10, n_samples)
    # 添加噪声的目标值
    y = 2 * X + 1 + np.random.randn(n_samples) * 2
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(X.reshape(-1, 1))
    y = torch.FloatTensor(y.reshape(-1, 1))
    
    return X, y

# 2. 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# 3. 训练模型
def train_model(model, X, y, epochs=100, lr=0.01):
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    
    # 训练循环
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

# 4. 可视化结果
def plot_results(X, y, model, losses):
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制数据和回归线
    X_np = X.numpy()
    y_np = y.numpy()
    
    # 预测值
    with torch.no_grad():
        y_pred = model(X).numpy()
    
    ax1.scatter(X_np, y_np, label='实际数据')
    ax1.plot(X_np, y_pred, 'r', label='回归线')
    ax1.set_title('线性回归拟合')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.legend()
    
    # 绘制损失曲线
    ax2.plot(losses)
    ax2.set_title('训练损失曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    plt.show()

# 5. 模型评估
def evaluate_model(model, X, y):
    with torch.no_grad():
        predictions = model(X)
        mse = nn.MSELoss()(predictions, y)
        mae = nn.L1Loss()(predictions, y)
        
    print("\n模型评估:")
    print(f"均方误差 (MSE): {mse.item():.4f}")
    print(f"平均绝对误差 (MAE): {mae.item():.4f}")
    
    # 获取模型参数
    w, b = model.linear.weight.item(), model.linear.bias.item()
    print(f"\n拟合的线性方程: y = {w:.4f}x + {b:.4f}")

if __name__ == "__main__":
    # 生成数据
    X, y = generate_data()
    
    # 创建和训练模型
    model = LinearRegression()
    losses = train_model(model, X, y)
    
    # 评估模型
    evaluate_model(model, X, y)
    
    # 可视化结果
    plot_results(X, y, model, losses) 