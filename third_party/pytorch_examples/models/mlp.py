# PyTorch模型 - 多层感知机 (MLP)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 数据准备
def prepare_data(n_samples=1000):
    # 生成月牙形数据
    X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

# 2. 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super(MLP, self).__init__()
        
        # 构建多层网络
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),  # 批标准化
                nn.Dropout(0.2)              # dropout正则化
            ])
            prev_dim = hidden_dim
        
        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        # 将所有层组合成一个序列
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 3. 训练模型
def train_model(model, X_train, y_train, epochs=200, batch_size=32, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_X, batch_y in dataloader:
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            epoch_losses.append(loss.item())
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 记录每个epoch的平均损失
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return losses

# 4. 评估模型
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        # 预测
        y_pred_proba = model(X_test)
        y_pred = (y_pred_proba >= 0.5).float()
        
        # 计算准确率
        accuracy = (y_pred == y_test).float().mean()
        
        # 计算损失
        criterion = nn.BCELoss()
        test_loss = criterion(y_pred_proba, y_test)
        
        print("\n模型评估:")
        print(f"测试集准确率: {accuracy.item():.4f}")
        print(f"测试集损失: {test_loss.item():.4f}")
    
    return y_pred, y_pred_proba

# 5. 可视化结果
def plot_results(X_train, y_train, model, losses, scaler):
    plt.figure(figsize=(15, 5))
    
    # 绘制决策边界
    plt.subplot(1, 2, 1)
    
    # 创建网格点
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    
    # 预测网格点的类别
    model.eval()
    with torch.no_grad():
        grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        Z = model(grid)
        Z = (Z >= 0.5).float().numpy()
    
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和数据点
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.8)
    plt.title('决策边界和训练数据')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 6. 模型预测
def predict_samples(model, samples, scaler):
    # 标准化输入数据
    samples_scaled = scaler.transform(samples)
    
    model.eval()
    with torch.no_grad():
        # 预测
        probas = model(torch.FloatTensor(samples_scaled))
        predictions = (probas >= 0.5).float()
        
        print("\n新样本预测结果:")
        for i, (sample, proba, pred) in enumerate(zip(samples, probas, predictions)):
            print(f"样本 {i+1} {sample}:")
            print(f"预测概率: {proba.item():.4f}")
            print(f"预测类别: {int(pred.item())}")
            print()

if __name__ == "__main__":
    # 准备数据
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    # 创建模型
    model = MLP(
        input_dim=2,
        hidden_dims=[64, 32],  # 两个隐藏层
        output_dim=1
    )
    
    # 训练模型
    losses = train_model(model, X_train, y_train)
    
    # 评估模型
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # 可视化结果
    plot_results(X_train, y_train, model, losses, scaler)
    
    # 预测新样本
    new_samples = np.array([
        [0.5, 0.5],
        [-0.5, -0.5],
        [0.0, 1.0]
    ])
    predict_samples(model, new_samples, scaler) 