# PyTorch模型 - 逻辑回归

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. 生成二分类数据
def generate_data(n_samples=1000):
    # 生成二分类数据
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1
    )
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# 2. 定义逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# 3. 训练模型
def train_model(model, X_train, y_train, epochs=100, lr=0.1):
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    
    # 训练循环
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        losses.append(loss.item())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

# 4. 评估模型
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        # 预测概率
        y_pred_proba = model(X_test)
        # 转换为类别标签
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
def plot_results(X_train, y_train, X_test, y_test, model, losses):
    plt.figure(figsize=(15, 5))
    
    # 绘制决策边界
    plt.subplot(1, 2, 1)
    
    # 创建网格点
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )
    
    # 预测网格点的类别
    with torch.no_grad():
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
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

# 6. 预测新样本
def predict_samples(model, samples):
    model.eval()
    with torch.no_grad():
        probas = model(torch.FloatTensor(samples))
        predictions = (probas >= 0.5).float()
        
        for i, (proba, pred) in enumerate(zip(probas, predictions)):
            print(f"样本 {i+1}:")
            print(f"预测概率: {proba.item():.4f}")
            print(f"预测类别: {int(pred.item())}")
            print()

if __name__ == "__main__":
    # 生成数据
    X_train, X_test, y_train, y_test = generate_data()
    
    # 创建和训练模型
    model = LogisticRegression(input_dim=2)
    losses = train_model(model, X_train, y_train)
    
    # 评估模型
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # 可视化结果
    plot_results(X_train, y_train, X_test, y_test, model, losses)
    
    # 预测新样本
    new_samples = np.array([
        [1.0, 1.0],
        [-1.0, -1.0],
        [0.0, 0.0]
    ])
    print("预测新样本:")
    predict_samples(model, new_samples) 