# PyTorch基础 - 神经网络

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 定义简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(2, 4)    # 输入层到隐藏层
        self.relu = nn.ReLU()         # 激活函数
        self.fc2 = nn.Linear(4, 1)    # 隐藏层到输出层
    
    def forward(self, x):
        # 定义前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. 训练神经网络
def train_network():
    print("训练神经网络示例：")
    
    # 生成训练数据
    X = torch.randn(100, 2)  # 100个样本，每个2个特征
    y = (X[:, 0] + X[:, 1]).view(-1, 1)  # 目标：两个特征之和
    
    # 创建模型和优化器
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练过程
    losses = []
    for epoch in range(100):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    
    return losses

# 3. 保存和加载模型
def save_load_model(model, filename="model.pth"):
    print("\n保存和加载模型示例：")
    
    # 保存模型
    torch.save(model.state_dict(), filename)
    print(f"模型已保存到 {filename}")
    
    # 加载模型
    new_model = SimpleNet()
    new_model.load_state_dict(torch.load(filename))
    new_model.eval()
    print("模型已加载")
    
    return new_model

# 4. 使用模型进行预测
def make_predictions(model):
    print("\n使用模型进行预测示例：")
    
    # 生成测试数据
    test_data = torch.tensor([[1.0, 2.0], [0.0, -1.0]], requires_grad=False)
    
    # 进行预测
    with torch.no_grad():
        predictions = model(test_data)
    
    # 打印结果
    for i, pred in enumerate(predictions):
        x1, x2 = test_data[i]
        print(f"输入: ({x1:.1f}, {x2:.1f}), 预测: {pred.item():.4f}, 实际: {(x1 + x2).item():.4f}")

# 5. 可视化训练过程
def plot_training_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 训练模型
    losses = train_network()
    
    # 创建和训练模型
    model = SimpleNet()
    losses = train_network()
    
    # 保存和加载模型
    loaded_model = save_load_model(model)
    
    # 进行预测
    make_predictions(loaded_model)
    
    # 可视化训练过程
    plot_training_loss(losses) 