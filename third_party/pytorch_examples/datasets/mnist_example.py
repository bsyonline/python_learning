# PyTorch数据集示例 - MNIST手写数字识别

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 数据加载和预处理
def load_mnist_data(batch_size=64):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载训练集
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # 加载测试集
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

# 2. 定义CNN模型
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

# 3. 训练模型
def train_model(model, train_loader, test_loader, epochs=10, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # 计算训练损失
        train_loss = evaluate_loss(model, train_loader, criterion, device)
        train_losses.append(train_loss)
        
        # 计算测试准确率
        test_accuracy = evaluate_accuracy(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Test Accuracy: {test_accuracy:.4f}')
    
    return train_losses, test_accuracies

# 4. 评估函数
def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 5. 可视化函数
def plot_results(train_losses, test_accuracies):
    plt.figure(figsize=(12, 4))
    
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制测试准确率
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, test_loader, device, num_images=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 显示图像和预测结果
    fig = plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f'P: {predicted[i]}\nT: {labels[i]}')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, test_loader = load_mnist_data()
    
    # 创建模型
    model = MNISTNet()
    
    # 训练模型
    train_losses, test_accuracies = train_model(
        model, train_loader, test_loader,
        epochs=5, device=device
    )
    
    # 可视化结果
    plot_results(train_losses, test_accuracies)
    
    # 可视化预测
    visualize_predictions(model, test_loader, device) 