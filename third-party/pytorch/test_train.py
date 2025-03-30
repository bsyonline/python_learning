import torch
import torchvision.datasets
from torch import nn
from models.model import MyNN
from torch.utils.tensorboard import SummaryWriter















def train_model(myNN, train_loader, test_loader, loss_fn, optimizer, epochs, device, writer):
    """训练模型并记录指标"""
    print("开始训练...")
    for epoch in range(epochs):
        print(f"-------第 {(epoch + 1)} 轮训练开始-------")

        # 训练开始
        myNN.train()
        running_loss = 0
        accuracy = 0
        total = 0
        step = 0
        for data in train_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            
            # 前向传播
            output = myNN(imgs)
            loss = loss_fn(output, targets)

            # 反向传播和优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            accuracy += (output.argmax(1) == targets).sum().item()
            step += 1
            
            # 每100个批次记录一次训练损失
            if step % 100 == 0:
                avg_loss = running_loss / 100
                accuracy = 100 * accuracy / (targets.size(0) * 100)
                print(f"epoch: {epoch}, step: {step}, avg loss: {avg_loss}， accuracy: {accuracy}")
                writer.add_scalar("train_loss", avg_loss, step)
                writer.add_scalar("train_accuracy", accuracy, step)
                running_loss = 0
                accuracy = 0

        # 评估阶段
        myNN.eval()
        accuracy = 0
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)
                
                output = myNN(imgs)
                loss = loss_fn(output, targets)

                test_loss += loss.item()
                accuracy += (output.argmax(1) == targets).sum().item()


        print("整体测试集上的AVG Loss: {}".format(test_loss / len(test_loader)))
        print("整体测试集上的正确率: {}".format(accuracy / len(test_loader)))
        writer.add_scalar("test_loss", test_loss / len(test_loader), epoch)
        writer.add_scalar("test_accuracy", accuracy / len(test_loader), epoch)

        # 保存模型
        torch.save(myNN.state_dict(), "third-party/pytorch/models/myNN_{}.pth".format(epoch))
        print("模型已保存")



if __name__ == "__main__":
    # 0. 设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 准备数据集
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='third-party/pytorch/datasets/CIFAR10', train=True, download=True, transform=transformer)
    test_dataset = torchvision.datasets.CIFAR10(root='third-party/pytorch/datasets/CIFAR10', train=False, download=True, transform=transformer)
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print("训练数据集的长度为：{}".format(train_data_size))
    print("测试数据集的长度为：{}".format(test_data_size))

    # 2. 加载数据集
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 3. 构建模型
    myNN = MyNN().to(device)

    # 4. 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    # 5. 定义优化器
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(myNN.parameters(), lr=learning_rate)

    # 6. 创建tensorboard
    writer = SummaryWriter("third-party/pytorch/logs")

    # 7. 设置训练参数
    epochs = 10
    total_train_step = 0
    total_test_step = 0

    # 8. 训练模型
    train_model(myNN, train_loader, test_loader, loss_fn, optimizer, epochs, device, writer)

    writer.close()

