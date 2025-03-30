import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 1. 数据转换
data_transformer = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. 官方数据集
train_dataset = torchvision.datasets.CIFAR10(root='third_party/pytorch/datasets/CIFAR10', train=True, download=True)
# 使用转换器
test_dataset = torchvision.datasets.CIFAR10(root='third_party/pytorch/datasets/CIFAR10', train=False, download=True, transform=data_transformer)

# 3. 数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 3. 可视化数据
writer = SummaryWriter("third_party/pytorch/datasets/logs")
step = 0
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)
    writer.add_images("test_data", imgs, step)
    step += 1

writer.close()
