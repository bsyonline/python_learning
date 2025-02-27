# PyTorch数据集示例 - 自定义数据集

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. 自定义数据集类
class CustomImageDataset(Dataset):
    """
    自定义图像数据集
    假设数据组织方式为：
    - data_dir/
        - images/
            - image1.jpg
            - image2.jpg
            ...
        - labels.csv (包含 'image_name' 和 'label' 列)
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.transform = transform
        
        # 读取标签文件
        self.labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
        
        # 检查图像文件是否存在
        self.image_files = []
        self.labels = []
        for _, row in self.labels_df.iterrows():
            img_path = os.path.join(self.image_dir, row['image_name'])
            if os.path.exists(img_path):
                self.image_files.append(img_path)
                self.labels.append(row['label'])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_files[idx]).convert('RGB')
        label = self.labels[idx]
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 2. 数据转换和增强
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# 3. 数据集准备示例
def prepare_sample_dataset(base_dir):
    """创建示例数据集结构"""
    os.makedirs(os.path.join(base_dir, 'images'), exist_ok=True)
    
    # 创建示例标签文件
    labels_df = pd.DataFrame({
        'image_name': [f'image_{i}.jpg' for i in range(10)],
        'label': [i % 2 for i in range(10)]  # 二分类示例
    })
    labels_df.to_csv(os.path.join(base_dir, 'labels.csv'), index=False)
    
    # 创建示例图像（这里只是示例，实际应用中需要真实图像）
    for img_name in labels_df['image_name']:
        img = Image.new('RGB', (100, 100), color='white')
        img.save(os.path.join(base_dir, 'images', img_name))

# 4. 数据加载器创建
def create_data_loaders(data_dir, batch_size=32, train_ratio=0.8):
    # 创建数据集
    full_dataset = CustomImageDataset(
        data_dir,
        transform=get_transforms(train=True)
    )
    
    # 划分训练集和验证集
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader

# 5. 数据可视化
def visualize_batch(data_loader):
    # 获取一个批次的数据
    images, labels = next(iter(data_loader))
    
    # 反标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    
    # 显示图像
    fig = plt.figure(figsize=(15, 5))
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 6. 使用示例
def dataset_usage_example():
    # 创建示例数据集
    data_dir = './custom_dataset'
    prepare_sample_dataset(data_dir)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(data_dir, batch_size=4)
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 可视化一个批次的数据
    print("\n显示训练集中的一个批次:")
    visualize_batch(train_loader)

if __name__ == "__main__":
    dataset_usage_example() 