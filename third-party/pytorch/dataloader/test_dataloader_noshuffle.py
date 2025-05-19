import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler


class SimpleDataset(Dataset):
    def __init__(self, size=17):
        self.data = torch.arange(size)
        self.labels = torch.arange(size)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def test_dataloader():
    # 创建数据集
    dataset = SimpleDataset(size=17)
    
    batch_size = 4
    drop_last = True
    sampler = RandomSampler(dataset)
    

    print(f"\nTesting with batch_size={batch_size}, drop_last={drop_last}")
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 禁用随机打乱
        drop_last=drop_last,
        sampler=sampler
    )
    
    # 收集每个epoch中使用的样本
    all_samples = []
    
    # 运行3个epoch
    for epoch in range(3):
        epoch_samples = []
        print(f"Epoch {epoch + 1}:")
        for batch_idx, (data, labels) in enumerate(dataloader):
            epoch_samples.extend(data.tolist())
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1} data: {data.tolist()}")

        all_samples.append(epoch_samples)
        print(f"Number of batches: {len(epoch_samples) // batch_size + (1 if len(epoch_samples) % batch_size != 0 else 0)}")
        print(f"Total samples used: {len(epoch_samples)}")

    

if __name__ == "__main__":
    test_dataloader()