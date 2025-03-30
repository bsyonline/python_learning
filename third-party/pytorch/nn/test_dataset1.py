from torch.utils.data import Dataset
import os
from PIL import Image

# 1. 自定义数据集类
class MyDataset(Dataset):
    """
    自定义图像数据集
    假设数据组织方式为：
    - hymenoptera_data_v1/
        - train/
            - ants/
                - aa.jpg
                - bb.jpg
                ...
            - bees/
                - cc.jpg
                - dd.jpg
                ...
        - val/
            - ants/
                ...
            - bees/
                ... 
    """
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.image_dir = os.path.join(self.root_dir, self.sub_dir)
        self.image_list = os.listdir(self.image_dir)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label = self.sub_dir
        # 加载图像
        image = Image.open(image_path)
        return image, label

    def __len__(self):
        return len(self.image_list)


ants_dataset = MyDataset("third_party/pytorch/datasets/hymenoptera_data/train", "ants")
ant_image, ant_label = ants_dataset[0]
filename = ant_image.filename
print(filename)
print(len(ants_dataset))
bees_dataset = MyDataset("third_party/pytorch/datasets/hymenoptera_data/train", "bees")
bee_image, bee_label = bees_dataset[0]
filename = bee_image.filename
print(filename)
print(len(bees_dataset))
train_dataset = ants_dataset + bees_dataset
print(len(train_dataset))

