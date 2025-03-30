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
            - ants_image/
                - aa.jpg
                - bb.jpg
                ...
            - bees_image/
                - cc.jpg
                - dd.jpg
                ...
            - ants_label/
                - aa.txt
                - bb.txt
                ...
            - bees_label/
                - cc.txt
                - dd.txt
                ...
        - val/
            - ants_image/
                ...
            - bees_image/
                ... 
            - ants_label/
                ...
            - bees_label/
                ... 
    """
    def __init__(self, root_dir, image_dir, label_dir):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_list = os.listdir(os.path.join(self.root_dir, self.image_dir))
        self.label_list = os.listdir(os.path.join(self.root_dir, self.label_dir))
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        label_name = self.label_list[idx]
        image_path = os.path.join(self.root_dir, self.image_dir, image_name)
        image = Image.open(image_path)
        label_path = os.path.join(self.root_dir, self.label_dir, label_name)
        with open(label_path, "r") as f:
            label_name = f.read().strip()
        return image, label_name

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)


ants_dataset = MyDataset("third_party/pytorch/datasets/hymenoptera_data_v1/train", "ants_image", "ants_label")
ant_image, ant_label = ants_dataset[0]
filename = ant_image.filename
print(filename)
print(ant_label)
print(len(ants_dataset))
bees_dataset = MyDataset("third_party/pytorch/datasets/hymenoptera_data_v1/train", "bees_image", "bees_label")
bee_image, bee_label = bees_dataset[0]
filename = bee_image.filename
print(filename)
print(bee_label)
print(len(bees_dataset))
train_dataset = ants_dataset + bees_dataset
print(len(train_dataset))

