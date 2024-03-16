from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片,将灰度图像转化成彩色图像
        if img.mode != 'RGB':
            # raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

