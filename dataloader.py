# dataloder.py
import os
import cv2
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 设置图像尺寸
img_dim = 64
PATH = "img/processed/"


class DataGenerater(Dataset):
    """
    数据集定义
    """

    def __init__(self, path=PATH, transform=None):
        """
        构造函数
        """
        self.dir = path
        self.datalist = os.listdir(PATH)
        self.image_size = (img_dim, img_dim)
        self.transform = transform

    def __getitem__(self, idx):
        """
        每次迭代时返回数据和对应的标签
        """
        img_path = self.dir + self.datalist[idx]
        img = io.imread(img_path)
        img = transform.resize(img, self.image_size)
        img = img.transpose((2, 0, 1))  # 转换为通道优先
        img = img.astype("float32")

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        """
        返回整个数据集的总数
        """
        return len(self.datalist)


# 定义数据集
train_dataset = DataGenerater()

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True,  # 使用 pin_memory 可以加速数据传输
    drop_last=True,
)

# # 可视化部分
# for batch_id, data in enumerate(train_loader):
#     plt.figure(figsize=(12, 12))
#     try:
#         for i in range(100):
#             image = data[i].numpy().transpose((1, 2, 0))  # 转换为 HWC
#             print(f"data[i].shape{data[i].shape}")
#             plt.subplot(10, 10, i + 1)
#             plt.imshow(image, vmin=-1, vmax=1)
#             plt.axis("off")
#             plt.xticks([])
#             plt.yticks([])
#             plt.subplots_adjust(wspace=0.1, hspace=0.1)
#         plt.suptitle("\n Training Images", fontsize=30)
#         plt.show()
#         break
#     except IOError:
#         print(IOError)
