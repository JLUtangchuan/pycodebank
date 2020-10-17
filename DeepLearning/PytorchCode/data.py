# -*- coding=utf-8 -*-
import os
from collections import defaultdict

import cv2
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# 其实用torchvision.datasets.ImageFolder这个处理这种格式的分类图片会更便捷
num_workers = 8
batch_size = 128
DATA_PATH = '/home/tangchuan/workdir/data/chest_xray/'
trans=transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.CenterCrop(1024),
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5)),
    ]
)

cats = ['NORMAL', 'PNEUMONIA']



################################################################
# utils
def getFiles(path=r".\\"):
    """输入路径名称，返回该路径下的文件名、文件夹名构成的字典
    """
    dic = defaultdict(list)
    for r,d,files in os.walk(path):
        for f in files:
            dic[r].append(os.path.join(r,f))
    return dic
################################################################


class CateDataset(Dataset):
    def __init__(self, root_path, transform, names=None):
        super(CateDataset, self).__init__()
        self.root_path = root_path
        self.transform = transform
        self.names = names
        if names == None:
            self.names = self._getnames()
        # 获取所有文件名和图片 
        self.filenames = []
        for i, n in enumerate(self.names):
            fs = os.listdir(os.path.join(self.root_path, n))
            self.filenames += list(zip(fs, (i,)*len(fs)))
        self.total_num = len(self.filenames)

    def _getnames(self):
        dbtype_list = os.listdir(self.root_path)
        for dbtype in dbtype_list:
            if os.path.isfile(os.path.join(self.root_path, dbtype)):
                dbtype_list.remove(dbtype)
        return dbtype_list

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        fn, label = self.filenames[idx] #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(os.path.join(self.root_path, self.names[label], fn))
        img = self.transform(img)
        return img, label

train_dataset = CateDataset(os.path.join(DATA_PATH, 'train'), trans, names=cats)
val_dataset = CateDataset(os.path.join(DATA_PATH, 'val'), trans, names=cats)
test_dataset = CateDataset(os.path.join(DATA_PATH, 'test'), trans, names=cats)

train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers = num_workers)
val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=16, 
                            shuffle=False)
test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=batch_size,  # 小一点，方便可视化
                            shuffle=False,
                            num_workers = num_workers)

if __name__ == "__main__":
    for (x,y) in train_loader:
        print(x.shape)