# -*- coding=utf-8 -*-
# 存放参数
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

type_num = 2
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
learning_rate = 1e-4
epochs = 5

writer_path = './train_log'
channels = 1

ckpt_path = "/home/tangchuan/workdir/pycodebank/DeepLearning/checkpoints"