# -*- coding=utf-8 -*-
# 完整的模型训练流程
# 1. 网络定义 resnet
# 2. 数据集导入、预处理
# 3. 模型多卡训练
# 4. tensorboard可视化 TODO
# 5. 模型的定期保存以及重启训练
# 6. 模型评价 TODO
# 7. 加入钩子函数 TODO

from IPython import embed # 调试用
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

import time, datetime
import numpy as np

# 0. 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 512
learning_rate = 1e-4
epochs = 5
num_workers = 8

###################################################################
# utils
import datetime
import os
from collections import defaultdict


def timeStampStr():
    """获取时间戳字符串
    """   
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parseTimeStamp(time_string):
    """时间字符串转时间戳

    Args:
        time_str ([type]): [description]
    """
    return datetime.datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S").timestamp()

def getLatest(li):
    """获取最新的时间戳,返回一个排好序的list
    """
    time_li = li.copy()
    time_li = sorted(time_li, key=lambda date: parseTimeStamp(date))
    return time_li


def getFiles(path=r".\\"):
    """输入路径名称，返回该路径下的文件名、文件夹名构成的字典
    """
    dic = defaultdict(list)
    for r,d,files in os.walk(path):
        for f in files:
            dic[r].append(os.path.join(r,f))
    return dic

def getDeterminedFiles(path=r".\\", tail=None, keyword=None):
    """获取文件夹下指定后缀的文件或包含某关键字的文件
    依赖函数：getFiles
    """
    dic = getFiles(path=path)
    li = []
    for k, v in dic.items():
        li+=v
    if tail is not None:
        li = filter(lambda x: x.split('.')[-1]==tail, li)

    if keyword is not None:
        li = filter(lambda x: keyword in x, li)

    return list(li)

###################################################################


# 1. 网络定义模块，这里定义resnet-50
class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, use_bias = False):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, stride, padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.ident = nn.Sequential()
        if stride != 1 or input_channel != output_channel:
            self.ident = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 1, stride=stride, bias=use_bias),
                nn.BatchNorm2d(output_channel)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.ident(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_channel, channels, loops, out_dim, use_bias = False):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, channels[0], 7, 2, padding=3, bias=use_bias)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.mp1 = nn.MaxPool2d(3, 2, padding=1)
        self.layer1 = self._make_layer(channels[0], loops[0], channels[0], stride=1)
        self.layer2 = self._make_layer(channels[0], loops[1], channels[1], stride=2)
        self.layer3 = self._make_layer(channels[1], loops[2], channels[2], stride=2)
        self.layer4 = self._make_layer(channels[2], loops[3], channels[3], stride=2)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], out_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.mp1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.ap(out)
        out = torch.flatten(out, 1)
        # out = out.view(batch_size, -1)
        out = self.fc(out)
        return out
    
    def _make_layer(self, input_channel, loop, channel, stride=1):
        layers = nn.ModuleList()
        layers.append(ResBlock(input_channel, channel, stride=stride))
        for _ in range(1, loop):
            layers.append(ResBlock(channel, channel))
        return nn.Sequential(*layers)
        
def get_model(name = None):
    """
    检查最近的ckpt, 返回对应字符串，若没有则返回None
    """
    latest = None
    if name == None:
        li = getDeterminedFiles(path="./checkpoints/", tail="ckpt")
        if li != []: 
            time_li = list(map(lambda string: string.split('@')[1], li))
            latest_time = getLatest(time_li)[-1]
            latest = [i for i in li if latest_time in i][0]
    else:
        latest = name
    
    return latest


def resnet_18(channels = 3):
    resnet_18 = ResNet(channels, [64,128,256,512], [2, 2, 2, 2], 10)
    if torch.cuda.device_count() > 1: 
        print("Use ", torch.cuda.device_count(), "GPUs!")
        resnet_18 = nn.DataParallel(resnet_18.cuda())
    return resnet_18


def train():
    model_name = get_model()
    if model_name is None:
        net = resnet_18(channels = 1)
    else:
        net = resnet_18(channels = 1)
        net_dict = torch.load(model_name)
        net.load_state_dict(net_dict)
        # net = nn.DataParallel(net.cuda())
        # net = net.cuda()


    # 2. 数据集导入、预处理
    # 后面需要自己整一个自定义的
    train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                            train=False, 
                                            transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers = num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False,
                                            num_workers = num_workers)
    # 3. 模型多卡训练
    # 首先先定义loss和optmizer
    # 定义模型加载和保存模块
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            predict = net(x)
            loss = ce_loss(predict, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss {:.4f} '
                        .format(epoch+1, epochs, i+1, total_step, loss.item()))
        filename = "./checkpoints/resnet_model@{}@loss@{:.4f}.ckpt".format(timeStampStr(), loss.item())
        torch.save(net.state_dict(), filename)
    



if __name__ == "__main__":
    train()