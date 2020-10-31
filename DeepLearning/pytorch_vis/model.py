# -*- coding=utf-8 -*-
# 调用一个经典的有预训练模型的模型

from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


from config import type_num
from utils import getDeterminedFiles, getLatest


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

    def _forward_operation(self, x):
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

    def forward(self, x):
        out = self._forward_operation(x)
        return out
    
    def _make_layer(self, input_channel, loop, channel, stride=1):
        layers = nn.ModuleList()
        layers.append(ResBlock(input_channel, channel, stride=stride))
        for _ in range(1, loop):
            layers.append(ResBlock(channel, channel))
        return nn.Sequential(*layers)
        
def get_model(load_path = './checkpoints/', name = None):
    """
    检查最近的ckpt, 返回对应字符串，若没有则返回None
    """
    latest = None
    if name == None:
        li = getDeterminedFiles(path=load_path, tail="ckpt")
        if li != []: 
            time_li = list(map(lambda string: string.split('@')[1], li))
            latest_time = getLatest(time_li)[-1]
            latest = [i for i in li if latest_time in i][0]
    else:
        latest = name
    
    return latest


def resnet_18(channels):
    resnet_18 = ResNet(channels, [64,128,256,512], [2, 2, 2, 2], type_num)
    if torch.cuda.device_count() > 1: 
        print("Use ", torch.cuda.device_count(), "GPUs!")
        resnet_18 = nn.DataParallel(resnet_18.cuda())
    return resnet_18

def get_resnet_18(load_path, channels = 3):
    model_name = get_model(load_path)
    if model_name is None:
        net = resnet_18(channels = channels)
        start_epoch = 0
    else:
        net = resnet_18(channels = channels)
        net_dict = torch.load(model_name)
        net.load_state_dict(net_dict)    
        start_epoch = int(model_name.split('Epoch@')[1].split('.ckpt')[0])
    return net, start_epoch
  