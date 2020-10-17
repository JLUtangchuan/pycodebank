# -*- coding=utf-8 -*-
# 完整的模型训练流程
# 1. 网络定义 resnet
# 2. 数据集导入、预处理
# 3. 模型多卡训练
# 4. tensorboard可视化
# 5. 模型的定期保存以及重启训练
# 6. 模型评价
# 7. 加入钩子函数 TODO

# 附加任务
# A1. 分布式训练优化 TODO
# A2. 自定义数据集 TODO
# A3. 各个模块拆分成独立文件、函数解耦 TODO

from IPython import embed # 调试用
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import time, datetime
import numpy as np
from PIL import ImageDraw, ImageFont

from PytorchCode.loss_col import FocalLoss

# 0. 参数设置
writer = SummaryWriter('./train_log')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 512
learning_rate = 1e-4
epochs = 5
num_workers = 4
channels = 1
type_num = 2
torch.backends.cudnn.benchmark = True

###################################################################
# utils
import datetime
import time
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

def timer(func):
    def time_wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('Took {} second'.format(end_time - start_time))
        return res
    return time_wrapper

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
    resnet_18 = ResNet(channels, [64,128,256,512], [2, 2, 2, 2], type_num)
    if torch.cuda.device_count() > 1: 
        print("Use ", torch.cuda.device_count(), "GPUs!")
        resnet_18 = nn.DataParallel(resnet_18.cuda())
    return resnet_18

def accuracy(predict, y, ks=(1,)):
    """计算topk准确率

    Args:
        predict ([type]): [description]
        y ([type]): [description]
        k (tuple, optional): [description]. Defaults to (1,).
    """
    with torch.no_grad():
        size = y.shape[0]
        maxk = max(ks)
        _, idx = predict.topk(maxk, 1)
        return (len([1 for i, t in zip(y, idx[:,:k]) if i in t])/size for k in ks)

def validate(net, val_loader, loss_func):
    """验证
    计算验证集loss
    计算top_k accuracy
    计算 Recall Precision
    计算mAP
    Args:
        val_loader ([type]): [description]
        loss_func ([type]): [description]
    """
    total_loss = AverageMeter("Val Loss", ":.4f")
    top1 = AverageMeter("Top1 accuracy", ":.4f")
    # top2 = AverageMeter("Top2 accuracy", ":.4f")
    net.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            predict = net(x)
            size = x.shape[0]
            loss = loss_func(predict, y)
            total_loss.update(loss.item(), size)
            t1, t2 = accuracy(predict, y, ks=(1, 2))
            top1.update(t1, size)
            # top2.update(t2, size)
    return total_loss, top1

def visPic(net, writer, loader, cats, step=0):
    """可视化结果
    """
    tor2img = transforms.ToPILImage()
    with torch.no_grad():
        x, y = next(iter(loader))
        predict = net(x)
        _, cls_num = predict.max(axis=1)
        pred = [cats[i.item()] for i in cls_num]
        gt = [cats[i.item()] for i in y]

        ttfont = ImageFont.truetype("/home/tangchuan/Download/simhei.ttf",20)
        
        imgs = []
        for p, g, img in zip(pred, gt, x):
            img = tor2img(img)
            draw = ImageDraw.Draw(img)
            draw.text((90, 100), f"GT:{g}-PRED:{p}",fill=(200), font=ttfont)
            imgs.append(img)

        ts = transforms.ToTensor()
        timg = [ts(i) for i in imgs]
        grid = torchvision.utils.make_grid(timg, padding=2)
        writer.add_image('Val Images', grid, step)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def train():
    model_name = get_model()
    if model_name is None:
        net = resnet_18(channels = channels)
        start_epoch = 0
    else:
        net = resnet_18(channels = channels)
        net_dict = torch.load(model_name)
        net.load_state_dict(net_dict)

        start_epoch = int(model_name.split('Epoch@')[1].split('.ckpt')[0])
        # net = nn.DataParallel(net.cuda())
        # net = net.cuda()


    # 2. 数据集导入、预处理
    # 后面需要自己整一个自定义的
    from PytorchCode.data import train_loader, val_loader, test_loader, cats
    # train_dataset = torchvision.datasets.MNIST(root='../../data', 
    #                                         train=True, 
    #                                         transform=transforms.ToTensor(),  
    #                                         download=True)

    # test_dataset = torchvision.datasets.MNIST(root='../../data', 
    #                                         train=False, 
    #                                         transform=transforms.ToTensor())

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
    #                                         batch_size=batch_size, 
    #                                         shuffle=True,
    #                                         num_workers = num_workers)

    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
    #                                         batch_size=batch_size, 
    #                                         shuffle=False,
    #                                         num_workers = num_workers)
    
    # 3. 模型多卡训练
    # 首先先定义loss和optmizer
    # 定义模型加载和保存模块
    # ce_loss = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(class_num=type_num, alpha=torch.tensor([0.25, 0.75]))
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    # 4. 模型评价
    # 两部分：1. 每次前向传播过程中计算的metric
    #         2. 每个epoch结束后跑一遍验证集，拿到验证集loss和训练集loss
    total_step = len(train_loader)
    for epoch in range(start_epoch, start_epoch+epochs):
        train_loss = AverageMeter("Train Loss", ":.4f")
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            net.train()
            predict = net(x)
            # loss = ce_loss(predict, y)
            loss = focal_loss(predict, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), y.shape[0])

            # tensorboard
            writer.add_scalar('train loss', train_loss.val, 1+i+total_step*epoch)

        # 跑验证集
        val_loss, top1 = validate(net, test_loader, focal_loss)
        # 可视化
        visPic(net, writer, val_loader, cats, epoch)
        # tensorboard
        # 服务器指令
        # tensorboard --logdir=./train_log --port 8032
        # 本地指令
        # ssh -L 8032:127.0.0.1:8032 tangchuan@59.72.118.127
        # 本地访问
        # 127.0.0.1:8032
        # 
        # scaler
        # writer.add_scalar('train loss', train_loss.avg, epoch)
        # writer.add_scalar('val loss', val_loss.avg, epoch)
        # writer.add_scalar('top1 accuracy', top1.avg, epoch)

        print("Epoch [{}/{}] ".format(epoch+1, start_epoch+epochs), val_loss, train_loss,  top1)

        # 模型保存
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        filename = "./checkpoints/resnet_model@{}@Epoch@{}.ckpt".format(timeStampStr(), str(epoch+1))
        torch.save(net.state_dict(), filename)
    
@timer
def main():
    train()


def summary_test():
    from PytorchCode.model_summary import summary
    net = resnet_18(channels = channels)
    x = torch.rand((batch_size, channels, 224, 224))
    summary(net, x)

if __name__ == "__main__":
    main()
    # summary_test()

