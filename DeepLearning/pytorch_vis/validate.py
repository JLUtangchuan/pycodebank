# -*- coding=utf-8 -*-
# 一些validation的方法
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from PIL import ImageDraw, ImageFont


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