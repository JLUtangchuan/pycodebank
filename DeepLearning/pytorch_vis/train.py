# -*- coding=utf-8 -*-
import os
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# 参数配置文件
from config import type_num, batch_size, learning_rate, epochs, writer_path, channels, writer_path, ckpt_path
from model import get_resnet_18
from data import train_loader, val_loader, test_loader, cats
from utils import timeStampStr
from validate import accuracy, validate, visPic, AverageMeter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# writer
writer = SummaryWriter(writer_path)

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def train():
    # load network
    net, start_epoch = get_resnet_18(load_path = '../checkpoints/', channels = channels)
    # loss
    focal_loss = FocalLoss(class_num=type_num, alpha=torch.tensor([0.25, 0.75]))
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # step per epoch
    total_step = len(train_loader)

    # train step
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
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        filename = ckpt_path + "/resnet_model@{}@Epoch@{}.ckpt".format(timeStampStr(), str(epoch+1))
        torch.save(net.state_dict(), filename)