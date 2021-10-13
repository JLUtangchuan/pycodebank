#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   torch_analyze.py
@Time    :   2021/01/15 20:40:42
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   pytorch模型分析工具学习
'''
import torch
import torchvision

# 查看每层的输出维度

# from torchsummary import summary
# from PytorchCode.model_summary import summary # 这个可能对于一些特殊的模型不太work
# summary(your_model, input_size=(channels, H, W))

# 关于统计模型运行时间的分析工具
# 1. torchprof
# 2. nnprof
# 3. torch.autograd.profile
# 4. pytorch 自带 bottleneck 工具
# 5. python 自带的 cProfile profile 工具


# torchprof


def torchprof_test():
    import torchprof
    model = torchvision.models.alexnet(pretrained=False).cuda()
    x = torch.rand([64, 3, 224, 224]).cuda()

    # `profile_memory` was added in PyTorch 1.6, this will output a runtime warning if unsupported.
    with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
        model(x)

    # equivalent to `print(prof)` and `print(prof.display())`
    print(prof.display(show_events=False))

# nnprof (目前项目比较新，可能bug比较多？)

# 使用nnprof进行模型推断时间计算，找到耗时长的部分
# 在windows安装的时候遇到一个坑，就是open默认的是gbk，而应该是utf-8，需要clone一下repo然后在setup.py中改一下

# from nnprof import profile, ProfileMode

# import torch
# import torchvision

# model = torchvision.models.alexnet(pretrained=False)
# x = torch.rand([1, 3, 224, 224])

# # mode could be anyone in LAYER, OP, MIXED, LAYER_TREE
# mode = ProfileMode.LAYER

# with profile(model, mode=mode) as prof:
#     y = model(x)

# print(prof.table(average=False, sorted_by="cpu_time"))


# bottleneck
# 获取帮助 python -m torch.utils.bottleneck -h
# 有两种模式： CPU-only-mode or CUDA-mode



if __name__ == "__main__":
    pass
