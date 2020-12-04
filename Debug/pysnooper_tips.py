#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pysnooper.py
@Time    :   2020/12/03 21:39:04
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   Pysnooper使用
'''

import pysnooper
import numpy as np
import torch
# export PYSNOOPER_DISABLED=1 # This makes PySnooper not do any snooping

AAA = 3

@pysnooper.snoop(watch=('AAA'), prefix='FUNC1 ')
def test_func(inputs):
    global AAA
    a = inputs
    a.append(3)
    b = np.random.rand(3,5)
    AAA += 1
    return a

def test_statement():
    with pysnooper.snoop(prefix='Statement '):
        a1 = 1
        a2 = np.array([13,3,3,4,4,6])
        a2 = a2.reshape(2,-1)
        a3 = 1
    a3 = a2
#################################################################
def large(l):
    return isinstance(l, list) and len(l) > 5

def print_list_size(l):
    return 'list(size={})'.format(len(l))

def print_ndarray(a):
    return 'ndarray(shape={}, dtype={})'.format(a.shape, a.dtype)

def print_tensor(tensor):
    return 'torch.Tensor(shape={}, dtype={}, device={})'.format(tensor.shape, tensor.dtype, tensor.device)

custom_repr = ((large, print_list_size), (np.ndarray, print_ndarray), (torch.Tensor, print_tensor))

#################################################################

@pysnooper.snoop(custom_repr=custom_repr)
def sum_to_x(x):
    l = list(range(x))
    a = np.zeros((10,10))
    c = torch.from_numpy(a).cuda(1)

    return sum(l)


if __name__ == "__main__":
    # test_func(inputs=[1,2,3])
    # test_statement()
    # max_variable_length
    sum_to_x(20)