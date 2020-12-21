#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   collection_tips.py
@Time    :   2020/12/21 15:14:22
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   Collection库简单介绍
'''

from collections import namedtuple


def convert(dictionary):
    """简单地将dict转化为namedtuple
    """
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


if __name__ == "__main__":
    dic = {'a': 1}
    t = convert(dic)
    print(t.a)
