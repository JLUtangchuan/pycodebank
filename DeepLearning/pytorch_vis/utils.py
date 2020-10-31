# -*- coding=utf-8 -*-
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