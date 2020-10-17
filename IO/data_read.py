# -*- coding=utf-8 -*-

import numpy as np
import os
from collections import defaultdict
import json

# 文件名获取相关
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
    for k,v in dic.items():
        li+=v
    if tail is not None:
        li = filter(lambda x: x.split('.')[-1]==tail, li)

    if keyword is not None:
        li = filter(lambda x: keyword in x, li)

    return list(li)

def getTxtAsMat(f, sep=',', header=None):
    """读取txt文件，将其转化成pandas.DataFrame对象
    输入的可以是filename，也可以是url
    还可以是open的文件
    """
    import pandas as pd
    return pd.read_csv(f, sep=sep, header=header)

def getDirNames(path):
    """返回路径下文件夹名称

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    dbtype_list = os.listdir(path)
    for dbtype in dbtype_list:
        if os.path.isfile(os.path.join(path,dbtype)):
            dbtype_list.remove(dbtype)
    return dbtype_list

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def getJson(filename):
    with open(filename, 'r') as f:
        dic = json.load(f) 
    return dic

if __name__ == "__main__":
    # mkdir('test')
    getJson('./hello.json')