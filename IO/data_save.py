# -*- coding=utf-8 -*-
import numpy as np              # 导入numpy库 
import pandas as pd             # 导入pandas库 
import matplotlib as mpl        # 导入matplotlib库 
import matplotlib.pyplot as plt  
import seaborn as sns           # 导入seaborn库 

mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['font.serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串, 

import json
import pickle
PI = 3.14159

# numpy的数据保存
def saveNumpy(data, filename='data.npy'):
    np.save(filename, data)
# json的数据保存
def saveJson(data, filename='data.json'):
    with open(filename,'w+') as f:
        json.dump(data, f)

# pandas的数据保存
def savePandas(data, filename='data.csv'):
    """以csv格式保存

    Args:
        data ([type]): [description]
        filename (str, optional): [description]. Defaults to 'data.csv'.
    """
    data = pd.DataFrame(data)
    data.to_csv(filename)


# 图片保存
def savePic(filename='data.svg'):
    """保存图片
    若是插入word推荐svg
    若是插入latex推荐pdf
    """    
    x = np.arange(-10*PI, 10*PI, 0.001)
    y = np.sin(2*x+0.3) + np.sin(0.4*x+0.1)
    plt.figure(figsize=(16, 8))
    plt.plot(x, y)

    plt.savefig(filename)
    


def main():
    # numpy
    data = np.random.rand(5,10)
    saveNumpy(data)

    # json
    js = {
        "name": "TangChuan",
        "age": 22,
        "birthday": "1998-07-01"
    }
    saveJson(js)

    # pandas
    data = pd.read_csv("http://download.tensorflow.org/data/iris_training.csv")
    savePandas(data)

    # mpl
    savePic()
    


if __name__ == "__main__":
    main()