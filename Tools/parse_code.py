# -*- coding=utf-8 -*-
# 这个工具的用途主要是转格式，填用户代码片段
import json

name = "vis"
prefix = "vis"

code = '''import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

# Version
print(mpl.__version__)  #> 3.0.0
print(sns.__version__)  #> 0.9.0
'''



def parseCode(code):
    li = code.split('\n')
    
    li.append("$0")
    return li

def make_json(name,prefix,code,description = None):
    if description is None:
        description = name
    code = parseCode(code)
    dic = dict({"prefix": prefix, "body": code, "description": description})
    
    return {name:dic}

if __name__ == "__main__":    
    string = make_json(name, prefix, code)
    
    with open('code.json','w') as f:
        json.dump(string, f, indent=2)
    
    


