# -*- coding=utf-8 -*-
import numpy as np
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



import datetime
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# Numpy,Pandas
import numpy as np
import pandas as pd
# plt.style.use('ggplot')  #风格设置近似R这种的ggplot库
import seaborn as sns

# matplotlib,seaborn,pyecharts

sns.set_style('whitegrid')

warnings.filterwarnings('ignore') 

# 绘制混淆矩阵图
def drawConfusionMatrix(data):          
    data = data.corr()
    mask = np.zeros_like(data)
    indices = np.triu_indices_from(data)
    mask[indices] = True
    f, (ax1) = plt.subplots(1, 1, figsize = (10, 9))

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    ax1 =sns.heatmap(data, ax = ax1, vmin = -1, vmax = 1, \
        cmap = cmap, square = False, linewidths = 0.5, \
        mask = mask, cbar_kws={'orientation': 'vertical', \
                                                'ticks': [-1, -0.5, 0, 0.5, 1]})
    ax1.set_xticklabels(ax1.get_xticklabels(), size = 16); 
    ax1.set_yticklabels(ax1.get_yticklabels(), size = 16); 
    ax1.set_title('Data', size = 20)

    plt.show()

# 绘制散点图
def drawScatterPlot():
    midwest = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest.csv")

    # Prepare Data 
    # Create as many colors as there are unique midwest['category']
    categories = np.unique(midwest['category'])
    colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

    # Draw Plot for Each Category
    plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')

    for i, category in enumerate(categories):
        plt.scatter('area', 'poptotal', 
                    data=midwest.loc[midwest.category==category, :], 
                    s=20, c=colors[i], label=str(category))

    # Decorations
    plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),
                xlabel='Area', ylabel='Population')

    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.title("Scatterplot of Midwest Area vs Population", fontsize=22)
    plt.legend(fontsize=12)    
    plt.show()    



def main():
    # data = pd.DataFrame(np.random.rand(10,20))
    # drawConfusionMatrix(data)
    drawScatterPlot()

if __name__ == "__main__":
    main()
