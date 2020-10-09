# -*- coding=utf-8 -*-
# 更多内容看博客中的笔记
# https://blog.tangchuan.ink/2020/10/08/ji-zhu/bian-cheng/python-debug/

import time
import pdb
pdb.set_trace() # 设置断点
import sys
print(sys.version)

def main():
    for i in range(5):
        print(f"range[{i+1}|5]") # 注意这种表示字符串表示方法 f"{valuename}"
        time.sleep(1)
        

if __name__ == "__main__":
    main()