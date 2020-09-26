# -*- coding=utf-8 -*-
import sys
import os

# 与系统操作相关的命令


def getCurDir():
    """获取当前绝对路径
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    return dir_path

def exeCmd():
    """执行系统命令
    """
    os.system('date')


if __name__ == "__main__":
    exeCmd()