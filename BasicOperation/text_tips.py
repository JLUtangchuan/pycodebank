# -*- coding=utf-8 -*-
# 与文本相关的操作
# 关键词替换
# 介绍第三方库 parse 
from parse import *

def parseUsage():
    # 特定格式字符匹配
    p1 = parse("I am {}, {} years old, {}", "I am Jack, 27 years old, male")
    print(p1)
    p2 = parse("I am {name}, {age} years old, {gender}", "I am Jack, 27 years old, male")
    print(p2)

    # 返回结果对象分析
    # parse 返回结果 result
    # fixed 一个无名称匹配结果list
    # named 一个有名称匹配结果dict

if __name__ == "__main__":
    pass