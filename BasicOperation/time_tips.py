# -*- coding=utf-8 -*-

import datetime
import time

# 与时间相关的函数


def timer(func):
    def time_wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('Took %f second' % (end_time - start_time))
        return res
    return time_wrapper


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
    """获取最新的时间戳
    """
    time_li = li.copy()
    time_li = sorted(time_li, key=lambda date: parseTimeStamp(date))
    return time_li
    


if __name__ == "__main__":
    # print(timeStampStr())
    # li = ['2020-10-04 18:40:11', '2020-10-04 18:40:31', '2020-10-04 18:40:19']
    # print(getLatest(li))
    print(parseTimeStamp('2020-10-04 18:40:11'))