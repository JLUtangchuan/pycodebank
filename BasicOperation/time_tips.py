# -*- coding=utf-8 -*-

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
