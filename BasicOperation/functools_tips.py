# -*- coding=utf-8 -*-

# partial
# reduce
# namedtuple
# lru_cache
# recursive_repr
# singledispatch
# wraps
# update_wrapper
# total_ordering
# get_cache_token
# cmp_to_key

from functools import *

def fixedParamFunc(func, *args, **kwargs):
    """封装部分参数固定的函数
    """
    return partial(func, *args, **kwargs)

@lru_cache()
def fibonacci(n):
    """lru_cache的主要作用是记录近期调用的重复实参的返回结果，当下次再次调用时直接从cache中返回；
    节省了计算花费的时间，适合计算开销大、无随机性的函数
    @lru_cache()
    """
    if n < 2:
        return n
    return fibonacci(n-2) + fibonacci(n-1)