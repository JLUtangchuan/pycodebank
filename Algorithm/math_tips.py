# -*- coding=utf-8 -*-
import numpy as np


def binaryEncode(i, NUM_DIG=20):
    """将一个10进制的数转成2进制

    Args:
        i ([type]): [description]
        NUM_DIG (int, optional): 表示这个2进制数的最大位数. Defaults to 10.
    """
    return np.array([i>>d & 1 for d in range(NUM_DIG)])[::-1]


if __name__ == "__main__":
    print(binaryEncode(16))