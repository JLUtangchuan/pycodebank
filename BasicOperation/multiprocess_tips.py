# -*- coding=utf-8 -*-
import random
import os
from multiprocessing import Process

def run_script(info):
    os.system('python run_test.py -n {}'.format(info))
    print('完成进程', info)





def main():
    threads = 5
    for i in range(threads):
        p = Process(target=run_script, args=(str(i)))
        p.start()
        print('开启进程:',i)


if __name__ == "__main__":
    main()