# -*- coding=utf-8 -*-
import time
import argparse


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n','--number')
    res = ap.parse_args()
    time.sleep(1)
    print('Id:', res.number)

if __name__ == "__main__":
    run()