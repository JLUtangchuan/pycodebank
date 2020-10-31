# -*- coding=utf-8 -*-
import argparse

from train import train
    

def main(args):
    train()


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-bs", "--batch_size", default=128)
    args = parse.parse_args()
    main(args)