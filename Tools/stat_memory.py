import sys
import pickle
import h5py
import numpy as np

h5_file = r"D:\data\sem_seg_h5\Table-1\train-01.h5"
# data = dict(zip(range(10000), range(10000)))
with h5py.File(h5_file, "r") as h5data:
    # h5data
    print(h5data.keys())
    dic = {'data': np.array(h5data['data']), 'label':'sad'}
    print("Size of data", sys.getsizeof(dic))