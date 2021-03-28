import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
import h5py as h5

W01 = 30
W12 = 150
W23 = 75
W34 = 40
W45 = 20
W56 = 3

# Take in params
# Read H5 file
f = h5.File("MLPmodel_1hand_float32.h5", "r")
# Get and print list of datasets within the H5 file
datasetNames = list(f.keys())

b1 = f['dense_50']['dense_50']['bias:0'].value
b2 = f['dense_51']['dense_51']['bias:0'].value
b3 = f['dense_52']['dense_52']['bias:0'].value
b4 = f['dense_53']['dense_53']['bias:0'].value
b5 = f['dense_54']['dense_54']['bias:0'].value
w1 = f['dense_50']['dense_50']['kernel:0'].value
w2 = f['dense_51']['dense_51']['kernel:0'].value
w3 = f['dense_52']['dense_52']['kernel:0'].value
w4 = f['dense_53']['dense_53']['kernel:0'].value
w5 = f['dense_54']['dense_54']['kernel:0'].value

print('{')
for i in range(W01):
    print('{',end='')
    for j in range(W12):
        print(w1[i][j],sep='',end='')
        if j != W12-1:
            print(',', sep='',end='')
    print('}')
    if i != W01-1:
        print(',', sep='',end='')
print('}')