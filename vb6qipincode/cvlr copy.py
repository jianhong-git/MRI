# -*-  coding:utf-8 -*-
import csv
import numpy as np
from numpy import array, cov, corrcoef, mean
from sklearn import metrics
from sklearn import preprocessing
from sklearn import datasets, linear_model
# from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import time
import os


def split(i):  # i-th fold

    if i < 8:
        test = index[fsize * i:fsize * (i + 1)]
        val = index[fsize * (i + 1):fsize * (i + 2)]
    elif i < 9:
        test = index[fsize * i:fsize * (i + 1)]
        val = index[-(fsize):]
    else:
        test = index[-(fsize):]
        val = index[:fsize]
    train = [x for x in index if x not in test and x not in val]

    return train, val, test


df = np.genfromtxt('seed400.csv', delimiter=',')
m, p = df.shape
df = df[~np.isnan(df).any(1)][:m // 10 * 10, :]
# D_in, H, D_out = p - 3, 30, 1
D_in, H, D_out = p - 3, 40, 1

fsize = m // 10
idx = np.arange(fsize * 10)
index = np.random.RandomState(seed=0).permutation(idx)

for epochs in range(10):
    train, val, test = split(epochs % 10)

    a_test = df[test, 1:-2]
    b_test = df[test, -2]
    a_train = df[train, 1:-2]
    b_train = df[train, -2]

    # R2_bestval = 0
    # while R2_bestval < 0.4:

    den_test = sum((b_test - b_test.mean())**2)
    regr = linear_model.LinearRegression()
    regr.fit(a_train, b_train)
    R2 = regr.score(a_test, b_test)
    print('fold %1d, R2=%.3f' % (epochs, R2))
    # print(regr.coef_, regr.intercept_)
