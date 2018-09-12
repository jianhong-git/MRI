# -*-  coding:utf-8 -*-
import csv
import numpy as np
from numpy import array, cov, corrcoef, mean
# import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn import datasets, linear_model
# from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import math
import torch
from torch.autograd import Variable


import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

time1 = time.time()
dirname, filename = os.path.split(os.path.abspath(__file__))


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

    # print(len(train))
    # print(len(val))
    # print(len(test))
    return train, val, test


def main():
    den_test = sum((b_test - b_test.mean())**2)
    regr = linear_model.LinearRegression()
    regr.fit(a_train, b_train)
    b_pred = regr.predict(a_test)
    mse = mean_squared_error(b_test, b_pred)
    se = mse * len(b_test)

    R2_besttest = (1 - (se / den_test))
    return R2_besttest


if __name__ == '__main__':
    df = np.genfromtxt('data_with_training_information.csv', delimiter=',')
    m, p = df.shape
    df = df[~np.isnan(df).any(1)][:m // 10 * 10, :]
    D_in, H, D_out = p - 3, 30, 1

    fsize = m // 10
    idx = np.arange(fsize * 10)
    index = np.random.RandomState(seed=0).permutation(idx)

    for epochs in range(10):
        train, val, test = split(epochs % 10)

        a_test = df[test, 1:-2]
        b_test = df[test, -2]
        a_val = df[val, 1:-2]
        b_val = df[val, -2]
        a_train = df[train, 1:-2]
        b_train = df[train, -2]
        a_train = preprocessing.scale(a_train)
        a_val = preprocessing.scale(a_val)
        a_test = preprocessing.scale(a_test)
        # x = torch.from_numpy(a_train).float().cuda()
        # y = torch.from_numpy(b_train).float().cuda()
        # x_val = torch.from_numpy(a_val).float().cuda()
        # y_val = torch.from_numpy(b_val).float().cuda()
        # x_test = torch.from_numpy(a_test).float().cuda()
        # y_test = torch.from_numpy(b_test).float().cuda()

        R2_test = main()
        # with open("./accuracy/1316-all13.txt", "a+") as text_file:
        #     text_file.write(str(epochs) + ' ' + str(run) + ' preg_train ' + 'accu_train ' + 'preg_test ' + 'accu_test ' + '0 ' + '/ ' + 'num_pre ' + 'TPR ' + 'FPR ' + 'AUC '
        #                     + 'thresh' + '\n' + result)
        # output += str(epochs) + '\n' + result + '\n'
        print('epochs=%2d,' % epochs + 'R2_test=%.5f' % R2_test)
