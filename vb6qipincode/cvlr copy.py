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

df = np.genfromtxt('seed1.csv', delimiter=',')
df = df[~np.isnan(df).any(1)]
df_test = df[df[:, -1] == 3]
df_train = df[df[:, -1] == 1]
a_test = df_test[:, 1:-2]
b_test = df_test[:, -2]
a_train = df_train[:, 1:-2]
b_train = df_train[:, -2]
# a_train = preprocessing.scale(a_train)
# a_test = preprocessing.scale(a_test)

den_test = sum((b_test - b_test.mean())**2)
regr = linear_model.LinearRegression()
regr.fit(a_train, b_train)
b_pred = regr.predict(a_test)
# mse = mean_squared_error(b_test, b_pred)
# se = mse * len(b_test)
se = sum((b_pred - b_test.mean())**2)


R2_besttest=se / den_test
# R2_besttest = (1 - (se / den_test))
R2 = regr .score(a_test, b_test)
print('R2_test=%.5f,R2=%.5f' % (R2_besttest, R2))
# print(regr.coef_, regr.intercept_)
