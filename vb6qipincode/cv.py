# -*-  coding:utf-8 -*-
import csv
import numpy as np
from numpy import array, cov, corrcoef, mean
# import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
# from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import math
import torch
from torch.autograd import Variable


import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

    print(len(train))
    print(len(val))
    print(len(test))
    return train, val, test


def main():

    t = 0.5  # dropout
    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(D_in),

        torch.nn.Linear(D_in, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=t, inplace=False),

        torch.nn.Linear(H, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=t, inplace=False),

        torch.nn.Linear(H, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=t, inplace=False),

        torch.nn.Linear(H, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=t, inplace=False),

        torch.nn.Linear(H, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=t, inplace=False),

        torch.nn.Linear(H, D_out),
    ).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # model.load_state_dict(torch.load('ANN5w.pt'))
    loss_fn = torch.nn.MSELoss(size_average=False)
    den = sum((y - y.mean())**2)
    den_val = sum((y_val - y_val.mean())**2)
    den_test = sum((y_test - y_test.mean())**2)

    learning_rate = 1e-3  # 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1)
    R2_besttest = 0
    R2_bestval = 0
    R2_besttrain = 0
    y_best_pred = 0.0
    k = 0
    p = 0.8
    R2 = np.zeros(5 * 10**2)
    Loss = np.zeros(5 * 10**2)
    for t in range(5 * 10**2):  # 10**5
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(Variable(x))
        loss = loss_fn(y_pred.view(-1), Variable(y))
        if (t + 1) % 1 == 0:
            # Compute and print loss.
            y_pred = model(Variable(x_val))
            # Compute and print loss.
            loss_val = loss_fn(y_pred.view(-1), Variable(y_val))
            R2[t] = (1 - (loss_val.data[0] / den_val))
            Loss[t] = (loss_val.data[0] / ( int(0.1 * m)))
            if R2_bestval > 1 - (loss_val.data[0] / den_val):
                pass
            else:
                R2_bestval = (1 - (loss_val.data[0] / den_val))
                R2_besttrain = (1 - (loss.data[0] / den))
                y_best_pred = y_pred
                k = t + 1

            time2 = time.time()
        if (t + 1) % 100 == 0:
            print('training iter ', t + 1)
            print('training loss : ', loss.data[0] / int(p * m))
            print ('R^2 : ', (1 - (loss.data[0] / den)))
            print ('val loss=',
                   (loss_val.data[0] / ( int(0.1 * m))))
            print ('val R^2 : ',
                   (1 - (loss_val.data[0] / den_val)))
            print ('best step', k)
            print ('best val R^2', R2_bestval)
            print ('best train R^2', R2_besttrain)
        if R2_bestval>0.40:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred = model(Variable(x_test))
    # Compute and print loss.
    loss_test = loss_fn(y_pred.view(-1), Variable(y_test))
    R2_besttest = (1 - (loss_test.data[0] / den_test))
    return R2, Loss, R2_bestval, R2_besttest


if __name__ == '__main__':
    df = np.genfromtxt('data_with_training_information.csv', delimiter=',')
    m, p = df.shape
    df = df[~np.isnan(df).any(1)][:m // 10 * 10, :]
    D_in, H, D_out = p - 3, 30, 1

    fsize = m // 10
    idx = np.arange(fsize * 10)
    index = np.random.RandomState(seed=0).permutation(idx)

    for epochs in [0]:
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
        x = torch.from_numpy(a_train).float().cuda()
        y = torch.from_numpy(b_train).float().cuda()
        x_val = torch.from_numpy(a_val).float().cuda()
        y_val = torch.from_numpy(b_val).float().cuda()
        x_test = torch.from_numpy(a_test).float().cuda()
        y_test = torch.from_numpy(b_test).float().cuda()

        R2, Loss, R2_bestval, R2_test = main()
        # with open("./accuracy/1316-all13.txt", "a+") as text_file:
        #     text_file.write(str(epochs) + ' ' + str(run) + ' preg_train ' + 'accu_train ' + 'preg_test ' + 'accu_test ' + '0 ' + '/ ' + 'num_pre ' + 'TPR ' + 'FPR ' + 'AUC '
        #                     + 'thresh' + '\n' + result)
        # output += str(epochs) + '\n' + result + '\n'
        print('R2_bestval=%.5f,'%R2_bestval+'R2_test=%.5f'%R2_test)

        csvfile = open('R2-%d.csv' % epochs, 'a+', newline='')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(R2)
        csvfile.close()
        csvfile = open('Loss-%d.csv' % epochs, 'a+', newline='')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(Loss)
        csvfile.close()
