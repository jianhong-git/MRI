# -*-  coding:utf-8 -*-
import csv
import numpy as np
from numpy import array, cov, corrcoef, mean
#import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
#from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import math
import torch
from torch.autograd import Variable


import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

time1 = time.time()
dirname, filename = os.path.split(os.path.abspath(__file__))

df = np.genfromtxt('data_with_training_information.csv', delimiter=',')
df = df[~np.isnan(df).any(1)]
m, p = df.shape
D_in, H, D_out = p - 3, 30, 1

df_test = df[df[:, -1] == 0]
df_train = df[df[:, -1] == 1]
a_test = df_test[:, 1:-2]
b_test = df_test[:, -2]
a_train = df_train[:, 1:-2]
b_train = df_train[:, -2]
a_train = preprocessing.scale(a_train)
a_test = preprocessing.scale(a_test)
a_train, b_train = shuffle(a_train, b_train)
a_test, b_test = shuffle(a_test, b_test)
x = torch.from_numpy(a_train).float().cuda()
y = torch.from_numpy(b_train).float().cuda()
x_test = torch.from_numpy(a_test).float().cuda()
y_test = torch.from_numpy(b_test).float().cuda()

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

    # torch.nn.Linear(H,H),
    # torch.nn.BatchNorm1d(H),
    # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H, H),
    # torch.nn.BatchNorm1d(H),
    # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H, H),
    # torch.nn.BatchNorm1d(H),
    # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H,H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H,H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H,H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H, H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H, H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H,H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H,H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H,H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H, H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H, H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H, H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    # torch.nn.Linear(H, H),
    # torch.nn.BatchNorm1d(H),
    # # torch.nn.ReLU(),
    # torch.nn.Dropout(p= t, inplace=False),

    torch.nn.Linear(H, D_out),
).cuda()
model = torch.nn.DataParallel(model).cuda()
# model.load_state_dict(torch.load('ANN5w.pt'))
loss_fn = torch.nn.MSELoss(size_average=False)
den = sum((y - y.mean())**2)
den_test = sum((y_test - y_test.mean())**2)

learning_rate = 1e-3  # 1e-4
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1)
R2_besttest = 0
R2_besttrain = 0
y_best_pred = 0.0
k = 0
p = 0.9
R2 = np.zeros(5 * 10**5)
Loss = np.zeros(5 * 10**5)
for t in range(5 * 10**2):  # 10**5
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(Variable(x))
    loss = loss_fn(y_pred.view(-1), Variable(y))
    if (t + 1) % 1 == 0:
        # Compute and print loss.
        y_pred = model(Variable(x_test))
        # Compute and print loss.
        loss_test = loss_fn(y_pred.view(-1), Variable(y_test))
        R2[t] = 1 - (loss_test.data[0].cpu() / den_test)
        Loss[t] = loss_test.data[0].cpu() / (m - int(p * m))
        if R2_besttest > 1 - (loss_test.data[0] / den_test):
            pass
        else:
            R2_besttest = 1 - (loss_test.data[0] / den_test)
            R2_besttrain = 1 - (loss.data[0] / den)
            y_best_pred = y_pred
            k = t + 1

        time2 = time.time()
    if (t + 1) % 100 == 0:
        print('training loss in iter ', t + 1,
              ': ', loss.data[0].cpu() / int(p * m))
        print ('R^2 : ', 1 - (loss.data[0].cpu() / den))
        print ('test loss=', loss_test.data[0] / (m - int(p * m)))
        print ('test R^2 : ', 1 - (loss_test.data[0] / den_test))
        print ('best step', k)
        print ('best test R^2', R2_besttest)
        print ('best train R^2', R2_besttrain)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

print(y_test.cpu().numpy().size, y_best_pred.cpu().data.squeeze().numpy().size)
# plt.plot(y_test.numpy(), y_best_pred.data.squeeze().numpy()-y_test.numpy(),'ro')
'''plot function'''
# plt.plot(y_test.cpu().numpy(),y_best_pred.cpu().data.squeeze().numpy() - y_test.cpu().numpy(),'ro')
# plt.plot([0,600],[0,0])
# plt.xlabel("Actual pyridoxal 5'-phosphate concentration (ug/L)")
# plt.ylabel("Difference between actual and predicted pyridoxal (ug/L)")
# plt.title(r"$R^2$=0.47")
# # plt.ylabel('difference')
# plt.savefig('result.png')

# plt.show()
# y_pred = model(Variable(x_test))
# Compute and print loss.
# loss_test = loss_fn(y_pred, Variable(y_test))
# print('test loss=', loss_test.data[0])
# print('R^2 : ',1-loss_test.data[0]/den_test)
# torch.save(model.state_dict(), 'ANN5w.pt')
'''write y_pred y_test'''
# csvfile = open('p_value.csv','a+',newline ='')
csvfile = open('p_value_4hidden.csv', 'a+')
writer = csv.writer(csvfile, delimiter=',')
# writer.writerow([t+1,loss.data[0]/int(p*m),1-(loss.data[0]/den),loss_test.data[0]/(m-int(p*m)),1-(loss_test.data[0]/den_test),time2 - time1])
writer.writerow(np.concatenate((np.array(['label']), y_test.cpu().numpy()), 0))
writer.writerow(np.concatenate(
    (np.array(['prediction']), y_best_pred.cpu().data.squeeze().numpy()), 0))
csvfile.close()
