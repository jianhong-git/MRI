# -*-  coding:utf-8 -*-
import csv
import numpy as np
from numpy import array, cov, corrcoef,mean
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
time1 = time.time()
dirname, filename = os.path.split(os.path.abspath(__file__))

df=np.genfromtxt('data_with_training_information.csv',delimiter=',')
df = df[~np.isnan(df).any(1)]
df = df[df[:,-4]==0]
m,p=df.shape
# b = df[1:,-1]
# a = df[1:,1:-1]
# a=preprocessing.scale(a)
#pca = PCA()
#pca.fit(a)
#a = pca.transform(a)
# b= (b-67.63)/81.24
# a,b = shuffle(a,b)

#dg=np.genfromtxt('09_modified.csv',delimiter=',')
#dg = dg[~np.isnan(dg).any(1)]
#dg = shuffle(dg)
#b_test = dg[1:,-1]
#n,_=dg.shape
#a_test = dg[1:,1:-1]
#a_test = preprocessing.scale(a_test)
#pca = PCA()
#pca.fit(a_test)
#a_test = pca.transform(a_test)
# b_test = (b_test-67.63)/81.24
#a_test, b_test = shuffle(a_test, b_test)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
#N, N_test,D_in, H, D_out = m,n, p-2, 30, 1
D_in, H, D_out = p-3, 30, 1

# m=m+n
# print(m)
# p= 0.9
# index=np.random.randint(m, size=m-int(p*m))
#a=np.concatenate((a, a_test))
#b=np.concatenate((b, b_test))
# print(a.shape,b.shape)
# a_test=a[int(p*m):,:]
df_test=df[df[:,-1]==0]
df_train=df[df[:,-1]==1]
a_test=df_test[1:,1:-2]
b_test=df_test[1:,-2]
a_train=df_train[1:,1:-2]
b_train=df_train[1:,-2]
a_train=preprocessing.scale(a_train)
a_test=preprocessing.scale(a_test)
# print(a_test)
# a=a[:int(p*m),:]
# np.random.shuffle(b)
# b_test=b[int(p*m):]
# b=b[:int(p*m)]
a_train, b_train = shuffle(a_train, b_train)
a_test, b_test = shuffle(a_test, b_test)
#print(b.shape, b_test.shape, b_test)
# print(a.shape,a_test.shape,b.shape,b_test.shape)

# x=torch.from_numpy(a.astype('float32'))
# y=torch.from_numpy(b.astype('float32'))
x=torch.from_numpy(a_train).float().cuda()
y=torch.from_numpy(b_train).float().cuda()
x_test=torch.from_numpy(a_test).float().cuda()
y_test=torch.from_numpy(b_test).float().cuda()

t = 0.5 # dropout
# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.BatchNorm1d(D_in),

    torch.nn.Linear(D_in, H),
    torch.nn.BatchNorm1d(H),
    torch.nn.ReLU(),
    torch.nn.Dropout(p= t, inplace=False),

    torch.nn.Linear(H,H),
    torch.nn.BatchNorm1d(H),
    torch.nn.ReLU(),
    torch.nn.Dropout(p= t, inplace=False),

    torch.nn.Linear(H,H),
    torch.nn.BatchNorm1d(H),
    torch.nn.ReLU(),
    torch.nn.Dropout(p= t, inplace=False),

    torch.nn.Linear(H,H),
    torch.nn.BatchNorm1d(H),
    torch.nn.ReLU(),
    torch.nn.Dropout(p= t, inplace=False),

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
den=sum((y-y.mean())**2)
den_test=sum((y_test-y_test.mean())**2)

#csvfile = open('time.csv','a+',newline ='')
#writer = csv.writer(csvfile, delimiter=',')
#writer.writerow(['iter','train loss','train R2','test loss','test R2','time'])
#csvfile.close()
# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
learning_rate = 1e-3 # 1e-4
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1)
R2_besttest = 0
R2_besttrain = 0
y_best_pred = 0.0
k=0
p=0.9
for t in range(5*10**4): #5*10**4
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(Variable(x))
    loss = loss_fn(y_pred.view(-1), Variable(y))
    if (t+1)%1==0:
    # Compute and print loss.
        print('training loss in iter ',t+1, ': ',loss.data[0]/int(p*m))
        print('R^2 : ',1-(loss.data[0]/den) )
        y_pred = model(Variable(x_test))
        # Compute and print loss.
        loss_test = loss_fn(y_pred.view(-1), Variable(y_test))
        print('test loss=', loss_test.data[0]/(m-int(p*m)))
        print('test R^2 : ',1-(loss_test.data[0]/den_test))
        if R2_besttest > 1-(loss_test.data[0]/den_test):
            pass
        else:
            R2_besttest = 1-(loss_test.data[0]/den_test)
            R2_besttrain = 1-(loss.data[0]/den)
            y_best_pred = y_pred
            k = t+1
        print('best step',k)
        print('best test R^2', R2_besttest)
        print('best train R^2', R2_besttrain,'\n')

        time2 = time.time()
        #csvfile = open('time.csv','a+',newline ='')
        #writer = csv.writer(csvfile, delimiter=',')
        #writer.writerow([t+1,loss.data[0]/int(p*m),1-(loss.data[0]/den),loss_test.data[0]/(m-int(p*m)),1-(loss_test.data[0]/den_test),time2 - time1])
        #csvfile.close()

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

# print(y_test.cpu().numpy().size,y_best_pred.cpu().data.squeeze().numpy().size)
# plt.plot(y_test.numpy(), y_best_pred.data.squeeze().numpy()-y_test.numpy(),'ro')
# plt.plot(y_test.cpu().numpy(),y_best_pred.cpu().data.squeeze().numpy(),'ro')
# plt.plot([0,600],[0,600])
# plt.xlabel("Actual pyridoxal 5'-phosphate concentration (ug/L)")
# plt.ylabel("Predicted pyridoxal 5'-phosphate concentration (ug/L)")
# plt.title(r"$R^2$=0.45")
# plt.ylabel('difference')
# plt.savefig('result.png')
# plt.show()
# y_pred = model(Variable(x_test))
# # Compute and print loss.
# loss_test = loss_fn(y_pred, Variable(y_test))
# print('test loss=', loss_test.data[0])
# print('R^2 : ',1-loss_test.data[0]/den_test)
# torch.save(model.state_dict(), 'ANN5w.pt')
