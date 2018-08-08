#coding=utf-8
import tensorflow as tf
import numpy as np
import nibabel as nib
import math, os, time
time1 = time.time()

a=np.load('conv1.npy')
print(a.shape)
# y1=np.load('Jacobi2.npy')
# x1=range(len(y1))
# y2=np.load('2grid.npy')
# x2=range(len(y2))
# y3=np.load('5grid.npy')
# x3=range(len(y3))

# a1=plt.plot(y1[:20],label='Jacobi')
# a2=plt.plot(y2,label='2grid')
# a3=plt.plot(y3,label='5grid')

# plt.legend()
# plt.show()

from skimage import io,data
img=a[0,20,:,:,3]
dst=io.imshow(img)
print(type(dst))
io.show()