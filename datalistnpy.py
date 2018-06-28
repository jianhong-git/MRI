#coding=utf-8
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
#matplotlib inline

AD=os.listdir('data/AD/')
# # #MCI=os.listdir('ADNIprocess/MCI/')
Normal=os.listdir('data/normal/')

data_AD=[(os.path.join('data/AD/', x), 0) for x in AD]
# # #data_MCI=[(os.path.join('ADNIprocess/MCI/', x), 2) for x in MCI]
data_Normal=[(os.path.join('data/normal/', x), 1) for x in Normal]

data_list=np.concatenate((data_AD, data_Normal))

# np.random.shuffle(data_list)

# AD=os.listdir('data/301/eventest/AD/')
# # # #MCI=os.listdir('ADNIprocess/MCI/')
# Normal=os.listdir('data/301/eventest/normal/')

# data_AD=[(os.path.join('data/301/eventest/AD/', x), 0) for x in AD]
# # # #data_MCI=[(os.path.join('ADNIprocess/MCI/', x), 2) for x in MCI]
# data_Normal=[(os.path.join('data/301/eventest/normal/', x), 1) for x in Normal]

# data_list2=np.concatenate((data_AD, data_Normal))

# np.random.shuffle(data_list2)

# # # np.save('data_list', data_list2)
# data_list=np.concatenate((data_list, data_list2))

# AD=os.listdir('pred/')

# data=[(os.path.join('pred/', x),0) for x in AD]

#data_list=np.concatenate((data))


np.save('ADNI_data_list', data_list)
