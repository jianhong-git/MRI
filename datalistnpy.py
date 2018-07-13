# coding=utf-8
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
# matplotlib inline

AD = os.listdir('data/AD/')
# MCI = os.listdir('data/MCI/')
Normal = os.listdir('data/normal/')

data_AD = [(os.path.join('data/AD/', x), 0) for x in AD]
# data_MCI = [(os.path.join('data/MCI/', x), 2) for x in MCI]
data_Normal = [(os.path.join('data/normal/', x), 1) for x in Normal]

data_list = np.concatenate((data_AD, data_Normal))
# data_list = np.concatenate((data_list, data_MCI))
np.random.shuffle(data_list)
print(len(data_list))
np.save('ADNI_data_list_an', data_list)


# AD = os.listdir('data/301/AD/')
# Normal = os.listdir('data/301/normal/')

# data_AD = [(os.path.join('data/301/AD/', x), 0) for x in AD]
# data_Normal = [(os.path.join('data/301/normal/', x), 1) for x in Normal]

# data_list = np.concatenate((data_AD, data_Normal))
# np.random.shuffle(data_list)
# print(len(data_list))
# np.save('301_data_list', data_list)
