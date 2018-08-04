# coding=utf-8
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
# matplotlib inline
AD = os.listdir('data/test/AD/')
# MCI = os.listdir('data/test/MCI/')
Normal = os.listdir('data/test/normal/')

data_AD = [(os.path.join('data/test/AD/', x), 0) for x in AD]
# data_MCI = [(os.path.join('data/test/MCI/', x), 2) for x in MCI]
data_Normal = [(os.path.join('data/test/normal/', x), 1) for x in Normal]

data_list = np.concatenate((data_AD, data_Normal))
# data_list = np.concatenate((data_AD, data_MCI))
# data_list = np.concatenate((data_MCI, data_Normal))
# data_list = np.concatenate((data_list, data_MCI))
np.random.shuffle(data_list)
print(len(data_list))
np.save('ADNI_test_an', data_list)


# AD = os.listdir('data/301/AD/')
# Normal = os.listdir('data/301/normal/')

# data_AD = [(os.path.join('data/301/AD/', x), 0) for x in AD]
# data_Normal = [(os.path.join('data/301/normal/', x), 1) for x in Normal]

# data_list = np.concatenate((data_AD, data_Normal))
# np.random.shuffle(data_list)
# print(len(data_list))
# np.save('301_data_list', data_list)
