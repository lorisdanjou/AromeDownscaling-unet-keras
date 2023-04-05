import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
import matplotlib.pyplot as plt
import data_loader as dl

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'


'''
Setup
'''
# params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
params = ["t2m"]
static_fields = []
dates_train = rangex(['2021020100-2021022800-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)

data_train = dl.Data(dates_train, echeances, data_train_location, data_static_location, params, static_fields=static_fields, resample=resample)
data_2 = data_train.copy()
means_X, stds_X, means_y, stds_y = data_train.standardize_X_y()
# data_train.destandardize_X_y(means_X, stds_X, means_y, stds_y)
# maxs_X, maxs_y = data_train.normalize_X_y()
# data_train.denormalize_X_y(maxs_X, maxs_y)
X_train, y_train = data_2.X, data_2.y

print(y_train.shape)
fig2, ax2 = plt.subplots()
im2 = ax2.imshow(y_train[0, :, :])
fig2.colorbar(im2, ax=ax2)
fig2.savefig('./y_test.png') 