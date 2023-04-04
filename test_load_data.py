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
X_train, y_train = data_train.load_normalized_X_y()

print(y_train.shape)
fig2, ax2 = plt.subplots()
im2 = ax2.imshow(y_train[0, :, :])
fig2.colorbar(im2, ax=ax2)
fig2.savefig('./y_test.png') 