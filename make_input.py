import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from make_unet import *
import matplotlib.pyplot as plt
from data_loader import *

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/'


'''
Setup
'''
# params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
params = ["t2m"]
static_fields = ['SURFGEOPOTENTIEL']
dates_train = rangex(['2021020100-2021020100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022020100-PT24H']) # à modifier
dates_test = rangex(['2022020100-2022020100-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)


'''
Loading data
'''
X_train, y_train = load_X_y_r(dates_train, echeances, data_train_location, data_static_location, params, static_fields=static_fields)
X_valid, y_valid = load_X_y_r(dates_valid, echeances, data_valid_location, data_static_location, params, static_fields=static_fields)
X_test, y_test = load_X_y_r(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields)


X_train_path = '/cnrm/recyf/Data/users/danjoul/unet_experiments/X_train.npy'
X_valid_path = '/cnrm/recyf/Data/users/danjoul/unet_experiments/X_valid.npy'
X_test_path  = '/cnrm/recyf/Data/users/danjoul/unet_experiments/X_test.npy'
y_train_path = '/cnrm/recyf/Data/users/danjoul/unet_experiments/y_train.npy'
y_valid_path = '/cnrm/recyf/Data/users/danjoul/unet_experiments/y_valid.npy'
y_test_path  = '/cnrm/recyf/Data/users/danjoul/unet_experiments/y_test.npy'
np.save(X_train_path, X_train,  allow_pickle=True)
np.save(X_valid_path, X_valid,  allow_pickle=True)
np.save(X_test_path,  X_test,   allow_pickle=True)
np.save(y_train_path, y_train,  allow_pickle=True)
np.save(y_valid_path, y_valid,  allow_pickle=True)
np.save(y_test_path,  y_test,   allow_pickle=True)