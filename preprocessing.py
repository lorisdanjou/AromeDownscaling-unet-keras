import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from make_unet import *
import matplotlib.pyplot as plt
from data_v2 import *
from time import perf_counter
import warnings

warnings.filterwarnings("ignore")

t0 = perf_counter()

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
baseline_location = '/cnrm/recyf/Data/users/danjoul/dataset/baseline/'
model_name = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'


'''
Setup
'''
# params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
params = ['t2m', 'cape', 'toa', 'tke', 'ts', 'u10', 'v10']
static_fields = ['SURFGEOPOTENTIEL', 'SURFIND.TERREMER', 'SFX.BATHY']
dates_train = rangex(['2020070100-2021053100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H', '2022040100-2022043000-PT24H', '2022060100-2022063000-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H', '2022050100-2022053100-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/tests/'

t1 = perf_counter()
print('setup time = ' + str(t1-t0))


'''
Load data
'''
data_train = load_data(
    dates_train, 
    echeances, 
    data_train_location,
    data_static_location,
    params,
    static_fields=static_fields,
    resample=resample)

data_valid = load_data(
    dates_valid, 
    echeances, 
    data_valid_location,
    data_static_location,
    params,
    static_fields=static_fields,
    resample=resample)

data_test = load_data(
    dates_test, 
    echeances, 
    data_test_location,
    data_static_location,
    params,
    static_fields=static_fields,
    resample=resample)

t2 = perf_counter()
print('loading time = ' + str(t2-t1))


'''
Preprocessing
'''
data_train = pad(data_train)
data_valid = pad(data_valid)
data_test = pad(data_test)

data_train = normalisation(data_train)
data_valid = normalisation(data_valid)
data_test  = normalisation(data_test)

data_train = standardisation(data_train)
data_valid = standardisation(data_valid)
data_test  = standardisation(data_test)

data_train.X.to_csv(output_dir + 'X_train.csv')
data_train.y.to_csv(output_dir + 'y_train.csv')
data_valid.X.to_csv(output_dir + 'X_valid.csv')
data_valid.y.to_csv(output_dir + 'y_valid.csv')

X_train = to_array(data_train.X)
y_train = to_array(data_train.y)
X_valid = to_array(data_valid.X)
y_valid = to_array(data_valid.y)

t3 = perf_counter()
print('preprocessing time = ' + str(t3-t2))

train_ds = tf.data.Dataset.from_tensor_slices(X_train)