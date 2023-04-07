import numpy as np
from os.path import exists
from data import *
from results import *
from bronx.stdtypes.date import daterangex as rangex


data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
baseline_location = '/cnrm/recyf/Data/users/danjoul/dataset/baseline/'

'''
Setup
'''
params = ['t2m']
static_fields = []
dates_train = rangex(['2021010100-2021033100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)
echeances_baseline = range(6, 37, 6)

working_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/unet_4/0.005_32_64/normalized_standardized/'
X_test = np.load(working_dir + 'X_test.npy')
y_test = np.load(working_dir + 'y_test.npy') 
y_pred = np.load(working_dir + 'y_pred.npy')
y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))


y_pred = y(dates_test, echeances, data_test_location, data_static_location, params, )
y_pred.y = y_pred
y_test = y(dates_test, echeances, data_test_location, data_static_location, params)
X_test = Data_X(dates_test, echeances, data_test_location, data_static_location, params)
data_baseline = Data_baseline(dates_test, echeances_baseline, baseline_location, data_static_location, params)

results = Results('t2m', 0, data_X_test, data_y_test, data_y_pred, data_baseline)
results.plot_firsts(working_dir, base=True)

rmse_baseline_matrix, rmse_pred_matrix, rmse_baseline_global, rmse_pred_global = results.rmse_global()
print('rmse:')
print('baseline : ' + str(np.mean(rmse_baseline_global)))
print('prediction : ' + str(np.mean(rmse_pred_global)))

