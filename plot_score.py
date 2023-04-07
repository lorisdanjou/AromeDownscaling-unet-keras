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
model_name = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

'''
Setup
'''
# params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
params = ['t2m', 'cape']
static_fields = []
dates_train = rangex(['2021010100-2021033100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)
echeances_baseline = range(6, 37, 6)
working_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/tests/'


y_pred = y(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields)
y_test = y(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields)
X_test = X(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields, resample=resample)
baseline = y(dates_test, echeances_baseline, baseline_location, data_static_location, params, base=True)
baseline.load()

X_test.X = np.load(working_dir + 'X_test.npy')
y_test.y = np.load(working_dir + 'y_test.npy')
y_pred.y = np.load(working_dir + 'y_pred.npy')


# print('y_pred shape : ' + str(y_pred.y.shape))
# print('y_test shape : ' + str(y_test.y.shape))
# print('baseline shape : ' + str(baseline.y.shape))
# print('X_test shape : ' + str(X_test.X.shape))


results = Results('t2m', 0, X_test, y_test, y_pred, baseline)
results.plot_firsts(working_dir, base=True)

mse_baseline_matrix, mse_pred_matrix, mse_baseline_global, mse_pred_global = results.mse_global()
mse_baseline_terre_matrix, mse_pred_terre_matrix, mse_baseline_terre_global, mse_pred_terre_global = results.mse_terre()
mse_baseline_mer_matrix, mse_pred_mer_matrix, mse_baseline_mer_global, mse_pred_mer_global = results.mse_mer()

print('rmse:')
print('  global:')
print('    baseline : ' + str(np.sqrt(np.mean(mse_baseline_global))))
print('    prediction : ' + str(np.sqrt(np.mean(mse_pred_global))))
print('  terre:')
print('    baseline : ' + str(np.sqrt(np.mean(mse_baseline_terre_global))))
print('    prediction : ' + str(np.sqrt(np.mean(mse_pred_terre_global))))
print('  mer:')
print('    baseline : ' + str(np.sqrt(np.mean(mse_baseline_mer_global))))
print('    prediction : ' + str(np.sqrt(np.mean(mse_pred_mer_global))))


results.plot_distrib_rmse(working_dir)



