import numpy as np
from os.path import exists
from data_loader import get_shape_2km5, get_shape_500m
from results import *
from bronx.stdtypes.date import daterangex as rangex

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
baseline_location = '/cnrm/recyf/Data/users/danjoul/dataset/baseline/'

'''
Baseline
'''
shape_500m = get_shape_500m()
shape_2km5 = get_shape_2km5(resample=resample)

y = np.zeros(shape=[len(dates_test), len(echeances_baseline), shape_500m[0], shape_500m[1]], dtype=np.float32)

for i_d, d in enumerate(dates_test):
    try:
        filepath = baseline_location + 'GG9B_' + d.isoformat() + 'Z_t2m.npy'
        y[i_d, :, :, :] = np.load(filepath).transpose([2, 0, 1])
    except FileNotFoundError:
        print('missing day')

print('initial y shape : ' + str(y.shape))
y = y.reshape((-1, shape_500m[0], shape_500m[1]))
print('reshaped y shape : ' + str(y.shape))
baseline = y

print('shapes : ')
print('y_pred : ' + str(y_pred.shape))
print('y_test : ' + str(y_test.shape))
print('baseline : ' + str(baseline.shape))


results = Results('t2m', 0, X_test, y_test, y_pred, baseline)

metric_global, metric_terre, metric_mer = results.metric(rmse)

print('RMSEs : ')
print('RMSE global : ' + str(metric_global))
print('RMSE terre : ' + str(metric_terre))
print('RMSE mer : ' + str(metric_mer))

score_global, score_terre, score_mer = results.score(rmse)

print('scores : ')
print('score global : ' + str(score_global))
print('score terre : ' + str(score_terre))
print('score mer : ' + str(score_mer))

results.plot_20(working_dir, base=True)
