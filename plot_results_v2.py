import numpy as np
import pandas as pd
from results_v2 import *
from bronx.stdtypes.date import daterangex as rangex
import matplotlib.pyplot as plt
from matplotlib import colors

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
baseline_location = '/cnrm/recyf/Data/users/danjoul/dataset/baseline/test/'

'''
Setup
'''
# params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
params = ['t2m']
static_fields = []
dates_train = rangex(['2020070100-2021053100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H', '2022040100-2022043000-PT24H', '2022060100-2022063000-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H', '2022050100-2022053100-PT24H']) # à modifier
resample = 'r'
param = 't2m'
echeances = range(6, 37, 3)
working_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/unet_4/0.005_32_100/t2m/'


'''
Load Data
'''
results_df = load_data(working_dir, dates_test, echeances, resample, data_test_location, baseline_location, param=param)


'''
Plots
'''
plot_results(results_df, param, working_dir)
plot_score_maps(results_df, mae, 'mae', working_dir)
plot_distrib(results_df, mse, 'mse', working_dir)
plot_distrib(results_df, mae, 'mae', working_dir)
'''
Print mean scores
'''
mse_global_df = get_scores(results_df, mse, 'mse') 
mse_terre_df  = get_scores_terre(results_df, mse, 'mse')
mse_mer_df    = get_scores_mer(results_df, mse, 'mse')

print('mse:')
print('  global:')
print('    baseline : ' + str(mse_global_df['mse_baseline_mean'].mean()))
print('    prediction : ' + str(mse_global_df['mse_y_pred_mean'].mean()))
print('  terre:')
print('    baseline : ' + str(mse_terre_df['mse_baseline_mean'].mean()))
print('    prediction : ' + str(mse_terre_df['mse_y_pred_mean'].mean()))
print('  mer:')
print('    baseline : ' + str(mse_mer_df['mse_baseline_mean'].mean()))
print('    prediction : ' + str(mse_mer_df['mse_y_pred_mean'].mean()))
    