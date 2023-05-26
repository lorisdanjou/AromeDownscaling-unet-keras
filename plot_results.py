import numpy as np
import pandas as pd
from results.results import *
from bronx.stdtypes.date import daterangex as rangex
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings

warnings.filterwarnings("ignore")

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
baseline_location = '/cnrm/recyf/Data/users/danjoul/dataset/baseline/test/'

# ========== Setup
# params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
params = ['t2m']
static_fields = []
dates_train = rangex(['2020070100-2021053100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H', '2022040100-2022043000-PT24H', '2022060100-2022063000-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H', '2022050100-2022053100-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)
working_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/t2m/old/params/t2m/'


# ========== Load Data
results_df = load_results(working_dir, dates_test, echeances, resample, data_test_location, baseline_location, param='t2m')


# ========== Plots
# plot_results(results_df, 't2m', working_dir)
# plot_score_maps(results_df, mae, 'mae', working_dir)
# plot_distrib(results_df_u, mse, 'mse', working_dir)
# plot_distrib(results_df, mae, 'mae', working_dir)
# plot_distrib(results_df, ssim, 'ssim', working_dir)
# plot_datewise_wasserstein_distance_distrib(results_df, working_dir)
# plot_PSDs(results_df, working_dir)
plot_unique_score_map(results_df, mse, 'mse', working_dir)
# plot_cor_len(results_df_u, working_dir)


# ========== Print mean scores
# mse_global_df = datewise_scores(results_df, mse, 'mse') 
# mse_terre_df  = datewise_scores_terre(results_df, mse, 'mse')
# mse_mer_df    = datewise_scores_mer(results_df, mse, 'mse')

# print('mse:')
# print('  global:')
# print('    baseline : ' + str(mse_global_df['mse_baseline_mean'].mean()))
# print('    prediction : ' + str(mse_global_df['mse_y_pred_mean'].mean()))
# print('  terre:')
# print('    baseline : ' + str(mse_terre_df['mse_baseline_mean'].mean()))
# print('    prediction : ' + str(mse_terre_df['mse_y_pred_mean'].mean()))
# print('  mer:')
# print('    baseline : ' + str(mse_mer_df['mse_baseline_mean'].mean()))
# print('    prediction : ' + str(mse_mer_df['mse_y_pred_mean'].mean()))


# ========== Correlation length
# corr_len_df = corr_len(results_df)

# print('correlation lenght :')
# print('baseline : ' + str(corr_len_df.corr_len_baseline[0]))
# print('pred : ' + str(corr_len_df.corr_len_pred[0]))
# print('test : ' + str(corr_len_df.corr_len_test[0]))