import numpy as np
import pandas as pd
from results import *
from synthesis import *
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
params = ['t2m']
static_fields = []
dates_train = rangex(['2020070100-2021053100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H', '2022040100-2022043000-PT24H', '2022060100-2022063000-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H', '2022050100-2022053100-PT24H']) # à modifier
resample = 'r'
param = 't2m'
echeances = range(6, 37, 3)
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/interp/'


'''
Define expes
'''
expes_names = ['nearest', 'linear', 'cubic']
expes_results = [
    load_results(output_dir + 'nearest/', dates_test, echeances, 'r', data_test_location, baseline_location, param=param),
    load_results(output_dir + 'bl/', dates_test, echeances, 'bl', data_test_location, baseline_location, param=param),
    load_results(output_dir + 'bc/', dates_test, echeances, 'bc', data_test_location, baseline_location, param=param)
]
# expes_names = ['mae', 'mse', 'huber']
# expes_results = [
#     load_results(output_dir + 'mae/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + 'mse/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + 'huber/', dates_test, echeances, resample, data_test_location, baseline_location, param=param)
# ]
# expes_names = ['16', '32', '64']
# expes_results = [
#     load_results(output_dir + '16/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + '32/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + '64/', dates_test, echeances, resample, data_test_location, baseline_location, param=param)
# ]
# expes_names = ['t2m', 'toa', 'ts', 'tke', 'cape', 'uv10', 'SURFGEOPOTENTIEL', 'SURFIND.TERREMER', 'SFX.BATHY']
# expes_results = [
#     load_results(output_dir + 't2m/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + 'toa/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + 'ts/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + 'tke/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + 'cape/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + 'uv10/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + 'SURFGEOPOTENTIEL/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + 'SURFIND.TERREMER/', dates_test, echeances, resample, data_test_location, baseline_location, param=param),
#     load_results(output_dir + 'SFX.BATHY/', dates_test, echeances, resample, data_test_location, baseline_location, param=param)
# ]


'''
Graphs
'''
# print('maps')
# synthesis_maps(expes_names, expes_results, output_dir, full=True)
# print('score maps')
# synthesis_score_maps(expes_names, expes_results, output_dir, mse, 'mse')
# synthesis_score_maps(expes_names, expes_results, output_dir, mae, 'mae')
# synthesis_score_maps(expes_names, expes_results, output_dir, biais, 'biais')
# print('distributions')
# synthesis_score_distribs(expes_names, expes_results, output_dir, mse, 'mse')
# synthesis_score_distribs(expes_names, expes_results, output_dir, mae, 'mae')
# print('wasserstein')
# synthesis_wasserstein_distance_distrib(expes_names, expes_results, output_dir)
# print('PSDs')
# synthesis_PSDs(expes_names, expes_results, output_dir)
print('corr_len')
synthesis_corr_len(expes_names, expes_results, output_dir)