import numpy as np
import pandas as pd
from results_v2 import *
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
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/unet_4/0.005_32_100/synthese_scores/'


'''
Define expes
'''
root_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/unet_4/0.005_32_100/'
expes = pd.DataFrame(
    {'name' : ['t2m','SURFGEOPOTENTIEL', 'uv10', 'toa', 'cape'],
    'dir' : [
    root_dir + 't2m/',
    root_dir + 'SURFGEOPOTENTIEL/',
    root_dir + 'uv10/',
    root_dir + 'toa/',
    root_dir + 'cape/'        
    ]}
)


'''
Graphs
'''
print('maps : ')
# synthesis_maps(expes, output_dir, dates_test, echeances, resample, data_test_location, baseline_location, param, full=True)
print('score maps : ')
synthesis_score_maps(expes, output_dir, mse, 'mse', dates_test, echeances, resample, data_test_location, baseline_location, param)
synthesis_score_maps(expes, output_dir, mae, 'mae', dates_test, echeances, resample, data_test_location, baseline_location, param)
print('distributions : ')
# synthesis_score_distribs(expes, output_dir, mse, 'mse', dates_test, echeances, resample, data_test_location, baseline_location, param)
# synthesis_score_distribs(expes, output_dir, mae, 'mae', dates_test, echeances, resample, data_test_location, baseline_location, param)