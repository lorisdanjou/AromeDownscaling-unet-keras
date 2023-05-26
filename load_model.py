import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from unet.architectures import *
from training.imports4training import *
from training.generator import DataGenerator
import matplotlib.pyplot as plt
from preprocessing.load_data import *
from preprocessing.normalisations import *
from time import perf_counter
# import warnings
# warnings.filterwarnings("ignore")

t0 = perf_counter()

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
baseline_location = '/cnrm/recyf/Data/users/danjoul/dataset/baseline/'
model_name = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'


# ========== Setup
params_in = ['u10', 'v10']
params_out = ['u10', 'v10']
static_fields = []

dates_train = rangex([
    '2020070100-2021053100-PT24H'
])
dates_valid = rangex([
    '2021080100-2021083100-PT24H',
    '2021100100-2021103100-PT24H',
    '2021100100-2021123100-PT24H',
    '2022020100-2022022800-PT24H',
    '2022040100-2022043000-PT24H',
    '2022060100-2022063000-PT24H'
])
dates_test = rangex([
    '2021070100-2021073100-PT24H',
    '2021090100-2021093000-PT24H',
    '2021110100-2021113000-PT24H',
    '2022030100-2022033100-PT24H',
    '2022050100-2022053100-PT24H'
])
resample = 'r'
echeances = range(6, 37, 3)
LR, batch_size, epochs = 0.005, 32, 100
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/tests/'

t1 = perf_counter()
print('setup time = ' + str(t1-t0))


# ========== Load Data
# X_train_df = load_X(
#     dates_train, 
#     echeances,
#     params_in,
#     data_train_location,
#     data_static_location,
#     static_fields = static_fields,
#     resample=resample
# )

# X_valid_df = load_X(
#     dates_valid, 
#     echeances,
#     params_in,
#     data_valid_location,
#     data_static_location,
#     static_fields = static_fields,
#     resample=resample
# )

X_test_df = load_X(
    dates_test, 
    echeances,
    params_in,
    data_test_location,
    data_static_location,
    static_fields = static_fields,
    resample=resample
)

# y_train_df = load_y(
#     dates_train,
#     echeances,
#     params_out,
#     data_train_location
# )

# y_valid_df = load_y(
#     dates_valid,
#     echeances,
#     params_out,
#     data_valid_location
# )

y_test_df = load_y(
    dates_test,
    echeances,
    params_out,
    data_test_location
)

t2 = perf_counter()
print('loading time = ' + str(t2-t1))


# ========== Preprocessing
# remove missing days
# X_train_df, y_train_df = delete_missing_days(X_train_df, y_train_df)
# X_valid_df, y_valid_df = delete_missing_days(X_valid_df, y_valid_df)
X_test_df, y_test_df = delete_missing_days(X_test_df, y_test_df)

# pad data
# X_train_df, y_train_df = pad(X_train_df), pad(y_train_df)
# X_valid_df, y_valid_df = pad(X_valid_df), pad(y_valid_df)
X_test_df, y_test_df = pad(X_test_df), pad(y_test_df)

# Normalisation:
# get_mean(y_train_df, output_dir)
# get_std(y_train_df, output_dir)
# X_train_df, y_train_df = standardisation(X_train_df, output_dir), standardisation(y_train_df, output_dir)
# X_valid_df, y_valid_df = standardisation(X_valid_df, output_dir), standardisation(y_valid_df, output_dir)
X_test_df , y_test_df  = standardisation(X_test_df, output_dir) , standardisation(y_test_df, output_dir)


# mean_train, std_train = get_mean_df(X_train_df), get_std_df(X_train_df)
# mean_valid, std_valid = get_mean_df(X_valid_df), get_std_df(X_valid_df)
# mean_test , std_test  = get_mean_df(X_test_df) , get_std_df(X_test_df)
# X_train_df = standardisation_df(X_train_df, mean_train, std_train)
# X_valid_df = standardisation_df(X_valid_df, mean_valid, std_valid)
# X_test_df  = standardisation_df(X_test_df, mean_test, std_test)

# X_train_df = standardisation_sample(X_train_df)
# X_valid_df = standardisation_sample(X_valid_df)
# X_test_df = standardisation_sample(X_test_df)

# X_train, y_train = df_to_array(X_train_df), df_to_array(y_train_df)
# X_valid, y_valid = df_to_array(X_valid_df), df_to_array(y_valid_df)
X_test, y_test = df_to_array(X_test_df), df_to_array(y_test_df)


t3 = perf_counter()
print('preprocessing time = ' + str(t3-t2))

# ========== Load Model
unet = ResUNet_maker(X_test[0, :, :, :].shape, output_channels=len(params_out))
weights_location = output_dir
unet.load_weights(weights_location + 'weights.14-0.17.hdf5', by_name=False)
unet.summary()


# ========== Prediction
y_pred = unet.predict(X_test)
print(X_test.shape)
print(y_pred.shape)

t5 = perf_counter()
print('predicting time = ' + str(t5-t3))


# ========== Postprocessing
y_pred_df = y_test_df.copy()
print(len(y_pred_df))
arrays_cols = get_arrays_cols(y_pred_df)
for i in range(len(y_pred_df)):
    for i_c, c in enumerate(arrays_cols):
        y_pred_df[c][i] = y_pred[i, :, :, i_c]

y_pred_df = destandardisation(y_pred_df, output_dir)
y_pred_df = crop(y_pred_df)

y_pred_df.to_pickle(output_dir + 'y_pred.csv')
