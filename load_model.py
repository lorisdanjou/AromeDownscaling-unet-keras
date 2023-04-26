import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
import matplotlib.pyplot as plt
from data_v2 import *
from make_unet import *
import warnings

warnings.filterwarnings("ignore")


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
params_in = ['t2m']
params_out = ['t2m'] # ! ne fonctionne pas pour 2 sorties (utile pour le vent)
static_fields = []
dates_train = rangex(['2020070100-2021053100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H', '2022040100-2022043000-PT24H', '2022060100-2022063000-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H', '2022050100-2022053100-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/normalisations/normalisation/'
working_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/normalisations/normalisation/'



'''
Load data
'''

data_test = load_data(
    dates_test, 
    echeances,
    params_in,
    params_out,
    data_test_location,
    data_static_location=data_static_location,
    static_fields=static_fields,
    resample=resample
)



'''
Preprocessing
'''
data_test = pad(data_test)
data_test = global_normalisation(data_test, working_dir)




'''
Loading model
'''
unet = unet_maker_manu_r(data_test.X[0][:, :, :].shape)
weights_location = working_dir
unet.load_weights(weights_location + 'weights.97-0.00.hdf5', by_name=False)


'''
Prediction
'''
X_pred = to_array(data_test.X)
y_pred = unet.predict(X_pred)
print(y_pred.shape)
# y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
# np.save(output_dir + 'y_pred_model.npy', y_pred, allow_pickle=True)



'''
Postprocessing
'''
data_pred = data_test.copy()
for i in range(len(data_pred)):
    data_pred.y[i] = y_pred[i, :, :]
data_pred.to_pickle(output_dir + 'data_pred_model.csv')
# data_pred.mean_y = data_pred.mean_X.apply(lambda x: x[0])
# data_pred.std_y = data_pred.std_X.apply(lambda x: x[0])
# data_pred.max_abs_y = data_pred.max_abs_X.apply(lambda x: x[0])
data_pred = global_denormalisation(data_pred, working_dir)
data_pred = crop(data_pred)
y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
print(y_pred.shape)
for i in range(len(data_pred)):
    data_pred.y[i] = data_pred.y[i].reshape((y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))  # ! futurs problèmes avec results_v2 (pas prise en compte nb canaux de sortie)

data_pred.to_pickle(output_dir + 'data_pred.csv')
