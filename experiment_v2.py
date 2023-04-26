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

t1 = perf_counter()
print('setup time = ' + str(t1-t0))


'''
Load data
'''
data_train = load_data(
    dates_train, 
    echeances,
    params_in,
    params_out,
    data_train_location,
    data_static_location=data_static_location,
    static_fields=static_fields,
    resample=resample
)

data_valid = load_data(
    dates_valid, 
    echeances,
    params_in,
    params_out,
    data_valid_location,
    data_static_location=data_static_location,
    static_fields=static_fields,
    resample=resample
)

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

t2 = perf_counter()
print('loading time = ' + str(t2-t1))


'''
Preprocessing
'''
data_train = pad(data_train)
data_valid = pad(data_valid)
data_test = pad(data_test)

get_max_abs(data_train, working_dir)

data_train = normalisation(data_train, working_dir)
data_valid = normalisation(data_valid, working_dir)
data_test  = normalisation(data_test, working_dir)

X_train = to_array(data_train.X)
y_train = to_array(data_train.y)
X_valid = to_array(data_valid.X)
y_valid = to_array(data_valid.y)

t3 = perf_counter()
print('preprocessing time = ' + str(t3-t2))



'''
Model definition
'''
unet = unet_maker_manu_r(data_train.X[0][:, :, :].shape)
print('unet creation ok')
      

'''
Training
'''
LR, batch_size, epochs = 0.005, 32, 100
unet.compile(optimizer=Adam(lr=LR), loss='mse', metrics=[rmse_k])  
print('compilation ok')
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1), ## we set some callbacks to reduce the learning rate during the training
             EarlyStopping(monitor='val_loss', patience=15, verbose=1),               ## Stops the fitting if val_loss does not improve after 15 iterations
             ModelCheckpoint(output_dir + model_name, monitor='val_loss', verbose=1, save_best_only=True)] ## Save only the best model

history = unet.fit(X_train, y_train, 
         batch_size=batch_size, epochs=epochs,  
         validation_data=(X_valid, y_valid), 
         callbacks = callbacks,
         verbose=2, shuffle=True, validation_split=0.1)

unet.summary()
print(history.history.keys())

# Curves :
# summarize history for accuracy
accuracy_curve = plt.figure()
plt.plot(history.history['rmse_k'])
plt.plot(history.history['val_rmse_k'])
plt.semilogy()
plt.title('model rmse_k')
plt.ylabel('rmse_k')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(output_dir + 'RMSE_curve.png')
# summarize history for loss
loss_curve = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.semilogy()
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(output_dir + 'Loss_curve.png')

t4 = perf_counter()
print('training time = ' + str(t4-t3))


'''
Prediction
'''
X_pred = to_array(data_test.X)
y_pred = unet.predict(X_pred)
print(y_pred.shape)
y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
# np.save(output_dir + 'y_pred_model.npy', y_pred, allow_pickle=True)

t5 = perf_counter()
print('predicting time = ' + str(t5-t3))


'''
Postprocessing
'''
data_pred = data_test.copy()
for i in range(len(data_pred)):
    data_pred.y[i] = y_pred[i, :, :, :]
data_pred.to_pickle(output_dir + 'data_pred_model.csv')
data_pred = denormalisation(data_pred, working_dir)
data_pred = crop(data_pred)

data_pred.to_pickle(output_dir + 'data_pred.csv')

# y_pred = to_array(data_pred.y)
# y_pred = y_pred.reshape([data_pred.dates.nunique(), data_pred.echeances.nunique(), y_pred.shape[1], y_pred.shape[2]])
# np.save(output_dir + 'y_pred.npy', y_pred, allow_pickle=True)

# data_test = destandardisation(data_test)
# data_test = crop(data_test) 


