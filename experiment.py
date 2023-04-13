import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from make_unet import *
import matplotlib.pyplot as plt
from data import *
from time import perf_counter

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
params = ['t2m', 'toa']
static_fields = []
dates_train = rangex(['2020070100-2021053100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H', '2022040100-2022043000-PT24H', '2022060100-2022063000-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H', '2022050100-2022053100-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)
# output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/unet_4/0.005_32_64/cape/'
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/unet_4/0.005_32_100/toa/'

t1 = perf_counter()
print('setup time = ' + str(t1-t0))


'''
Loading data
'''
X_train = X(dates_train, echeances, data_train_location, data_static_location, params, static_fields=static_fields, resample=resample, missing_days=[])
X_train.load()
y_train = y(dates_train, echeances, data_train_location, data_static_location, params, static_fields=static_fields, missing_days=[])
y_train.load()

X_train.delete_missing_days(y_train)
y_train.delete_missing_days(X_train)
X_train.reshape_4()
y_train.reshape_3()

X_valid = X(dates_valid, echeances, data_valid_location, data_static_location, params, static_fields=static_fields, resample=resample, missing_days=[])
X_valid.load()
y_valid = y(dates_valid, echeances, data_valid_location, data_static_location, params, static_fields=static_fields, missing_days=[])
y_valid.load()

X_valid.delete_missing_days(y_valid)
y_valid.delete_missing_days(X_valid)
X_valid.reshape_4()
y_valid.reshape_3()

X_test = X(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields, resample=resample, missing_days=[])
X_test.load()
y_test = y(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields, missing_days=[])
y_test.load()

X_test.delete_missing_days(y_test)
y_test.delete_missing_days(X_test)
X_test.reshape_4()
y_test.reshape_3()

t2 = perf_counter()
print('loading time = ' + str(t2-t1))


'''
Pre-processing
'''
X_train.pad()
y_train.pad()

X_valid.pad()
y_valid.pad()

X_test.pad()
y_test.pad()

max_X_train, max_y_train = X_train.normalize(), y_train.normalize()
max_X_valid, max_y_valid = X_valid.normalize(), y_valid.normalize()
max_X_test, max_y_test = X_test.normalize(), y_test.normalize()

mean_X_train, std_X_train = X_train.standardize()
mean_y_train, std_y_train = y_train.standardize()
mean_X_valid, std_X_valid = X_valid.standardize()
mean_y_valid, std_y_valid = y_valid.standardize()
mean_X_test, std_X_test = X_test.standardize()
mean_y_test, std_y_test = y_test.standardize()


np.save(output_dir + 'X_test.npy', X_test.X, allow_pickle=True)
np.save(output_dir + 'y_test.npy', y_test.y, allow_pickle=True)

t3 = perf_counter()
print('preprocessing time = ' + str(t3-t2))


'''
Model definition
'''
unet = unet_maker_manu_r(X_train.X[0, :, :, :].shape)
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

history = unet.fit(X_train.X, y_train.y, 
         batch_size=batch_size, epochs=epochs,  
         validation_data=(X_valid,y_valid), 
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
y_pred = y_test.copy()
y_pred.y = unet.predict(X_test.X)
y_pred.y = np.reshape(y_pred.y, (y_pred.y.shape[0], y_pred.y.shape[1], y_pred.y.shape[2]))
np.save(output_dir + 'y_pred_model.npy', y_pred.y, allow_pickle=True)

t5 = perf_counter()
print('predicting time = ' + str(t5-t3))


'''
Post-processing
'''
# Test:
y_test.destandardize(mean_y_test,  std_y_test)
y_test.denormalize(max_y_test)
y_test.crop()
y_test.reshape_4()

X_test.destandardize(mean_X_test, std_X_test)
X_test.denormalize(max_X_test)
X_test.crop()
X_test.reshape_5()

# Pred:
# /!\ indice du paramètre d'intérêt
max_y_pred  = max_X_test[:, 0] 
mean_y_pred,  std_y_pred  = mean_X_test[:, 0],  std_X_test[:, 0]
y_pred.destandardize(mean_y_pred,  std_y_pred)
y_pred.denormalize(max_y_pred)
y_pred.crop()

y_pred.reshape_4()

# y_pred.delete_missing_days(y_test)
# y_pred.delete_missing_days(y_pred)

np.save(output_dir + 'y_pred.npy', y_pred.y, allow_pickle=True)
np.save(output_dir + 'X_test.npy', X_test.X, allow_pickle=True)
np.save(output_dir + 'y_test.npy', y_test.y, allow_pickle=True)

t6 = perf_counter()
print('postprocessing time = ' + str(t6-t5))


'''
Plot Results
'''
# results = Results('t2m', 0, X_test, y_test, y_pred)
# results.plot_20(output_dir)

print('total time = ' + str(t6-t0))