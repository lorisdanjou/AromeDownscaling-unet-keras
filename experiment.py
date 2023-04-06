import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from make_unet import *
import matplotlib.pyplot as plt
from data_loader import Data
from results import Results


data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
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
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/unet_4/0.005_32_64/cape/'


'''
Loading data
'''
data_train = Data(dates_train, echeances, data_train_location, data_static_location, params, static_fields=static_fields)
data_valid = Data(dates_valid, echeances, data_valid_location, data_static_location, params, static_fields=static_fields)
data_test = Data(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields)


'''
Pre-processing
'''
data_train.pad_X_y()
data_valid.pad_X_y()
data_test.pad_X_y()

max_X_train, max_y_train = data_train.normalize_X_y()
max_X_valid, max_y_valid = data_valid.normalize_X_y()
max_X_test,  max_y_test  = data_test.normalize_X_y()

mean_X_train, std_X_train, mean_y_train, std_y_train = data_train.standardize_X_y()
mean_X_valid, std_X_valid, mean_y_valid, std_y_valid = data_valid.standardize_X_y()
mean_X_test,  std_X_test,  mean_y_test,  std_y_test  = data_test.standardize_X_y()

X_train, y_train = data_train.X, data_train.y
X_valid, y_valid = data_valid.X, data_valid.y
X_test , y_test  = data_test.X,  data_test.y

np.save(output_dir + 'X_test.npy', X_test, allow_pickle=True)
np.save(output_dir + 'y_test.npy', y_test, allow_pickle=True)


'''
Model definition
'''
unet = unet_maker_manu_r(X_train[0, :, :, :].shape)
print('unet creation ok')
      

'''
Training
'''
LR, batch_size, epochs = 0.005, 32, 64
unet.compile(optimizer=Adam(lr=LR), loss='mse', metrics=[rmse_k])  
print('compilation ok')
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1), ## we set some callbacks to reduce the learning rate during the training
             EarlyStopping(monitor='val_loss', patience=15, verbose=1),               ## Stops the fitting if val_loss does not improve after 15 iterations
             ModelCheckpoint(output_dir + model_name, monitor='val_loss', verbose=1, save_best_only=True)] ## Save only the best model

history = unet.fit(X_train, y_train, 
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


'''
Prediction
'''
data_pred = data_test.copy()
y_pred = unet.predict(X_test)
y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
data_pred.y = y_pred
np.save(output_dir + 'y_pred_model.npy', y_pred, allow_pickle=True)


'''
Post-processing
'''
# Test:
data_test.destandardize_X_y(mean_X_test,  std_X_test,  mean_y_test,  std_y_test)
data_test.denormalize_X_y(max_X_test,  max_y_test)
data_test.crop_X_y()
X_test, y_test = data_test.X, data_test.y

# Pred:
# /!\ indice du paramètre d'intérêt
max_X_pred,  max_y_pred  = max_X_test, max_X_test[:, 0] 
mean_X_pred,  std_X_pred,  mean_y_pred,  std_y_pred  = mean_X_test,  std_X_test,  mean_X_test[:, 0],  std_X_test[:, 0]
data_pred.destandardize_X_y(mean_X_pred,  std_X_pred,  mean_y_pred,  std_y_pred)
data_pred.denormalize_X_y(max_X_pred,  max_y_pred)
data_pred.crop_X_y()
X_pred, y_pred = data_pred.X, data_pred.y

np.save(output_dir + 'y_pred.npy', y_pred, allow_pickle=True)
np.save(output_dir + 'X_test.npy', X_test, allow_pickle=True)
np.save(output_dir + 'y_test.npy', y_test, allow_pickle=True)


'''
Plot Results
'''
results = Results('t2m', 0, X_test, y_test, y_pred)
results.plot_20(output_dir)