
import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from make_unet import *

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'


'''
Setup
'''
params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
dates_train = rangex(['2020070100-2020070500-PT24H']) # à modifier
dates_test = rangex(['2022020100-2022020500-PT24H']) # à modifier
echeances = range(6, 37, 3)
field_500m = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/G9KP_2021-01-01T00:00:00Z_rr.npy')
field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_c_2021-01-01T00:00:00Z_rr.npy')
geometry_500m = field_500m.shape
geometry_2km5 = field_2km5.shape


'''
Loading data
'''
# initial shape of the data:
    # X[date, ech, x, y, param]
    # y[date, ech, x, y]
X_train = np.zeros(shape=[len(dates_train), len(echeances), geometry_2km5[0], geometry_2km5[1], len(params)], dtype=np.float32)
y_train = np.zeros(shape=[len(dates_train), len(echeances), geometry_500m[0], geometry_500m[1]], dtype=np.float32)
X_test = np.zeros(shape=[len(dates_test), len(echeances), geometry_2km5[0], geometry_2km5[1], len(params)], dtype=np.float32)
y_test = np.zeros(shape=[len(dates_test), len(echeances), geometry_500m[0], geometry_500m[1]], dtype=np.float32)

for i_d, d in enumerate(dates_train):
    try:
        filepath_y_train = data_train_location + 'G9L1_' + d.isoformat() + 'Z_t2m.npy'
        y_train[i_d, :, :, :] = np.load(filepath_y_train).transpose([2,0,1])
    except :
        filepath_y_train = data_train_location + 'G9KP_' + d.isoformat() + 'Z_t2m.npy'
        y_train[i_d, :, :, :] = np.load(filepath_y_train).transpose([2,0,1])
    for i_p, p in enumerate(params):
        filepath_X_train = data_train_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
        X_train[i_d, :, :, :, i_p] = np.load(filepath_X_train).transpose([2,0,1])

for i_d, d in enumerate(dates_test):
    filepath_y_test = data_test_location + 'G9L1_' + d.isoformat() + 'Z_t2m.npy'
    y_test[i_d, :, :, :] = np.load(filepath_y_test).transpose([2,0,1])
    for i_p, p in enumerate(params):
        filepath_X_test = data_test_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
        X_test[i_d, :, :, :, i_p] = np.load(filepath_X_test).transpose([2,0,1])

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# final shape of the data : 
    # X[date/ech, x, y, param]
    # y[date/ech, x, y]
X_train = X_train.reshape((-1, geometry_2km5[0], geometry_2km5[1], len(params)))
y_train = y_train.reshape((-1, geometry_500m[0], geometry_500m[1]))
X_test = X_test.reshape((-1, geometry_2km5[0], geometry_2km5[1], len(params)))
y_test = y_test.reshape((-1, geometry_500m[0], geometry_500m[1]))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


'''
Model definition
'''
unet=unet_maker( nb_inputs=1,
                size_target_domain=geometry_500m.min, # domaine de sortie = carré ?
                shape_inputs=[X_train[0, :, :, :].shape],
                filters = 1 )
LR, batch_size, epochs = 0.005, 32, 100
unet.compile(optimizer=Adam(lr=LR), loss='mse', metrics=[rmse_k])  
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1), ## we set some callbacks to reduce the learning rate during the training
             EarlyStopping(monitor='val_loss', patience=15, verbose=1)]                ## Stops the fitting if val_loss does not improve after 15 iterations
             #ModelCheckpoint(path_model, monitor='val_loss', verbose=1, save_best_only=True)] ## Save only the best model
                                # à définir         

## FIt of the EMUL-UNET
unet.fit(X_train, y_train , 
         batch_size=batch_size, epochs=epochs,  
         validation_data=(X_test,y_test), 
         callbacks = callbacks)

