
import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from make_unet import *
import matplotlib.pyplot as plt

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'


'''
Setup
'''
params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
dates_train = rangex(['2020070100-2020070500-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022020500-PT24H']) # à modifier
echeances = range(6, 37, 3)
field_500m = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/G9KP_2021-01-01T00:00:00Z_rr.npy')
field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_c_2021-01-01T00:00:00Z_rr.npy')
geometry_500m = field_500m[:,:,0].shape
size_500m = min(geometry_500m)
size_500m_crop = highestPowerof2(min(geometry_500m))
print(size_500m, size_500m_crop)
geometry_2km5 = field_2km5[:,:,0].shape
size_2km5 = min(geometry_2km5)
size_2km5_crop = highestPowerof2(min(geometry_2km5))


'''
Loading data
'''
# initial shape of the data:
    # X[date, ech, x, y, param]
    # y[date, ech, x, y]
X_train = np.zeros(shape=[len(dates_train), len(echeances), geometry_2km5[0], geometry_2km5[1], len(params)], dtype=np.float32)
y_train = np.zeros(shape=[len(dates_train), len(echeances), geometry_500m[0], geometry_500m[1]], dtype=np.float32)
X_valid = np.zeros(shape=[len(dates_valid), len(echeances), geometry_2km5[0], geometry_2km5[1], len(params)], dtype=np.float32)
y_valid = np.zeros(shape=[len(dates_valid), len(echeances), geometry_500m[0], geometry_500m[1]], dtype=np.float32)

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

for i_d, d in enumerate(dates_valid):
    filepath_y_valid = data_valid_location + 'G9L1_' + d.isoformat() + 'Z_t2m.npy'
    y_valid[i_d, :, :, :] = np.load(filepath_y_valid).transpose([2,0,1])
    for i_p, p in enumerate(params):
        filepath_X_valid = data_valid_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
        X_valid[i_d, :, :, :, i_p] = np.load(filepath_X_valid).transpose([2,0,1])

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)

# final shape of the data : 
    # X[date/ech, x, y, param]
    # y[date/ech, x, y]
X_train = X_train.reshape((-1, geometry_2km5[0], geometry_2km5[1], len(params)))
y_train = y_train.reshape((-1, geometry_500m[0], geometry_500m[1]))
X_valid = X_valid.reshape((-1, geometry_2km5[0], geometry_2km5[1], len(params)))
y_valid = y_valid.reshape((-1, geometry_500m[0], geometry_500m[1]))

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)

X_train = X_train[:, int(0.5* (geometry_2km5[0] - size_2km5_crop)):int(0.5* (geometry_2km5[0] + size_2km5_crop)), int(0.5* (geometry_2km5[1] - size_2km5_crop)):int(0.5* (geometry_2km5[1] + size_2km5_crop)), :]
y_train = y_train[:, int(0.5* (geometry_500m[0] - size_500m_crop)):int(0.5* (geometry_500m[0] + size_500m_crop)), int(0.5* (geometry_500m[1] - size_500m_crop)):int(0.5* (geometry_500m[1] + size_500m_crop))]
X_valid = X_valid[:, int(0.5* (geometry_2km5[0] - size_2km5_crop)):int(0.5* (geometry_2km5[0] + size_2km5_crop)), int(0.5* (geometry_2km5[1] - size_2km5_crop)):int(0.5* (geometry_2km5[1] + size_2km5_crop)), :]
y_valid = y_valid[:, int(0.5* (geometry_500m[0] - size_500m_crop)):int(0.5* (geometry_500m[0] + size_500m_crop)), int(0.5* (geometry_500m[1] - size_500m_crop)):int(0.5* (geometry_500m[1] + size_500m_crop))]

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)


'''
Model definition
'''
unet=unet_maker( nb_inputs=1,
                size_target_domain=size_500m_crop, # domaine de sortie = carré ?
                shape_inputs=[X_train[0, :, :, :].shape],
                filters = 1 )
LR, batch_size, epochs = 0.005, 32, 100
unet.compile(optimizer=Adam(lr=LR), loss='mse', metrics=[rmse_k])  
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1), ## we set some callbacks to reduce the learning rate during the training
             EarlyStopping(monitor='val_loss', patience=15, verbose=1)]                ## Stops the fitting if val_loss does not improve after 15 iterations
             #ModelCheckpoint(path_model, monitor='val_loss', verbose=1, save_best_only=True)] ## Save only the best model
                                # à définir         

## FIt of the EMUL-UNET
unet.fit(X_train, y_train, 
         batch_size=batch_size, epochs=epochs,  
         validation_data=(X_valid,y_valid), 
         callbacks = callbacks)

'''
Prediction
'''
dates_test = rangex(['2022020100-2022020500-PT24H']) # à modifier
X_test = np.zeros(shape=[len(dates_test), len(echeances), geometry_2km5[0], geometry_2km5[1], len(params)], dtype=np.float32)
y_test = np.zeros(shape=[len(dates_test), len(echeances), geometry_500m[0], geometry_500m[1]], dtype=np.float32)

for i_d, d in enumerate(dates_test):
    filepath_y_test = data_valid_location + 'G9L1_' + d.isoformat() + 'Z_t2m.npy'
    y_test[i_d, :, :, :] = np.load(filepath_y_test).transpose([2,0,1])
    for i_p, p in enumerate(params):
        filepath_X_test = data_valid_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
        X_test[i_d, :, :, :, i_p] = np.load(filepath_X_test).transpose([2,0,1])

X_test = X_test.reshape((-1, geometry_2km5[0], geometry_2km5[1], len(params)))
y_test = y_test.reshape((-1, geometry_500m[0], geometry_500m[1]))

X_test = X_test[:, int(0.5* (geometry_2km5[0] - size_2km5_crop)):int(0.5* (geometry_2km5[0] + size_2km5_crop)), int(0.5* (geometry_2km5[1] - size_2km5_crop)):int(0.5* (geometry_2km5[1] + size_2km5_crop)), :]
y_test = y_test[:, int(0.5* (geometry_500m[0] - size_500m_crop)):int(0.5* (geometry_500m[0] + size_500m_crop)), int(0.5* (geometry_500m[1] - size_500m_crop)):int(0.5* (geometry_500m[1] + size_500m_crop))]

print(X_test.shape)
print(y_test.shape)

y_pred = unet.predict(X_test)

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(y_pred[0, :, :])
fig1.colorbar(im1, ax=ax1)
fig1.savefig('./y_pred.png')

fig2, ax2 = plt.subplots()
im2 = ax2.imshow(y_test[0, :, :])
fig2.colorbar(im2, ax=ax2)
fig2.savefig('./y_test.png') 