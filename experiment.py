
import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from make_unet import *
import matplotlib.pyplot as plt
from data_loader import *

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'


'''
Setup
'''
params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
static_fields = ['SURFGEOPOTENTIEL', 'SURFIND.TERREMER']
dates_train = rangex(['2020070100-2020070500-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022020500-PT24H']) # à modifier
dates_test = rangex(['2022020100-2022020500-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)


'''
Loading data
'''
# X_train, y_train = load_X_y(dates_train, echeances, data_train_location, params, resample=resample)
# X_valid, y_valid = load_X_y(dates_valid, echeances, data_valid_location, params, resample=resample)
X_train, y_train = load_X_y_static(dates_train, echeances, data_train_location, data_static_location, params, resample=resample, static_fields=static_fields)
X_valid, y_valid = load_X_y_static(dates_valid, echeances, data_valid_location, data_static_location, params, resample=resample, static_fields=static_fields)


'''
Model definition
'''
unet=unet_maker( nb_inputs=1,
                size_target_domain=get_size_500m()[1], # domaine de sortie = carré ?
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
X_test, y_test = load_X_y_static(dates_test, echeances, data_test_location, data_static_location, params, resample=resample, static_fields=static_fields)

y_pred = unet.predict(X_test)

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(y_pred[0, :, :])
fig1.colorbar(im1, ax=ax1)
fig1.savefig('./y_pred.png')

fig2, ax2 = plt.subplots()
im2 = ax2.imshow(y_test[0, :, :])
fig2.colorbar(im2, ax=ax2)
fig2.savefig('./y_test.png') 