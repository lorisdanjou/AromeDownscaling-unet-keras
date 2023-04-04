import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from make_unet import *
import matplotlib.pyplot as plt
import data_loader as dl

model_name = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/unet_4/0.005_32_64/surfgeopotentiel/'

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'


'''
Setup
'''
# params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
params = ["t2m"]
static_fields = ['SURFGEOPOTENTIEL']
dates_train = rangex(['2021010100-2021033100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)


'''
Loading data
'''
data_train = dl.Data(dates_train, echeances, data_train_location, data_static_location, params, static_fields=static_fields)
X_train, y_train = data_train.load_standardized_X_y()
data_valid = dl.Data(dates_valid, echeances, data_valid_location, data_static_location, params, static_fields=static_fields)
X_valid, y_valid = data_valid.load_standardized_X_y()
data_test = dl.Data(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields)
X_test, y_test = data_test.load_standardized_X_y()
np.save(output_dir + 'X_test.npy', X_test, allow_pickle=True)
np.save(output_dir + 'y_test.npy', y_test, allow_pickle=True)


'''
Model definition
'''
# unet=unet_maker_doury( nb_inputs=1,
#                 size_target_domain=get_highestPowerof2_500m()[1], # domaine de sortie = carré ?
#                 shape_inputs=[X_train[0, :, :, :].shape],
#                 filters = 1 )

unet = unet_maker_manu_r(X_train[0, :, :, :].shape)
print('unet created')
LR, batch_size, epochs = 0.005, 32, 64
unet.compile(optimizer=Adam(lr=LR), loss='mse', metrics=[rmse_k])  
print('compilation ok')
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1), ## we set some callbacks to reduce the learning rate during the training
             EarlyStopping(monitor='val_loss', patience=15, verbose=1),               ## Stops the fitting if val_loss does not improve after 15 iterations
             ModelCheckpoint(output_dir + model_name, monitor='val_loss', verbose=1, save_best_only=True)] ## Save only the best model      

## FIt of the EMUL-UNET
history = unet.fit(X_train, y_train, 
         batch_size=batch_size, epochs=epochs,  
         validation_data=(X_valid,y_valid), 
         callbacks = callbacks,
         verbose=2, shuffle=True, validation_split=0.1)
unet.summary()
print(history.history.keys())

#========== Curves ========================================
# summarize history for accuracy
accuracy_curve = plt.figure()
plt.plot(history.history['rmse_k'])
plt.plot(history.history['val_rmse_k'])
plt.title('model rmse_k')
plt.ylabel('rmse_k')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(output_dir + 'RMSE_curve.png')
# summarize history for loss
loss_curve = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(output_dir + 'Loss_curve.png')



'''
Prediction
'''
y_pred = unet.predict(X_test)
np.save(output_dir + 'y_pred.npy', y_pred, allow_pickle=True)

# for i in range(y_pred.shape[0]):

#     fig1, ax1 = plt.subplots()
#     im1 = ax1.imshow(y_pred[i, :, :])
#     fig1.colorbar(im1, ax=ax1)
#     fig1.savefig(output_dir + name_experiment + '/y_' + str(i) + '_pred.png')

#     fig2, ax2 = plt.subplots()
#     im2 = ax2.imshow(y_test[i, :, :])
#     fig2.colorbar(im2, ax=ax2)
#     fig2.savefig(output_dir + name_experiment + '/y_' + str(i) + '_test.png') 