import numpy as np 
import random as rn
from make_unet import *
import matplotlib.pyplot as plt
from data_loader import *

name_experiment = ''
model_name = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/'

X_train = np.load('/cnrm/recyf/Data/users/danjoul/unet_experiments/X_train.npy')
X_valid = np.load('/cnrm/recyf/Data/users/danjoul/unet_experiments/X_valid.npy')
X_test  = np.load('/cnrm/recyf/Data/users/danjoul/unet_experiments/X_test.npy')
y_train = np.load('/cnrm/recyf/Data/users/danjoul/unet_experiments/y_train.npy')
y_valid = np.load('/cnrm/recyf/Data/users/danjoul/unet_experiments/y_valid.npy')
y_test  = np.load('/cnrm/recyf/Data/users/danjoul/unet_experiments/y_test.npy')


'''
Model definition
'''
# unet=unet_maker_doury( nb_inputs=1,
#                 size_target_domain=get_highestPowerof2_500m()[1], # domaine de sortie = carré ?
#                 shape_inputs=[X_train[0, :, :, :].shape],
#                 filters = 1 )

unet = unet_maker_manu_r(X_train[0, :, :, :].shape)
print('unet created')
LR, batch_size, epochs = 0.001, 16, 64
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
plt.savefig(output_dir + name_experiment + '/RMSE_curve.png')
# summarize history for loss
loss_curve = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(output_dir + name_experiment + '/Loss_curve.png')



'''
Prediction
'''
y_pred = unet.predict(X_test)

for i in range(0, 30, 3):

    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(y_pred[i, :, :])
    fig1.colorbar(im1, ax=ax1)
    fig1.savefig(output_dir + 'y_' + str(i) + '_pred.png')

    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(y_test[i, :, :])
    fig2.colorbar(im2, ax=ax2)
    fig2.savefig(output_dir + 'y_' + str(i) + '_test.png') 