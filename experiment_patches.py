import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from make_unet import *
import matplotlib.pyplot as plt
from preprocessing.load_data import *
from preprocessing.normalisations import *
from preprocessing.patches import *
from time import perf_counter
# import warnings

# warnings.filterwarnings("ignore")

physical_devices = tf.config.list_physical_devices()
print(physical_devices)

t0 = perf_counter()

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
baseline_location = '/cnrm/recyf/Data/users/danjoul/dataset/baseline/'
model_name = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'


"""
Setup
"""
# params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
params_in = ['t2m']
params_out = ['t2m']
static_fields = ['SURFIND.TERREMER']
dates_train = rangex(['2020070100-2021053100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H', '2022040100-2022043000-PT24H', '2022060100-2022063000-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H', '2022050100-2022053100-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/patches/tests/'

t1 = perf_counter()
print('setup time = ' + str(t1-t0))


"""
Load Data
"""
X_train_df = load_X(
    dates_train, 
    echeances,
    params_in,
    data_train_location,
    data_static_location,
    static_fields = static_fields,
    resample=resample
)

X_valid_df = load_X(
    dates_valid, 
    echeances,
    params_in,
    data_valid_location,
    data_static_location,
    static_fields = static_fields,
    resample=resample
)

X_test_df = load_X(
    dates_test, 
    echeances,
    params_in,
    data_test_location,
    data_static_location,
    static_fields = static_fields,
    resample=resample
)

y_train_df = load_y(
    dates_train,
    echeances,
    params_out,
    data_train_location
)

y_valid_df = load_y(
    dates_valid,
    echeances,
    params_out,
    data_valid_location
)

y_test_df = load_y(
    dates_test,
    echeances,
    params_out,
    data_test_location
)

t2 = perf_counter()
print('loading time = ' + str(t2-t1))


"""
Preprocessing
"""
# remove missing days
X_train_df, y_train_df = delete_missing_days(X_train_df, y_train_df)
X_valid_df, y_valid_df = delete_missing_days(X_valid_df, y_valid_df)
X_test_df , y_test_df  = delete_missing_days(X_test_df, y_test_df)


# Normalisation:
get_mean(X_train_df, output_dir)
get_std(X_train_df, output_dir)
# X_train_patches_df, y_train_patches_df = standardisation(X_train_patches_df, output_dir), standardisation(y_train_patches_df, output_dir)
# X_valid_patches_df, y_valid_patches_df = standardisation(X_valid_patches_df, output_dir), standardisation(y_valid_patches_df, output_dir)
X_train_df, y_train_df = standardisation(X_train_df, output_dir), standardisation(y_train_df, output_dir)
X_valid_df, y_valid_df = standardisation(X_valid_df, output_dir), standardisation(y_valid_df, output_dir)
X_test_df , y_test_df  = standardisation(X_test_df, output_dir) , standardisation(y_test_df, output_dir)


# patches
patch_size = 128
n_patches  = 4
X_train_patches_df, y_train_patches_df = extract_patches(X_train_df, y_train_df, patch_size, patch_size, n_patches)


# pad data
X_valid_df, y_valid_df  = pad(X_valid_df),  pad(y_valid_df)
X_test_df , y_test_df   = pad(X_test_df) ,  pad(y_test_df)


X_train_patches, y_train_patches = df_to_array(X_train_patches_df), df_to_array(y_train_patches_df)
X_valid, y_valid = df_to_array(X_valid_df), df_to_array(y_valid_df)
X_test , y_test  = df_to_array(X_test_df) , df_to_array(y_test_df)

t3 = perf_counter()
print('preprocessing time = ' + str(t3-t2))


"""
Model definition
"""
unet = unet_maker_manu_r((None, None, X_train_patches.shape[3]))
print('unet creation ok')
      

"""
Training
"""
LR, batch_size, epochs = 0.005, 32, 100
unet.compile(optimizer=Adam(lr=LR), loss='mse', metrics=[rmse_k])  
print('compilation ok')
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1), ## we set some callbacks to reduce the learning rate during the training
             EarlyStopping(monitor='val_loss', patience=15, verbose=1),               ## Stops the fitting if val_loss does not improve after 15 iterations
             ModelCheckpoint(output_dir + model_name, monitor='val_loss', verbose=1, save_best_only=True)] ## Save only the best model

history = unet.fit(X_train_patches, y_train_patches, 
         batch_size=batch_size, epochs=epochs,  
         validation_data=(X_valid, y_valid), 
         callbacks = callbacks,
         verbose=2, shuffle=True)#, validation_split=0.1)

unet.summary()
print(history.history.keys())

# Curves :
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

t4 = perf_counter()
print('training time = ' + str(t4-t3))


"""
Prediction
"""
y_pred = unet.predict(X_test)
print(y_pred.shape)

t5 = perf_counter()
print('predicting time = ' + str(t5-t4))


"""
Postprocessing
"""
y_pred_df = y_test_df.copy()
print(len(y_pred_df))
arrays_cols = get_arrays_cols(y_pred_df)
for i in range(len(y_pred_df)):
    for i_c, c in enumerate(arrays_cols):
        y_pred_df[c][i] = y_pred[i, :, :, i_c]

y_pred_df = destandardisation(y_pred_df, output_dir)
y_pred_df = crop(y_pred_df)

y_pred_df.to_pickle(output_dir + 'y_pred.csv')