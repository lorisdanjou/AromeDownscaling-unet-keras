import numpy as np
from os.path import exists
from data import *
from results import *
from bronx.stdtypes.date import daterangex as rangex
import matplotlib.pyplot as plt
from matplotlib import colors


data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
baseline_location = '/cnrm/recyf/Data/users/danjoul/dataset/baseline/test/'
model_name = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

'''
Setup
'''
# params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
params = ['t2m']
static_fields = []
dates_train = rangex(['2020070100-2021053100-PT24H']) # à modifier
dates_valid = rangex(['2022020100-2022022800-PT24H', '2022040100-2022043000-PT24H', '2022060100-2022063000-PT24H']) # à modifier
dates_test = rangex(['2022030100-2022033100-PT24H', '2022050100-2022053100-PT24H']) # à modifier
resample = 'r'
echeances = range(6, 37, 3)
echeances_baseline = range(6, 37, 3)
working_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/unet_4/0.005_32_100/t2m/'


y_pred = y(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields)
y_test = y(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields)
X_test = X(dates_test, echeances, data_test_location, data_static_location, params, static_fields=static_fields, resample=resample)
baseline = y(dates_test, echeances_baseline, baseline_location, data_static_location, params, base=True)
X_test.load()
y_test.load()
y_pred.y = np.load(working_dir + 'y_pred.npy', allow_pickle=True)
baseline.load()

y_test.delete_missing_days(baseline)
y_test.delete_missing_days(y_pred)
y_pred.delete_missing_days(baseline)
y_pred.delete_missing_days(y_test)
baseline.delete_missing_days(y_pred)
baseline.delete_missing_days(y_test)


# print('y_pred shape : ' + str(y_pred.y.shape))
# print('y_test shape : ' + str(y_test.y.shape))
# print('baseline shape : ' + str(baseline.y.shape))
# print('X_test shape : ' + str(X_test.X.shape))

results = Results('t2m', 0, X_test, y_test, y_pred, baseline)
# results.plot_firsts(working_dir, base=True)

mse_baseline_matrix, mse_pred_matrix, mse_baseline_global, mse_pred_global = results.mse_global()
mse_baseline_terre_matrix, mse_pred_terre_matrix, mse_baseline_terre_global, mse_pred_terre_global = results.mse_terre()
mse_baseline_mer_matrix, mse_pred_mer_matrix, mse_baseline_mer_global, mse_pred_mer_global = results.mse_mer()

print('rmse:')
print('  global:')
print('    baseline : ' + str(np.sqrt(np.mean(mse_baseline_global))))
print('    prediction : ' + str(np.sqrt(np.mean(mse_pred_global))))
print('  terre:')
print('    baseline : ' + str(np.sqrt(np.mean(mse_baseline_terre_global))))
print('    prediction : ' + str(np.sqrt(np.mean(mse_pred_terre_global))))
print('  mer:')
print('    baseline : ' + str(np.sqrt(np.mean(mse_baseline_mer_global))))
print('    prediction : ' + str(np.sqrt(np.mean(mse_pred_mer_global))))


results.plot_distrib_rmse(working_dir)

# fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (21, 15))
# images = []
# data = [mse_baseline_matrix[0,0,:,:], mse_baseline_terre_matrix[0,0,:,:], mse_baseline_mer_matrix[0,0,:,:]]
# for i in range(len(data)):
#     images.append(axs[i].imshow(data[i], cmap='Blues'))
#     axs[i].label_outer()
# vmin = min(image.get_array().min() for image in images)
# vmax = max(image.get_array().max() for image in images)
# norm = colors.Normalize(vmin=vmin, vmax=vmax)
# for im in images:
#     im.set_norm(norm)
# axs[0].set_title('baseline global')
# axs[1].set_title('baseline terre')
# axs[2].set_title('baseline mer')
# fig.colorbar(images[0], ax=axs)
# plt.savefig(working_dir + 'mse_baseline.png')


# fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (21, 7))
# images = []
# data = [mse_pred_matrix[0,0,:,:], mse_pred_terre_matrix[0,0,:,:], mse_pred_mer_matrix[0,0,:,:]]
# for i in range(len(data)):
#     images.append(axs[i].imshow(data[i], cmap='Blues'))
#     axs[i].label_outer()
# vmin = min(image.get_array().min() for image in images)
# vmax = max(image.get_array().max() for image in images)
# norm = colors.Normalize(vmin=vmin, vmax=vmax)
# for im in images:
#     im.set_norm(norm)
# axs[0].set_title('pred global')
# axs[1].set_title('pred terre')
# axs[2].set_title('pred mer')
# fig.colorbar(images[0], ax=axs)
# plt.savefig(working_dir + 'mse_pred.png')


fig, axs = plt.subplots(nrows=2,ncols=3, figsize = (21, 15))
images = []
data = np.array([[mse_baseline_matrix[0,0,:,:], mse_baseline_terre_matrix[0,0,:,:], mse_baseline_mer_matrix[0,0,:,:]],
        [mse_pred_matrix[0,0,:,:], mse_pred_terre_matrix[0,0,:,:], mse_pred_mer_matrix[0,0,:,:]]])
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        images.append(axs[i, j].imshow(data[i, j], cmap='Blues'))
        axs[i, j].label_outer()
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)
axs[0,0].set_title('baseline global')
axs[0,1].set_title('baseline terre')
axs[0,2].set_title('baseline mer')
axs[1,0].set_title('pred global')
axs[1,1].set_title('pred terre')
axs[1,2].set_title('pred mer')
fig.colorbar(images[0], ax=axs)
plt.savefig(working_dir + 'mse.png')

