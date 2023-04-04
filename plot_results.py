import numpy as np
from results import Results

X_test = np.load('/cnrm/recyf/Data/users/danjoul/unet_experiments/X_test.npy')
y_test = np.load('/cnrm/recyf/Data/users/danjoul/unet_experiments/y_test.npy') 
y_pred = np.load('/cnrm/recyf/Data/users/danjoul/unet_experiments/y_pred.npy')
output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/'

results = Results('t2m', X_test, y_test, y_pred)
results.plot_all(output_dir)
