import numpy as np
from results import Results

output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/unet_4_0.005_32_64_jan_mar_standardized/'
X_test = np.load(output_dir + 'X_test.npy')
y_test = np.load(output_dir + 'y_test.npy') 
y_pred = np.load(output_dir + 'y_pred.npy')

results = Results('t2m', 0, X_test, y_test, y_pred)
results.plot_all(output_dir)
