import numpy as np
import matplotlib.pyplot as plt
from random import *
from data_manager import *

def get_ind_terre_mer_500m():
    filepath = '/cnrm/recyf/Data/users/danjoul/dataset/static_G9KP_SURFIND.TERREMER.npy'
    return np.load(filepath)


def get_indices_baseline(size_y_pred):
    indices = []
    nb_days = size_y_pred//11
    for i_d in range(nb_days):
        for i_ech in range(0, 11, 2):
            indices.append(11*i_d + i_ech)
    return indices



class Results():

    def __init__(self, p, i_p, data_X_test, data_y_test, data_y_pred, data_baseline): # /!\ jours chargement de la baseline
        self.p = p
        self.i_p = i_p
        self.data_X_test = data_X_test.copy()
        self.data_y_test = data_y_test.copy()
        self.data_y_pred = data_y_pred.copy()
        self.data_baseline = data_baseline.copy()

        X_test = np.reshape(data_X_test.X, (len(data_X_test.dates), len(data_X_test.echeances), data_X_test.X.shape[1], data_X_test.X.shape[2], data_X_test.X.shape[3]))
        y_test = np.reshape(data_y_test.y, (len(data_y_test.dates), len(data_y_test.echeances), data_y_test.y.shape[1], data_y_test.y.shape[2]))
        y_pred = np.reshape(data_y_pred.y, (len(data_y_pred.dates), len(data_y_pred.echeances), data_y_pred.y.shape[1], data_y_pred.y.shape[2]))
        baseline = np.reshape(data_baseline.baseline, (len(data_baseline.dates), len(data_baseline.echeances), data_baseline.baseline.shape[1], data_baseline.baseline.shape[2]))

        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.baseline = baseline


    def plot_d_ech(self, i_d, i_ech, output_dir, base=False):
        if base:
            fig, axs = plt.subplots(nrows=1,ncols=4, figsize = (28, 7))
            im = axs[0].imshow(self.X_test[i_d, 2*i_ech, :, :, self.i_p])
            im = axs[1].imshow(self.baseline[i_d, i_ech, :, :])
            im = axs[2].imshow(self.y_pred[i_d, 2*i_ech, :, :])
            im = axs[3].imshow(self.y_test[i_d, 2*i_ech, :, :])

            axs[0].set_title('X_test')
            axs[1].set_title('baseline')
            axs[2].set_title('y_pred')
            axs[3].set_title('y_test')

            fig.colorbar(im, ax=axs, label=self.p)

            plt.savefig(output_dir + 'results_' + str(i_d) + '_' + str(i_ech) + '_' + self.p + '.png')

        else:
            fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (21, 7))
            im = axs[0].imshow(self.X_test[i_d, i_ech, :, :, self.i_p])
            im = axs[1].imshow(self.y_pred[i_d, i_ech, :, :])
            im = axs[2].imshow(self.y_test[i_d, i_ech, :, :])

            axs[0].set_title('X_test')
            axs[1].set_title('y_pred')
            axs[2].set_title('y_test')

            fig.colorbar(im, ax=axs, label=self.p)

            plt.savefig(output_dir + 'results_' + str(i_d) + '_' + str(i_ech) + '_' + self.p + '.png')



    def plot_all(self, output_dir, base=False):
        if base:
            for i_d in range(len(self.data_y_test.dates)):
                for i_ech in range(len(self.data_baseline.echeances)):
                    self.plot_d_ech(i_d, i_ech, output_dir, base=base)
        else:
            for i_d in range(len(self.data_y_test.dates)):
                for i in range(len(self.data_y_pred.exheances)):
                    self.plot_i(i, output_dir, base=base)

    def plot_firsts(self, output_dir, base=False):
        if base:
            for i_d in range(5):
                for i_ech in range(len(self.data_baseline.echeances)):
                    self.plot_d_ech(i_d, i_ech, output_dir, base=base)
        else:
            for i_d in range(5):
                for i in range(len(self.data_y_pred.exheances)):
                    self.plot_i(i, output_dir, base=base)

    def rmse_global(self):
        rmse_baseline_matrix = np.zeros(self.baseline.shape)
        rmse_pred_matrix = np.zeros(self.y_pred.shape)

        rmse_baseline_global = []
        rmse_pred_global = []

        for i_d in range(self.baseline.shape[0]):
            for i_ech in range(self.baseline.shape[1]):
                rmse_baseline_matrix[i_d, i_ech, :, :] = (self.baseline[i_d, i_ech, :, :] - self.y_test[i_d, i_ech, :, :])**2
                rmse_baseline_global.append(np.mean(rmse_baseline_matrix[i_d, i_ech, :, :]))
            for i_ech in range(self.y_pred.shape[1]):
                rmse_pred_matrix[i_d, i_ech, :, :] = (self.y_pred[i_d, i_ech, :, :] - self.y_test[i_d, i_ech, :, :])**2
                rmse_pred_global.append(np.mean(rmse_pred_matrix[i_d, i_ech, :, :]))

        return rmse_baseline_matrix, rmse_pred_matrix, rmse_baseline_global, rmse_pred_global
        
    def rmse_terre(self):
        rmse_baseline_terre_matrix = np.zeros(self.baseline.shape)
        rmse_pred_terre_matrix = np.zeros(self.y_pred.shape)
        rmse_baseline_terre_global = []
        rmse_pred_terre_global = []
        rmse_baseline_matrix, rmse_pred_matrix, rmse_baseline_global, rmse_pred_global = self.rmse_global()
        ind_terre_mer = get_ind_terre_mer_500m()
        
        for i_d in range(self.baseline.shape[0]):
            for i_ech in range(self.baseline.shape[1]):
                rmse_baseline_terre_matrix[i_d, i_ech, :, :] = rmse_baseline_matrix[i_d, i_ech, :, :] * ind_terre_mer
                rmse_baseline_terre_global.append(np.mean(rmse_baseline_terre_global))
            for i_ech in range(self.y_pred.shape[1]):
                rmse_pred_terre_matrix[i_d, i_ech, :, :] = rmse_pred_matrix[i_d, i_ech, :, :] * ind_terre_mer
                rmse_pred_terre_global.append(np.mean(rmse_pred_terre_global))

        return rmse_baseline_terre_matrix, rmse_pred_terre_matrix, rmse_baseline_terre_global, rmse_pred_terre_global
        

    def rmse_mer(self):
        rmse_baseline_mer_matrix = np.zeros(self.baseline.shape)
        rmse_pred_mer_matrix = np.zeros(self.y_pred.shape)
        rmse_baseline_mer_global = []
        rmse_pred_mer_global = []
        rmse_baseline_matrix, rmse_pred_matrix, rmse_baseline_global, rmse_pred_global = self.rmse_global()
        ind_terre_mer = get_ind_terre_mer_500m()
        
        for i_d in range(self.baseline.shape[0]):
            for i_ech in range(self.baseline.shape[1]):
                rmse_baseline_mer_matrix[i_d, i_ech, :, :] = rmse_baseline_matrix[i_d, i_ech, :, :] * ind_terre_mer
                rmse_baseline_mer_global.append(np.mean(rmse_baseline_mer_global))
            for i_ech in range(self.y_pred.shape[1]):
                rmse_pred_mer_matrix[i_d, i_ech, :, :] = rmse_pred_matrix[i_d, i_ech, :, :] * ind_terre_mer
                rmse_pred_mer_global.append(np.mean(rmse_pred_mer_global))

        return rmse_baseline_mer_matrix, rmse_pred_mer_matrix, rmse_baseline_mer_global, rmse_pred_mer_global



        


    