import numpy as np
import matplotlib.pyplot as plt
from random import *

def get_ind_terre_mer_500m():
    filepath = '/cnrm/recyf/Data/users/danjoul/dataset/static_G9KP_SURFIND.TERREMER.npy'
    return np.load(filepath)

def terre_mer(y, ind_terre_mer):
    y_terre, y_mer = np.zeros(y.shape), np.zeros(y.shape)
    for i_ech in range(y.shape[0]):
        y_terre[i_ech, :, :] = y[i_ech, :, :] * ind_terre_mer
        y_mer[i_ech, :, :]   = y[i_ech, :, :] * (1 - ind_terre_mer)
    return y_terre, y_mer

def get_indices_baseline(size_y_pred):
    indices = []
    nb_days = size_y_pred//11
    for i_d in range(nb_days):
        for i_ech in range(0, 11, 2):
            indices.append(11*i_d + i_ech)
    return indices



def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2))

class Results():

    def __init__(self, p, i_p, X_test, y_test, y_pred, baseline=[]): # /!\ jours chargement de la baseline
        self.p = p
        self.i_p = i_p
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.baseline = baseline


    def plot_i(self, i, output_dir, base=False):
        if base:
            fig, axs = plt.subplots(nrows=1,ncols=4, figsize = (25, 5))
            im = axs[0].imshow(self.X_test[i, :, :, self.i_p])
            im = axs[1].imshow(self.y_pred[i, :, :])
            im = axs[2].imshow(self.y_test[i, :, :])
            im = axs[3].imshow(self.baseline[i, :, :])

            axs[0].set_title('X_test')
            axs[1].set_title('baseline')
            axs[2].set_title('y_pred')
            axs[3].set_title('y_test')
            
            fig.colorbar(im, ax=axs, label=self.p)

            plt.savefig(output_dir + 'results_' + str(i) + '_' + self.p + '.png')
        else :
            fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (15, 4))
            im = axs[0].imshow(self.X_test[i, :, :, self.i_p])
            im = axs[1].imshow(self.y_pred[i, :, :])
            im = axs[2].imshow(self.y_test[i, :, :])

            axs[0].set_title('X_test')
            axs[1].set_title('y_pred')
            axs[2].set_title('y_test')

            fig.colorbar(im, ax=axs, label=self.p)

            plt.savefig(output_dir + 'results_' + str(i) + '_' + self.p + '.png')


    def plot_all(self, output_dir, base=False):
        if base:
            for i in range(self.baseline.shape[0]):
                self.plot_i(i, output_dir, base=base)
        else:
            for i in range(self.y_pred.shape[0]):
                self.plot_i(i, output_dir, base=base)

    def plot_20(self, output_dir, base=False):
        if base:
            indices = get_indices_baseline(self.y_pred.shape[0])
            indices = [indices[randint(0, len(indices))] for i in range(20)]
            for i in indices:
                self.plot_i(i, output_dir, base=base)
        else:
            indices = [randint(0, self.y_pred.shape[0]) for i in range(20)]
            for i in indices:
                self.plot_i(i, output_dir, base=base)

    def metric(self, metric):
        indices = get_indices_baseline(self.y_pred.shape[0])
        metric_global = metric(self.y_pred, self.y_test)

        ind_terre_mer = get_ind_terre_mer_500m()
        y_test_terre, y_test_mer = terre_mer(self.y_test, ind_terre_mer)
        y_pred_terre, y_pred_mer = terre_mer(self.y_pred, ind_terre_mer)
        baseline_terre, baseline_mer = terre_mer(self.baseline, ind_terre_mer)

        metric_mer = metric(y_pred_mer, y_test_mer)
        metric_terre = metric(y_pred_terre, y_test_terre)
        
        return metric_global, metric_terre, metric_mer

    def score(self, metric):
        indices = get_indices_baseline(self.y_pred.shape[0])
        score_global = metric(self.y_pred, self.y_test) / metric(self.baseline, self.y_test[indices, :, :])

        ind_terre_mer = get_ind_terre_mer_500m()
        y_test_terre, y_test_mer = terre_mer(self.y_test, ind_terre_mer)
        y_pred_terre, y_pred_mer = terre_mer(self.y_pred, ind_terre_mer)
        baseline_terre, baseline_mer = terre_mer(self.baseline, ind_terre_mer)


        score_mer = metric(y_pred_mer, y_test_mer) / metric(baseline_mer, y_test_mer[indices, :, :])
        score_terre = metric(y_pred_terre, y_test_terre) / metric(baseline_terre, y_test_terre[indices, :, :])
        
        return score_global, score_terre, score_mer

    