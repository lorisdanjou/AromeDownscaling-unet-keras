import numpy as np
import matplotlib.pyplot as plt
from random import *

def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2))

class Results():

    def __init__(self, p, i_p, X_test, y_test, y_pred, baseline=[]):
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
        for i in range(self.y_pred.shape[0]):
            self.plot_i(i, output_dir, base=base)

    def plot_20(self, output_dir, base=False):
        index = [randint(0, self.y_pred.shape[0]) for i in range(20)]
        for i in index:
            self.plot_i(i, output_dir, base=base)

    def rmse_score(self):
        return rmse(self.y_pred[::2, :, :], self.baseline)

    