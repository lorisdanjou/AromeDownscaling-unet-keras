import numpy as np
import matplotlib.pyplot as plt

class Results():

    def __init__(self, param, X_test, y_test, y_pred):
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.param = param


    def plot_i(self, i, output_dir):
        fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (15, 4))
        im = axs[0].imshow(self.X_test[i, :, :, self.param])
        im = axs[1].imshow(self.y_pred[i, :, :])
        im = axs[2].imshow(self.y_test[i, :, :])

        fig.colorbar(im, ax=axs, label=str(self.param))

        plt.savefig(output_dir + 'results_' + str(i) + '_' + self.param + '.png')


    def plot_all(self, output_dir):
        for i in range(self.y_pred.shape[0]):
            self.plot_i(i, output_dir)