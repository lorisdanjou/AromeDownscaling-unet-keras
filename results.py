import numpy as np
import matplotlib.pyplot as plt

class Results():

    def __init__(self, p, i_p, X_test, y_test, y_pred):
        self.p = p
        self.i_p = i_p
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred


    def plot_i(self, i, output_dir):
        fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (15, 4))
        im = axs[0].imshow(self.X_test[i, :, :, self.i_p])
        im = axs[1].imshow(self.y_pred[i, :, :])
        im = axs[2].imshow(self.y_test[i, :, :])

        axs[0].set_title('X_test')
        axs[1].set_title('y_pred')
        axs[2].set_title('y_test')

        fig.colorbar(im, ax=axs, label=self.p)

        plt.savefig(output_dir + 'results_' + str(i) + '_' + self.p + '.png')


    def plot_all(self, output_dir):
        for i in range(self.y_pred.shape[0]):
            self.plot_i(i, output_dir)