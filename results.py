import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from random import *
from data import *

def get_ind_terre_mer_500m():
    filepath = '/cnrm/recyf/Data/users/danjoul/dataset/static_G9KP_SURFIND.TERREMER.npy'
    return np.load(filepath)


class Results():

    def __init__(self, p, i_p, X_test, y_test, y_pred, baseline):
        self.p = p
        self.i_p = i_p
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.y_pred = y_pred.copy()
        self.baseline = baseline.copy()


    def plot_d_ech(self, i_d, i_ech, output_dir, base=False):
        if base:
            if i_ech != 0:
                # fig, axs = plt.subplots(nrows=1,ncols=4, figsize = (28, 7))
                # im = axs[0].imshow(self.X_test.X[i_d, 2*i_ech, :, :, self.i_p])
                # im = axs[1].imshow(self.baseline.y[i_d, i_ech, :, :])
                # im = axs[2].imshow(self.y_pred.y[i_d, 2*i_ech, :, :])
                # im = axs[3].imshow(self.y_test.y[i_d, 2*i_ech, :, :])

                # axs[0].set_title('X_test')
                # axs[1].set_title('baseline')
                # axs[2].set_title('y_pred')
                # axs[3].set_title('y_test')

                # fig.colorbar(im, ax=axs, label=self.p)

                # plt.savefig(output_dir + 'results_' + str(i_d) + '_' + str(i_ech) + '_' + self.p + '.png')

                fig, axs = plt.subplots(nrows=1,ncols=4, figsize = (28, 7))
                images = []
                data = [self.X_test.X[i_d, i_ech, :, :, self.i_p], self.baseline.y[i_d, i_ech, :, :], self.y_pred.y[i_d, i_ech, :, :], self.y_test.y[i_d, i_ech, :, :]]
                for i in range(4):
                    images.append(axs[i].imshow(data[i]))
                    axs[i].label_outer()
                vmin = min(image.get_array().min() for image in images)
                vmax = max(image.get_array().max() for image in images)
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                for im in images:
                    im.set_norm(norm)
                axs[0].set_title('X_test')
                axs[1].set_title('baseline')
                axs[2].set_title('y_pred')
                axs[3].set_title('y_test')
                fig.colorbar(images[0], ax=axs)
                plt.savefig(output_dir + 'results_' + str(i_d) + '_' + str(i_ech) + '_' + self.p + '.png')

        else:
            # fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (21, 7))
            # im = axs[0].imshow(self.X_test.X[i_d, i_ech, :, :, self.i_p])
            # im = axs[1].imshow(self.y_pred.y[i_d, i_ech, :, :])
            # im = axs[2].imshow(self.y_test.y[i_d, i_ech, :, :])

            # axs[0].set_title('X_test')
            # axs[1].set_title('y_pred')
            # axs[2].set_title('y_test')

            # fig.colorbar(im, ax=axs, label=self.p)

            # plt.savefig(output_dir + 'results_' + str(i_d) + '_' + str(i_ech) + '_' + self.p + '.png')

            fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (21, 7))
            images = []
            data = [self.baseline.y[i_d, i_ech, :, :], self.y_pred.y[i_d, 2*i_ech, :, :], self.y_test.y[i_d, 2*i_ech, :, :]]
            for i in range(3):
                images.append(axs[i].imshow(data[i]))
                axs[i].label_outer()
            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            axs[0].set_title('X_test')
            axs[1].set_title('y_pred')
            axs[2].set_title('y_test')
            fig.colorbar(images[0], ax=axs)
            plt.savefig(output_dir + 'results_' + str(i_d) + '_' + str(i_ech) + '_' + self.p + '.png')



    def plot_all(self, output_dir, base=False):
        if base:
            for i_d in range(len(self.y_test.dates)):
                for i_ech in range(len(self.baseline.echeances)):
                    self.plot_d_ech(i_d, i_ech, output_dir, base=base)
        else:
            for i_d in range(len(self.y_test.dates)):
                for i in range(len(self.y_pred.exheances)):
                    self.plot_i(i, output_dir, base=base)

    def plot_firsts(self, output_dir, base=False):
        if base:
            for i_d in range(5):
                for i_ech in range(len(self.baseline.echeances)):
                    self.plot_d_ech(i_d, i_ech, output_dir, base=base)
        else:
            for i_d in range(5):
                for i in range(len(self.y_pred.echeances)):
                    self.plot_i(i, output_dir, base=base)

    def mse_global(self):
        mse_baseline_matrix = np.zeros(self.baseline.y.shape)
        mse_pred_matrix = np.zeros(self.y_pred.y.shape)

        mse_baseline_global = []
        mse_pred_global = []

        for i_d in range(self.baseline.y.shape[0]):
            for i_ech in range(self.baseline.y.shape[1]):
                mse_baseline_matrix[i_d, i_ech, :, :] = (self.baseline.y[i_d, i_ech, :, :] - self.y_test.y[i_d, i_ech, :, :])**2
                mse_baseline_global.append(np.mean(mse_baseline_matrix[i_d, i_ech, :, :]))
            for i_ech in range(self.y_pred.y.shape[1]):
                mse_pred_matrix[i_d, i_ech, :, :] = (self.y_pred.y[i_d, i_ech, :, :] - self.y_test.y[i_d, i_ech, :, :])**2
                mse_pred_global.append(np.mean(mse_pred_matrix[i_d, i_ech, :, :]))

        return mse_baseline_matrix, mse_pred_matrix, mse_baseline_global, mse_pred_global
        
    def mse_terre(self):
        mse_baseline_terre_matrix = np.zeros(self.baseline.y.shape)
        mse_pred_terre_matrix = np.zeros(self.y_pred.y.shape)
        mse_baseline_terre_global = []
        mse_pred_terre_global = []
        mse_baseline_matrix, mse_pred_matrix, mse_baseline_global, mse_pred_global = self.mse_global()
        ind_terre_mer = get_ind_terre_mer_500m()
        
        for i_d in range(self.baseline.y.shape[0]):
            for i_ech in range(self.baseline.y.shape[1]):
                mse_baseline_terre_matrix[i_d, i_ech, :, :] = mse_baseline_matrix[i_d, i_ech, :, :] * ind_terre_mer
                mse_baseline_terre_global.append(np.mean(mse_baseline_terre_matrix[i_d, i_ech, :, :]))
            for i_ech in range(self.y_pred.y.shape[1]):
                mse_pred_terre_matrix[i_d, i_ech, :, :] = mse_pred_matrix[i_d, i_ech, :, :] * ind_terre_mer
                mse_pred_terre_global.append(np.mean(mse_pred_terre_matrix[i_d, i_ech, :, :]))

        return mse_baseline_terre_matrix, mse_pred_terre_matrix, mse_baseline_terre_global, mse_pred_terre_global
        

    def mse_mer(self):
        mse_baseline_mer_matrix = np.zeros(self.baseline.y.shape)
        mse_pred_mer_matrix = np.zeros(self.y_pred.y.shape)
        mse_baseline_mer_global = []
        mse_pred_mer_global = []
        mse_baseline_matrix, mse_pred_matrix, mse_baseline_global, mse_pred_global = self.mse_global()
        ind_terre_mer = get_ind_terre_mer_500m()
        
        for i_d in range(self.baseline.y.shape[0]):
            for i_ech in range(self.baseline.y.shape[1]):
                mse_baseline_mer_matrix[i_d, i_ech, :, :] = mse_baseline_matrix[i_d, i_ech, :, :] * (1 - ind_terre_mer)
                mse_baseline_mer_global.append(np.mean(mse_baseline_mer_matrix[i_d, i_ech, :, :]))
            for i_ech in range(self.y_pred.y.shape[1]):
                mse_pred_mer_matrix[i_d, i_ech, :, :] = mse_pred_matrix[i_d, i_ech, :, :] * (1 - ind_terre_mer)
                mse_pred_mer_global.append(np.mean(mse_pred_mer_matrix[i_d, i_ech, :, :]))

        return mse_baseline_mer_matrix, mse_pred_mer_matrix, mse_baseline_mer_global, mse_pred_mer_global


    def plot_distrib_rmse(self, output_dir):
        mse_baseline_matrix, mse_pred_matrix, mse_baseline_global, mse_pred_global = self.mse_global()
        mse_baseline_terre_matrix, mse_pred_terre_matrix, mse_baseline_terre_global, mse_pred_terre_global = self.mse_terre()
        mse_baseline_mer_matrix, mse_pred_mer_matrix, mse_baseline_mer_global, mse_pred_mer_global = self.mse_mer()

        D_baseline = np.zeros((len(mse_baseline_global), 3))
        for i in range(len(mse_baseline_global)):
            D_baseline[i, 0] = np.sqrt(mse_baseline_global[i])
            D_baseline[i, 1] = np.sqrt(mse_baseline_terre_global[i])
            D_baseline[i, 2] = np.sqrt(mse_baseline_mer_global[i])

        D_pred = np.zeros((len(mse_pred_global), 3))
        for i in range(len(mse_pred_global)):
            D_pred[i, 0] = np.sqrt(mse_pred_global[i])
            D_pred[i, 1] = np.sqrt(mse_pred_terre_global[i])
            D_pred[i, 2] = np.sqrt(mse_pred_mer_global[i])

        D = np.concatenate([D_baseline, D_pred], axis=1)
        # labels = ['global', 'terre', 'mer']
        labels = ['global_baseline', 'terre_baseline', 'mer_baseline', 'global_pred', 'terre_pred', 'mer_pred']

        fig, ax = plt.subplots(figsize=(10, 11))
        plt.grid()
        VP = ax.boxplot(D, positions=[3, 6, 9, 12, 15, 18], widths=1.5, patch_artist=True,
                        showmeans=True, showfliers=False,
                        medianprops={"color": "white", "linewidth": 0.5},
                        boxprops={"facecolor": "C0", "edgecolor": "white",
                                "linewidth": 0.5},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5},
                        labels=labels)
        ax.set_title('RMSE distribution')
        ax.tick_params(axis='x', rotation=45)

        # fig, axs = plt.subplots(1, 2)
        # plt.grid()
        # VP = axs[0].boxplot(D_baseline, positions=[2,4,6], widths=1.5, patch_artist=True,
        #                 showmeans=False, showfliers=False,
        #                 medianprops={"color": "white", "linewidth": 0.5},
        #                 boxprops={"facecolor": "C0", "edgecolor": "white",
        #                         "linewidth": 0.5},
        #                 whiskerprops={"color": "C0", "linewidth": 1.5},
        #                 capprops={"color": "C0", "linewidth": 1.5},
        #                 labels=labels)

        # VP = axs[1].boxplot(D_pred, positions=[2,4,6], widths=1.5, patch_artist=True,
        #                 showmeans=False, showfliers=False,
        #                 medianprops={"color": "white", "linewidth": 0.5},
        #                 boxprops={"facecolor": "C0", "edgecolor": "white",
        #                         "linewidth": 0.5},
        #                 whiskerprops={"color": "C0", "linewidth": 1.5},
        #                 capprops={"color": "C0", "linewidth": 1.5},
        #                 labels=labels)

        # axs[0].set_title('RMSE distribution (baseline)')
        # axs[1].set_title('RMSE distribution (y_pred)')

        plt.savefig(output_dir + 'distribution_rmse.png')



        


    