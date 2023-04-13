import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from random import *
from data import *

def get_ind_terre_mer_500m():
    filepath = '/cnrm/recyf/Data/users/danjoul/dataset/static_G9KP_SURFIND.TERREMER.npy'
    return np.load(filepath)

def mse(a, b):
    return (a - b)**2

def mae(a, b):
    return (np.abs(a - b))

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
        mse_baseline_matrix = np.zeros(self.baseline.y[:, 1:, :, :].shape)
        mse_pred_matrix = np.zeros(self.y_pred.y[:, 1:, :, :].shape)

        mse_baseline_global = []
        mse_pred_global = []

        for i_d in range(self.baseline.y.shape[0]):
            for i_ech in range(1, self.baseline.y.shape[1]):
                mse_baseline_matrix[i_d, i_ech-1, :, :] = (self.baseline.y[i_d, i_ech, :, :] - self.y_test.y[i_d, i_ech, :, :])**2
                mse_baseline_global.append(np.mean(mse_baseline_matrix[i_d, i_ech-1, :, :]))
            # for i_ech in range(self.y_pred.y.shape[1]):
                mse_pred_matrix[i_d, i_ech-1, :, :] = (self.y_pred.y[i_d, i_ech, :, :] - self.y_test.y[i_d, i_ech, :, :])**2
                mse_pred_global.append(np.mean(mse_pred_matrix[i_d, i_ech-1, :, :]))

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
                mse_baseline_terre_matrix[i_d, i_ech-1, :, :] = mse_baseline_matrix[i_d, i_ech-1, :, :] * ind_terre_mer
                mean = np.sum(mse_baseline_terre_matrix[i_d, i_ech-1, :, :])/np.sum(ind_terre_mer)
                mse_baseline_terre_global.append(mean)
                # mse_baseline_terre_global.append(np.mean(mse_baseline_terre_matrix[i_d, i_ech, :, :]))
            for i_ech in range(self.y_pred.y.shape[1]):
                mse_pred_terre_matrix[i_d, i_ech-1, :, :] = mse_pred_matrix[i_d, i_ech-1, :, :] * ind_terre_mer
                mean = np.sum(mse_pred_terre_matrix[i_d, i_ech-1, :, :])/np.sum(ind_terre_mer)
                mse_pred_terre_global.append(mean)
                # mse_pred_terre_global.append(np.mean(mse_pred_terre_matrix[i_d, i_ech, :, :]))

        return mse_baseline_terre_matrix, mse_pred_terre_matrix, mse_baseline_terre_global, mse_pred_terre_global
        

    def mse_mer(self):
        mse_baseline_mer_matrix = np.zeros(self.baseline.y[:, 1:, :, :].shape)
        mse_pred_mer_matrix = np.zeros(self.y_pred.y[:, 1:, :, :].shape)
        mse_baseline_mer_global = []
        mse_pred_mer_global = []
        mse_baseline_matrix, mse_pred_matrix, mse_baseline_global, mse_pred_global = self.mse_global()
        ind_terre_mer = get_ind_terre_mer_500m()
        
        for i_d in range(self.baseline.y.shape[0]):
            for i_ech in range(1, self.baseline.y.shape[1]):
                mse_baseline_mer_matrix[i_d, i_ech-1, :, :] = mse_baseline_matrix[i_d, i_ech-1, :, :] * (1 - ind_terre_mer)
                mean = np.sum(mse_baseline_mer_matrix[i_d, i_ech-1, :, :])/np.sum(1 - ind_terre_mer)
                mse_baseline_mer_global.append(mean)
                # mse_baseline_mer_global.append(np.mean(mse_baseline_mer_matrix[i_d, i_ech, :, :]))
            # for i_ech in range(self.y_pred.y.shape[1]):
                mse_pred_mer_matrix[i_d, i_ech-1, :, :] = mse_pred_matrix[i_d, i_ech-1, :, :] * (1 - ind_terre_mer)
                mean = np.sum(mse_pred_mer_matrix[i_d, i_ech-1, :, :])/np.sum(1 - ind_terre_mer)
                mse_pred_mer_global.append(mean)
                # mse_pred_mer_global.append(np.mean(mse_pred_mer_matrix[i_d, i_ech, :, :]))

        return mse_baseline_mer_matrix, mse_pred_mer_matrix, mse_baseline_mer_global, mse_pred_mer_global


    def score(self, metric):
        score_baseline_matrix = np.zeros(self.baseline.y[:, 1:, :, :].shape)
        score_pred_matrix = np.zeros(self.y_pred.y[:, 1:, :, :].shape)
        score_baseline = []
        score_pred = []

        for i_d in range(self.baseline.y.shape[0]):
            for i_ech in range(1, self.baseline.y.shape[1]):
                score_baseline_matrix[i_d, i_ech-1, :, :] = metric(self.baseline.y[i_d, i_ech, :, :], self.y_test.y[i_d, i_ech, :, :])
                score_baseline.append(np.mean(score_baseline_matrix[i_d, i_ech-1, :, :]))
                score_pred_matrix[i_d, i_ech-1, :, :] = metric(self.y_pred.y[i_d, i_ech, :, :], self.y_test.y[i_d, i_ech, :, :])
                score_pred.append(np.mean(score_pred_matrix[i_d, i_ech-1, :, :]))

        return score_baseline_matrix, score_pred_matrix, score_baseline, score_pred


    def score_terre(self, metric):
        score_baseline_terre_matrix = np.zeros(self.baseline.y[:, 1:, :, :].shape)
        score_pred_terre_matrix = np.zeros(self.y_pred.y[:, 1:, :, :].shape)
        score_baseline_terre = []
        score_pred_terre = []
        score_baseline_matrix, score_pred_matrix, score_baseline, score_pred = self.score(metric)
        ind_terre_mer = get_ind_terre_mer_500m()
        
        for i_d in range(self.baseline.y.shape[0]):
            for i_ech in range(1, self.baseline.y.shape[1]):
                score_baseline_terre_matrix[i_d, i_ech-1, :, :] = score_baseline_matrix[i_d, i_ech-1, :, :] * ind_terre_mer
                mean = np.sum(score_baseline_terre_matrix[i_d, i_ech-1, :, :])/np.sum(ind_terre_mer)
                score_baseline_terre.append(mean)
                score_pred_terre_matrix[i_d, i_ech-1, :, :] = score_pred_matrix[i_d, i_ech-1, :, :] * ind_terre_mer
                mean = np.sum(score_pred_terre_matrix[i_d, i_ech-1, :, :])/np.sum(ind_terre_mer)
                score_pred_terre.append(mean)

        return score_baseline_terre_matrix, score_pred_terre_matrix, score_baseline_terre, score_pred_terre


    def score_mer(self, metric):
        score_baseline_mer_matrix = np.zeros(self.baseline.y[:, 1:, :, :].shape)
        score_pred_mer_matrix = np.zeros(self.y_pred.y[:, 1:, :, :].shape)
        score_baseline_mer = []
        score_pred_mer = []
        score_baseline_matrix, score_pred_matrix, score_baseline, score_pred = self.score(metric)
        ind_terre_mer = get_ind_terre_mer_500m()
        
        for i_d in range(self.baseline.y.shape[0]):
            for i_ech in range(1, self.baseline.y.shape[1]):
                score_baseline_mer_matrix[i_d, i_ech-1, :, :] = score_baseline_matrix[i_d, i_ech-1, :, :] * (1 - ind_terre_mer)
                mean = np.sum(score_baseline_mer_matrix[i_d, i_ech-1, :, :])/np.sum(1 - ind_terre_mer)
                score_baseline_mer.append(mean)
                score_pred_mer_matrix[i_d, i_ech-1, :, :] = score_pred_matrix[i_d, i_ech-1, :, :] * (1 - ind_terre_mer)
                mean = np.sum(score_pred_mer_matrix[i_d, i_ech-1, :, :])/np.sum((1 - ind_terre_mer))
                score_pred_mer.append(mean)

        return score_baseline_mer_matrix, score_pred_mer_matrix, score_baseline_mer, score_pred_mer


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

        plt.savefig(output_dir + 'distribution_rmse.png')



    def plot_distrib(self, metric, metric_name, output_dir):
        score_baseline_matrix, score_pred_matrix, score_baseline, score_pred = self.score(metric)
        score_baseline_terre_matrix, score_pred_terre_matrix, score_baseline_terre, score_pred_terre = self.score_terre(metric)
        score_baseline_mer_matrix, score_pred_mer_matrix, score_baseline_mer, score_pred_mer = self.score_mer(metric)

        D_baseline = np.zeros((len(score_baseline), 3))
        for i in range(len(score_baseline)):
            D_baseline[i, 0] = score_baseline[i]
            D_baseline[i, 1] = score_baseline_terre[i]
            D_baseline[i, 2] = score_baseline_mer[i]

        D_pred = np.zeros((len(score_pred), 3))
        for i in range(len(score_pred)):
            D_pred[i, 0] = score_pred[i]
            D_pred[i, 1] = score_pred_terre[i]
            D_pred[i, 2] = score_pred_mer[i]

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
        ax.set_title(metric_name + ' distribution')
        ax.tick_params(axis='x', rotation=45)

        plt.savefig(output_dir + 'distribution_' +  metric_name + '.png')



        


    