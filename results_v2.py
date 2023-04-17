import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

def get_ind_terre_mer_500m():
    filepath = '/cnrm/recyf/Data/users/danjoul/dataset/static_G9KP_SURFIND.TERREMER.npy'
    return np.load(filepath)

'''
Metrics
'''
def mse(a, b):
    return (a - b)**2

def mae(a, b):
    return (np.abs(a - b))

def biais(a, b):
    return a - b


'''
Load Data
'''
def load_data(working_dir, dates_test, echeances, resample, data_test_location, baseline_location, param='t2m'):
    data_pred = np.load(working_dir + 'y_pred.npy')
    results_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'X_test' : [],
        'baseline' : [],
        'y_pred' : [],
        'y_test' : []}
    )
    for i_d, d in enumerate(dates_test):
        # Load X_test :
        try:
            if resample == 'c':
                filepath_X_test = data_test_location + 'oper_c_' + d.isoformat() + 'Z_' + param + '.npy'
            else:
                filepath_X_test = data_test_location + 'oper_r_' + d.isoformat() + 'Z_' + param + '.npy'
            X_test = np.load(filepath_X_test)
        except FileNotFoundError:
            print('missing day : ' + d.isoformat())
            X_test = None

        # Load baseline : 
        filepath_baseline = baseline_location + 'GG9B_' + d.isoformat() + 'Z_' + param + '.npy'
        baseline = np.load(filepath_baseline)
        try:
            filepath_baseline = baseline_location + 'GG9B_' + d.isoformat() + 'Z_' + param + '.npy'
            baseline = np.load(filepath_baseline)
        except FileNotFoundError:
            print('missing day : ' + d.isoformat())
            baseline = None

        # Load y_test : 
        try:
            filepath_X_test = data_test_location + 'G9L1_' + d.isoformat() + 'Z_' + param + '.npy'
            y_test = np.load(filepath_X_test)
        except FileNotFoundError:
            print('missing day : ' + d.isoformat())
            y_test = None

        for i_ech, ech in enumerate(echeances):
            try:
                results_d_ech = pd.DataFrame(
                    {'dates' : [dates_test[i_d].isoformat()],
                    'echeances' : [echeances[i_ech]],
                    'X_test' : [X_test[:, :, i_ech]],
                    'baseline' : [baseline[:, :, i_ech]],
                    'y_pred' : [data_pred[i_d, i_ech, :, :]],
                    'y_test' : [y_test[:, :, i_ech]]}
                )
            except TypeError:
                results_d_ech = pd.DataFrame(
                    {'dates' : [],
                    'echeances' : [],
                    'X_test' : [],
                    'baseline' : [],
                    'y_pred' : [],
                    'y_test' : []}
                )
            results_df = pd.concat([results_df, results_d_ech])
    return results_df.reset_index(drop=True)


'''
Get scores global/terre/mer
'''
def get_scores(results_df, metric, metric_name):
    metric_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(results_df)):
        metric_i = pd.DataFrame(
            {'dates' : [results_df.dates[i]],
            'echeances' : [results_df.echeances[i]],
            metric_name + '_baseline_map' : [metric(results_df.baseline[i], results_df.y_test[i])],
            metric_name + '_y_pred_map' : [metric(results_df.y_pred[i], results_df.y_test[i])],
            metric_name + '_baseline_mean' : [np.mean(metric(results_df.baseline[i], results_df.y_test[i]))],
            metric_name + '_y_pred_mean' : [np.mean(metric(results_df.y_pred[i], results_df.y_test[i]))]}
        )
        metric_df = pd.concat([metric_df, metric_i])
    return metric_df.reset_index(drop=True)


def get_scores_terre(results_df, metric, metric_name):
    metric_df = get_scores(results_df, metric, metric_name)
    ind_terre_mer = get_ind_terre_mer_500m()
    metric_terre_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(results_df)):
        metric_i = pd.DataFrame(
            {'dates' : [results_df.dates[i]],
            'echeances' : [results_df.echeances[i]],
            metric_name + '_baseline_map' : [metric_df[metric_name + '_baseline_map'][i]*ind_terre_mer],
            metric_name + '_y_pred_map' : [metric_df[metric_name + '_y_pred_map'][i]*ind_terre_mer],
            metric_name + '_baseline_mean' : [np.sum(metric_df[metric_name + '_baseline_map'][i]*ind_terre_mer)/np.sum(ind_terre_mer)],
            metric_name + '_y_pred_mean' : [np.sum(metric_df[metric_name + '_y_pred_map'][i]*ind_terre_mer)/np.sum(ind_terre_mer)]}
        )
        metric_terre_df = pd.concat([metric_terre_df, metric_i])
    return metric_terre_df.reset_index(drop=True)


def get_scores_mer(results_df, metric, metric_name):
    metric_df = get_scores(results_df, metric, metric_name)
    ind_terre_mer = get_ind_terre_mer_500m()
    metric_mer_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(results_df)):
        metric_i = pd.DataFrame(
            {'dates' : [results_df.dates[i]],
            'echeances' : [results_df.echeances[i]],
            metric_name + '_baseline_map' : [metric_df[metric_name + '_baseline_map'][i]*(1-ind_terre_mer)],
            metric_name + '_y_pred_map' : [metric_df[metric_name + '_y_pred_map'][i]*(1-ind_terre_mer)],
            metric_name + '_baseline_mean' : [np.sum(metric_df[metric_name + '_baseline_map'][i]*(1-ind_terre_mer))/np.sum((1-ind_terre_mer))],
            metric_name + '_y_pred_mean' : [np.sum(metric_df[metric_name + '_y_pred_map'][i]*(1-ind_terre_mer))/np.sum((1-ind_terre_mer))]}
        )
        metric_mer_df = pd.concat([metric_mer_df, metric_i])
    return metric_mer_df.reset_index(drop=True)


'''
Plots
'''
def plot_results(results_df, param,  output_dir):
    for i in range(10):
        fig, axs = plt.subplots(nrows=1,ncols=4, figsize = (28, 7))
        data = [results_df.X_test[i], results_df.baseline[i], results_df.y_pred[i], results_df.y_test[i]]
        images = []
        for j in range(4):
            images.append(axs[j].imshow(data[j]))
            axs[j].label_outer()
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
        plt.savefig(output_dir + 'results_' + str(i) + '_' + param + '.png')


def plot_score_maps(results_df, metric, metric_name, output_dir):
    for i in range(10):
        metric_df = get_scores(results_df, metric, metric_name)
        fig, axs = plt.subplots(nrows=1,ncols=2, figsize = (25, 12))
        images = []
        data = [metric_df[metric_name + '_baseline_map'][i], metric_df[metric_name + '_y_pred_map'][i]]
        for j in range(len(data)):
            im = axs[j].imshow(data[j], cmap='coolwarm')
            images.append(im)
            axs[j].label_outer()
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        axs[0].set_title('baseline global')
        axs[1].set_title('pred global')
        fig.colorbar(images[0], ax=axs)
        plt.savefig(output_dir + metric_name + str(i) + '_map.png')


def plot_distrib(results_df, metric, metric_name, output_dir):
        score_baseline = get_scores(results_df, metric, metric_name)[metric_name + '_baseline_mean']
        score_baseline_terre = get_scores_terre(results_df, metric, metric_name)[metric_name + '_baseline_mean']
        score_baseline_mer = get_scores_mer(results_df, metric, metric_name)[metric_name + '_baseline_mean']
        score_pred = get_scores(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
        score_pred_terre = get_scores_terre(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
        score_pred_mer = get_scores_mer(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
        
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
                        showmeans=True, meanline=True, showfliers=False,
                        medianprops={"color": "white", "linewidth": 0.5},
                        boxprops={"facecolor": "C0", "edgecolor": "white",
                                "linewidth": 0.5},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5},
                        meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                        labels=labels)
        ax.set_title(metric_name + ' distribution')
        ax.tick_params(axis='x', rotation=45)

        plt.savefig(output_dir + 'distribution_' +  metric_name + '.png')
