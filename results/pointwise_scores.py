import numpy as np
from skimage.metrics import structural_similarity
import pandas as pd
import utils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs


# score functions
def mse(a, b):
    return (a - b)**2

def mae(a, b):
    return (np.abs(a - b))

def bias(a, b):
    return a - b

def ssim(a, b):
    _, ssim_map = structural_similarity(
        a,
        b, 
        data_range=b.max() - b.min(),
        win_size=None,
        full=True
    )
    return ssim_map


# compute pointwise scores for a dataframe
def compute_score(results_df, metric, metric_name):
    metric_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(results_df)):
        metric_df.loc[len(metric_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            metric(results_df.baseline.iloc[i], results_df.y_test.iloc[i]),
            metric(results_df.y_pred.iloc[i], results_df.y_test.iloc[i]),
            np.mean(metric(results_df.baseline.iloc[i], results_df.y_test.iloc[i])),
            np.mean(metric(results_df.y_pred.iloc[i], results_df.y_test.iloc[i])),
        ]
    return metric_df


def compute_score_terre(metric_df, metric_name):
    ind_terre_mer = utils.get_ind_terre_mer_500m()
    metric_terre_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(metric_df)):
        metric_terre_df.loc[len(metric_terre_df)] = [
            metric_df.dates[i],
            metric_df.echeances[i],
            metric_df[metric_name + '_baseline_map'][i]*ind_terre_mer,
            metric_df[metric_name + '_y_pred_map'][i]*ind_terre_mer,
            np.sum(metric_df[metric_name + '_baseline_map'][i]*ind_terre_mer)/np.sum(ind_terre_mer),
            np.sum(metric_df[metric_name + '_y_pred_map'][i]*ind_terre_mer)/np.sum(ind_terre_mer)
        ]
    return metric_terre_df


def compute_score_mer(metric_df, metric_name):
    ind_terre_mer = utils.get_ind_terre_mer_500m()
    metric_mer_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(metric_df)):
        metric_mer_df.loc[len(metric_mer_df)] = [
            metric_df.dates[i],
            metric_df.echeances[i],
            metric_df[metric_name + '_baseline_map'][i]*(1 - ind_terre_mer),
            metric_df[metric_name + '_y_pred_map'][i]*(1 - ind_terre_mer),
            np.sum(metric_df[metric_name + '_baseline_map'][i]*(1 - ind_terre_mer))/np.sum((1 - ind_terre_mer)),
            np.sum(metric_df[metric_name + '_y_pred_map'][i]*(1 - ind_terre_mer))/np.sum((1 - ind_terre_mer))
        ]
    return metric_mer_df


def plot_score_maps(metric_df, output_dir, metric_name, unit, cmap="viridis", n=10):
    for i in range(n):
        fig = plt.figure(figsize=[25, 12])
        fig.suptitle(metric_name, fontsize=30)
        axs = []
        for j in range(2):
            axs.append(fig.add_subplot(1, 2, j+1, projection=ccrs.PlateCarree()))
            axs[j].set_extent(utils.IMG_EXTENT)
            axs[j].coastlines(resolution='10m', color='black', linewidth=1)

        data = [metric_df[metric_name + '_baseline_map'][i], metric_df[metric_name + '_y_pred_map'][i]]
        images = []
        for j in range(2):
            images.append(axs[j].imshow(data[j], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
            axs[j].label_outer()
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        axs[0].set_title("fullpos", fontdict={"fontsize": 20})
        axs[1].set_title("Unet", fontdict={"fontsize": 20})
        fig.colorbar(images[0], ax=axs, label="{} [{}]".format(metric_name, unit))
        plt.savefig(output_dir + metric_name + str(i) + '_map.png', bbox_inches="tight")


def plot_unique_score_map(metric_df, output_dir, metric_name, unit, cmap="viridis", n=10):
    metric_baseline = metric_df[metric_name + '_baseline_map'].mean()
    metric_y_pred   = metric_df[metric_name + '_y_pred_map'].mean()
    fig = plt.figure(figsize=[25, 12])
    fig.suptitle(metric_name, fontsize=30)
    axs = []
    for j in range(2):
        axs.append(fig.add_subplot(1, 2, j+1, projection=ccrs.PlateCarree()))
        axs[j].set_extent(utils.IMG_EXTENT)
        axs[j].coastlines(resolution='10m', color='black', linewidth=1)

    data = [metric_baseline, metric_y_pred]
    images = []
    for j in range(2):
        images.append(axs[j].imshow(data[j], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
        axs[j].label_outer()
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    axs[0].set_title("fullpos", fontdict={"fontsize": 20})
    axs[1].set_title("Unet", fontdict={"fontsize": 20})
    fig.colorbar(images[0], ax=axs, label="{} [{}]".format(metric_name, unit))
    plt.savefig(output_dir + metric_name + '_unique_map.png', bbox_inches="tight")


######################################## Anciennes versions avec imshow #####################################################
# plot score maps
# def plot_score_maps(metric_df, metric_name, output_dir, cmap='coolwarm'):
#     for i in range(10):
#         metric_df = compute_score(metric_df, metric_name)
#         fig, axs = plt.subplots(nrows=1,ncols=2, figsize = (25, 12))
#         images = []
#         data = [metric_df[metric_name + '_baseline_map'][i], metric_df[metric_name + '_y_pred_map'][i]]
#         for j in range(len(data)):
#             im = axs[j].imshow(data[j], cmap=cmap, origin="lower")
#             images.append(im)
#             axs[j].label_outer()
#         vmin = min(image.get_array().min() for image in images)
#         vmax = max(image.get_array().max() for image in images)
#         norm = colors.Normalize(vmin=vmin, vmax=vmax)
#         for im in images:
#             im.set_norm(norm)
#         axs[0].set_title('baseline global')
#         axs[1].set_title('pred global')
#         fig.colorbar(images[0], ax=axs)
#         plt.savefig(output_dir + metric_name + str(i) + '_map.png')

# def plot_unique_score_map(metric_df, metric_name, output_dir, cmap='coolwarm'):
#     metric_df = compute_score(metric_df, metric_name)
#     metric_baseline = metric_df[metric_name + '_baseline_map'].mean()
#     metric_y_pred   = metric_df[metric_name + '_baseline_map'].mean()
#     fig, axs = plt.subplots(nrows=1,ncols=2, figsize = (25, 12))
#     images = []
#     data = [metric_baseline, metric_y_pred]
#     for j in range(len(data)):
#         im = axs[j].imshow(data[j], cmap=cmap, origin="lower")
#         images.append(im)
#         axs[j].label_outer()
#     vmin = min(image.get_array().min() for image in images)
#     vmax = max(image.get_array().max() for image in images)
#     norm = colors.Normalize(vmin=vmin, vmax=vmax)
#     for im in images:
#         im.set_norm(norm)
#     mean = metric_baseline.mean()
#     axs[0].set_title('baseline ' + metric_name  + ' ' + f'{mean:.2f}')
#     mean = metric_y_pred.mean()
#     axs[1].set_title('pred ' + metric_name  + ' ' + f'{mean:.2f}')
#     fig.colorbar(images[0], ax=axs)
#     plt.savefig(output_dir + metric_name + '_unique_map.png')
############################################################################################################################

def plot_distrib(metric_df, metric_name, output_dir):
    score_baseline = metric_df[metric_name + '_baseline_mean']
    score_baseline_terre = compute_score_terre(metric_df, metric_name)[metric_name + '_baseline_mean']
    score_baseline_mer = compute_score_mer(metric_df, metric_name)[metric_name + '_baseline_mean']
    score_pred = metric_df[metric_name + '_y_pred_mean']
    score_pred_terre = compute_score_terre(metric_df, metric_name)[metric_name + '_y_pred_mean']
    score_pred_mer = compute_score_mer(metric_df, metric_name)[metric_name + '_y_pred_mean']
    
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


if __name__ == "__main__":
    import os
    import argparse
    import core.logger as logger
    import load_results as lr
    import warnings
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_example.jsonc',
                        help='JSON file for configuration')

    # parse configs
    args = parser.parse_args()
    opt = logger.parse(args)


    # load & plot results
    y_pred_path = os.path.join(opt["path"]["experiment"], "y_pred.csv")

    for i_p, param in enumerate(opt["data"]["params_out"]):
        results_df = lr.load_results(
            "/cnrm/recyf/Data/users/danjoul/unet_experiments/tests/y_pred.csv",
            resample = opt["data"]["interp"],
            data_test_location = opt["data"]["data_test_location"],
            baseline_location = opt["data"]["baseline_location"],
            param=param
        )
        
        # MAE
        mae_df = compute_score(results_df, mae, "MAE")
        plot_score_maps(mae_df, output_dir=opt["path"]["results"], metric_name="MAE", unit="K", cmap=opt["results"]["cmap"])
        plot_unique_score_map(mae_df, output_dir=opt["path"]["results"], metric_name="MAE", unit="K", cmap=opt["results"]["cmap"])
        plot_distrib(mae_df, "MAE", opt["path"]["results"])
