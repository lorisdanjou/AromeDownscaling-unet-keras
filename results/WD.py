import numpy as np
import pandas as pd
import scipy.stats as sc
import utils
import matplotlib.pyplot as plt


def wasserstein_distance(a, b):
    dist_a = np.reshape(a, -1)
    dist_b = np.reshape(b, -1)
    return sc.wasserstein_distance(dist_a, dist_b)


# compute WD for the whole results dataframe
def compute_datewise_WD(results_df):
    wasserstein_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'datewise_wasserstein_distance_baseline' : [],
        'datewise_wasserstein_distance_pred' : []}
    )
    for i in range(len(results_df)):
        wasserstein_df.loc[len(wasserstein_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            wasserstein_distance(results_df.baseline.iloc[i], results_df.y_test.iloc[i]),
            wasserstein_distance(results_df.y_pred.iloc[i], results_df.y_test.iloc[i])
        ]
    return wasserstein_df


def compute_datewise_WD_terre(results_df):
    ind_terre_mer = np.reshape(utils.get_ind_terre_mer_500m(), -1)
    wasserstein_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'datewise_wasserstein_distance_baseline' : [],
        'datewise_wasserstein_distance_pred' : []}
    )
    for i in range(len(results_df)):
        dist_test     = ind_terre_mer * np.reshape(results_df.y_test.iloc[i], -1)
        dist_pred     = ind_terre_mer * np.reshape(results_df.y_pred.iloc[i], -1)
        dist_baseline = ind_terre_mer * np.reshape(results_df.baseline.iloc[i], -1)

        wasserstein_df.loc[len(wasserstein_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            sc.wasserstein_distance(dist_baseline, dist_test),
            sc.wasserstein_distance(dist_pred, dist_test)
        ]
    return wasserstein_df


def compute_datewise_WD_mer(results_df):
    ind_terre_mer = np.reshape(utils.get_ind_terre_mer_500m(), -1)
    wasserstein_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'datewise_wasserstein_distance_baseline' : [],
        'datewise_wasserstein_distance_pred' : []}
    )
    for i in range(len(results_df)):
        dist_test     = (1 - ind_terre_mer) * np.reshape(results_df.y_test.iloc[i], -1)
        dist_pred     = (1 - ind_terre_mer) * np.reshape(results_df.y_pred.iloc[i], -1)
        dist_baseline = (1 - ind_terre_mer) * np.reshape(results_df.baseline.iloc[i], -1)

        wasserstein_df.loc[len(wasserstein_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            sc.wasserstein_distance(dist_baseline, dist_test),
            sc.wasserstein_distance(dist_pred, dist_test)
        ]
    return wasserstein_df


# plot distributions
def plot_datewise_wasserstein_distance_distrib(wd_df, wd_df_terre, wd_df_mer, output_dir):    
    wd_df_baseline = wd_df['datewise_wasserstein_distance_baseline']
    wd_df_pred = wd_df['datewise_wasserstein_distance_pred']
    wd_df_baseline_terre = wd_df_terre['datewise_wasserstein_distance_baseline']
    wd_df_pred_terre = wd_df_terre['datewise_wasserstein_distance_pred']
    wd_df_baseline_mer = wd_df_mer['datewise_wasserstein_distance_baseline']
    wd_df_pred_mer = wd_df_mer['datewise_wasserstein_distance_pred']

    D_baseline = np.zeros((len(wd_df), 3))
    for i in range(len(wd_df_baseline)):
        D_baseline[i, 0] = wd_df_baseline[i]
        D_baseline[i, 1] = wd_df_baseline_terre[i]
        D_baseline[i, 2] = wd_df_baseline_mer[i]

    D_pred = np.zeros((len(wd_df), 3))
    for i in range(len(wd_df_pred)):
        D_pred[i, 0] = wd_df_pred[i]
        D_pred[i, 1] = wd_df_pred_terre[i]
        D_pred[i, 2] = wd_df_pred_mer[i]
        
    D = np.concatenate([D_baseline, D_pred], axis=1)
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
    ax.set_title('datewise wasserstein distance distribution')
    ax.tick_params(axis='x', rotation=45)

    plt.savefig(output_dir + 'distribution_wd.png')


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

        wd_df       = compute_datewise_WD(results_df)
        wd_df_terre = compute_datewise_WD_terre(results_df)
        wd_df_mer   = compute_datewise_WD_mer(results_df)

        plot_datewise_wasserstein_distance_distrib(wd_df, wd_df_terre, wd_df_mer, opt["path"]["results"])
