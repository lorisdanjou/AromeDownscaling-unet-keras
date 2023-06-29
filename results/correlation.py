import numpy as np
import pandas as pd
import utils
import scipy.stats as sc
import matplotlib.pyplot as plt


def correlation(results_df):
    corr_df = pd.DataFrame(
        [],
        columns=['dates', 'echeances', 'r_baseline', 'r_pred']
    )
    for i in range(len(results_df)):
        r_baseline, _ = sc.pearsonr(results_df.y_test.iloc[i].reshape(-1), results_df.baseline.iloc[i].reshape(-1))
        r_pred    , _ = sc.pearsonr(results_df.y_test.iloc[i].reshape(-1), results_df.y_pred.iloc[i].reshape(-1))
        corr_df.loc[len(corr_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            r_baseline,
            r_pred
        ]
    return corr_df


def correlation_terre(results_df): # using masks
    ind_terre_mer = utils.get_ind_terre_mer_500m()
    corr_df = pd.DataFrame(
        [],
        columns=['dates', 'echeances', 'r_baseline', 'r_pred']
    )
    for i in range(len(results_df)):
        y_test = np.ma.masked_array(results_df.y_test.iloc[i], (1-ind_terre_mer))
        y_pred = np.ma.masked_array(results_df.y_pred.iloc[i], (1-ind_terre_mer))
        baseline = np.ma.masked_array(results_df.baseline.iloc[i], (1-ind_terre_mer))

        r_baseline, _ = sc.pearsonr(
            baseline.reshape(-1),
            y_test.reshape(-1)
        )
        r_pred    , _ = sc.pearsonr(
            y_pred.reshape(-1),
            y_test.reshape(-1)
        )
        corr_df.loc[len(corr_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            r_baseline,
            r_pred
        ]
    return corr_df


def correlation_mer(results_df): # using masks
    ind_terre_mer = utils.get_ind_terre_mer_500m()
    corr_df = pd.DataFrame(
        [],
        columns=['dates', 'echeances', 'r_baseline', 'r_pred']
    )
    for i in range(len(results_df)):
        y_test = np.ma.masked_array(results_df.y_test.iloc[i], ind_terre_mer)
        y_pred = np.ma.masked_array(results_df.y_pred.iloc[i], ind_terre_mer)
        baseline = np.ma.masked_array(results_df.baseline.iloc[i], ind_terre_mer)

        r_baseline, _ = sc.pearsonr(
            baseline.reshape(-1),
            y_test.reshape(-1)
        )
        r_pred    , _ = sc.pearsonr(
            y_pred.reshape(-1),
            y_test.reshape(-1)
        )
        corr_df.loc[len(corr_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            r_baseline,
            r_pred
        ]
    return corr_df


def plot_corr_distrib(corr_df, corr_df_terre, corr_df_mer, output_dir):
    corr_df_pred           = corr_df['r_pred']
    corr_df_baseline       = corr_df['r_baseline']
    corr_df_pred_terre     = corr_df_terre['r_pred']
    corr_df_baseline_terre = corr_df_terre['r_baseline']
    corr_df_pred_mer       = corr_df_mer['r_pred']
    corr_df_baseline_mer   = corr_df_mer['r_baseline']

    D_baseline = np.zeros((len(corr_df), 3))
    for i in range(len(corr_df_baseline)):
        D_baseline[i, 0] = corr_df_baseline[i]
        D_baseline[i, 1] = corr_df_baseline_terre[i]
        D_baseline[i, 2] = corr_df_baseline_mer[i]

    D_pred = np.zeros((len(corr_df), 3))
    for i in range(len(corr_df_pred)):
        D_pred[i, 0] = corr_df_pred[i]
        D_pred[i, 1] = corr_df_pred_terre[i]
        D_pred[i, 2] = corr_df_pred_mer[i]
        
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
    ax.set_title('pearson correlation distribution')
    ax.tick_params(axis='x', rotation=45)

    plt.savefig(output_dir + 'distribution_corr.png')