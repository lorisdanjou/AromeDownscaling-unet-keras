import numpy as np
import pandas as pd
from results_v2 import *
from bronx.stdtypes.date import daterangex as rangex
import matplotlib.pyplot as plt
from matplotlib import colors


def synthesis_maps(expes, output_dir, dates_test, echeances, resample, data_test_location, baseline_location, param='t2m', full=False):
    for k in range(10):
        if full:
            fig, axs = plt.subplots(nrows=4, ncols=len(expes), figsize=(5*len(expes), 16))
            images = []
            for j in range(len(expes)):
                working_dir = expes.dir[j]
                name = expes.name[j]
                results_df = load_data(working_dir, dates_test, echeances, resample, data_test_location, baseline_location, param=param)
                data = [results_df.X_test[k], results_df.baseline[k], results_df.y_pred[k], results_df.y_test[k]]
                for i in range(len(data)):
                    im = axs[i, j].imshow(data[i], cmap='viridis')
                    images.append(im)
                    axs[i, j].label_outer()
                axs[0, j].set_title('X_test ' + name)
                axs[1, j].set_title('baseline ' + name)
                axs[2, j].set_title('y_pred ' + name)
                axs[3, j].set_title('y_test ' + name)

            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            fig.colorbar(images[0], ax=axs)
            plt.savefig(output_dir + 'synthesis_' + str(k) + '_map.png', bbox_inches='tight')
        else:
            fig, axs = plt.subplots(nrows=1, ncols=len(expes), figsize=(5*len(expes), 4))
            images = []
            for j in range(len(expes)):
                working_dir = expes.dir[j]
                name = expes.name[j]
                results_df = load_data(working_dir, dates_test, echeances, resample, data_test_location, baseline_location, param=param)
                im = axs[j].imshow(results_df.y_pred[k], cmap='viridis')
                images.append(im)
                axs[j].label_outer()
                axs[j].set_title('y_pred ' + name)

            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            fig.colorbar(images[0], ax=axs)
            plt.savefig(output_dir + 'synthesis_' + str(k) + '_map.png', bbox_inches='tight')


def synthesis_score_maps(expes, output_dir, metric, metric_name, dates_test, echeances, resample, data_test_location, baseline_location, param='t2m'):
    for k in range(10):
        fig, axs = plt.subplots(nrows=1, ncols=len(expes)+1, figsize=(5*len(expes), 5))
        images = []
        for j in range(len(expes)):
            working_dir = expes.dir[j]
            name = expes.name[j]
            results_df = load_data(working_dir, dates_test, echeances, resample, data_test_location, baseline_location, param=param)
            metric_df  = get_scores(results_df, metric, metric_name)
            im = axs[j].imshow(metric_df[metric_name + '_y_pred_map'][k], cmap='coolwarm')
            images.append(im)
            axs[j].label_outer()
            axs[j].set_title(metric_name + ' y_pred ' + name)
        im = axs[j+1].imshow(metric_df[metric_name + '_baseline_map'][k], cmap='coolwarm')
        images.append(im)
        axs[j+1].label_outer()
        axs[j+1].set_title(metric_name + ' baseline ')

        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        fig.colorbar(images[0], ax=axs)
        plt.savefig(output_dir + metric_name + '_' + str(k) + '_map.png', bbox_inches='tight')


def synthesis_score_distribs(expes, output_dir, metric, metric_name, dates_test, echeances, resample, data_test_location, baseline_location, param):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))    
    D = []
    D_terre = []
    D_mer = []
    labels = list(expes.name) + ['baseline']
    for i in range(len(expes)):
        working_dir = expes.dir[i]
        results_df = load_data(working_dir, dates_test, echeances, resample, data_test_location, baseline_location, param=param)
        score_baseline = get_scores(results_df, metric, metric_name)[metric_name + '_baseline_mean']
        score_pred = get_scores(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
        score_baseline_terre = get_scores_terre(results_df, metric, metric_name)[metric_name + '_baseline_mean']
        score_pred_terre = get_scores_terre(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
        score_baseline_mer = get_scores_mer(results_df, metric, metric_name)[metric_name + '_baseline_mean']
        score_pred_mer = get_scores_mer(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
        
        D.append(score_pred)
        D_terre.append(score_pred_terre)
        D_mer.append(score_pred_mer)
    D.append(score_baseline)
    D_terre.append(score_baseline_terre)
    D_mer.append(score_baseline_mer)
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    VP = axs[0].boxplot(D, positions=range(0, 3*(len(expes)+1), 3), widths=1.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels)
    VP = axs[1].boxplot(D_terre, positions=range(0, 3*(len(expes)+1), 3), widths=1.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels)
    VP = axs[2].boxplot(D_mer, positions=range(0, 3*(len(expes)+1), 3), widths=1.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels)
    axs[0].set_title(metric_name + ' distribution')
    axs[1].set_title(metric_name + ' terre distribution')
    axs[2].set_title(metric_name + ' mer distribution')
    # axs.tick_params(axis='x', rotation=90)
    plt.savefig(output_dir + 'synthesis_distributions_' +  metric_name + '.png', bbox_inches='tight')