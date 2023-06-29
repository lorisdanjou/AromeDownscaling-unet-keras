import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs


def get_metric_tensor(eps, sca):
    
    """
    Compute the metric correlation tensor of a given field eps
    with a unit length scale sca
    Inputs : 
        
        eps : array of shape B X C X H x W
        sca : float
    Returns :
        
        g : array of shape 2 x 2 x C  x (H-1) x (W-1) 
    """
    
    C, H, W = eps.shape[1], eps.shape[2], eps.shape[3]
    
    d_eps_x = np.diff(eps, axis = 2)[:,:,:,1:]/sca
    d_eps_y = np.diff(eps, axis = 3)[:,:,1:,]/sca
    
    dx_dx = np.expand_dims(np.mean(d_eps_x * d_eps_x, axis = 0), axis = 0)
    dx_dy = np.expand_dims(np.mean(d_eps_x * d_eps_y, axis = 0), axis = 0)
    
    dx = np.concatenate((dx_dx, dx_dy) , axis =0)
    
    dy_dx = np.expand_dims(np.mean(d_eps_y * d_eps_x, axis = 0), axis = 0)
    dy_dy = np.expand_dims(np.mean(d_eps_y * d_eps_y, axis = 0), axis = 0)
    
    dy = np.concatenate((dy_dx, dy_dy) , axis =0)
    
    # print(dx.shape, dy.shape)
    g = np.concatenate((dx, dy), axis = 0)
    
    return g.reshape(2, 2, C, H-1, W-1)


def get_normalized_field(eps):
    """
    Normalizes a given field with respect to Batch and spatial dimensions
    Inputs :
        eps : array of shape B x C x H x W
        
    Returns : 
        array of shape B X C x H x W
    """
    sig = np.std(eps, axis = (0,2,3), keepdims = True)
    mean = np.mean(eps, axis = (0,2,3), keepdims = True)
    
    return (eps-mean)/sig 


def correlation_length(g, sca):
    """
    Give an estimate of the correlation length present in a metric tensor g
    with a given length scale sca
    Inputs :
        g : array of shape  2 x 2 x C x H x W
        sca : float
    Returns :
        ls : array of shape C x H x W
    """
    
    correl = 0.5*(np.trace(np.sqrt(np.abs(g))))
    
    ls = (1.0/correl)
    
    return ls


def length_scale(eps, sca = 1.0) :
    """
    Give an estimate of correlation length maps given a field eps and
    a scale sca
    Inputs :
        eps : array of shape B x C x H x W
        sca : float
    Returns :
        ls : array of shape C x H x W
    """
    
    eps_0 = get_normalized_field(eps)
    g = get_metric_tensor(eps_0, sca)
    ls = correlation_length(g, sca)
    
    return ls


# for a whole dataframe
def compute_corr_len(results_df):
    y_pred = np.zeros((len(results_df), 1, results_df.y_pred[0].shape[0], results_df.y_pred[0].shape[1]))
    y_test = np.zeros((len(results_df), 1, results_df.y_test[0].shape[0], results_df.y_test[0].shape[1]))
    baseline = np.zeros((len(results_df), 1, results_df.baseline[0].shape[0], results_df.baseline[0].shape[1]))

    for k in range(len(results_df)):
        y_test[k, 0, :, :] = results_df.y_test[k]
        y_pred[k, 0, :, :] = results_df.y_pred[k]
        baseline[k, 0, :, :] = results_df.baseline[k]

    corr_len_pred = length_scale(y_pred, sca=2.5)
    corr_len_test = length_scale(y_test, sca=2.5)
    corr_len_baseline = length_scale(baseline, sca=2.5)

    corr_len_df = pd.DataFrame(
        {'corr_len_test': [corr_len_test[0, :, :]],
        'corr_len_pred' : [corr_len_pred[0, :, :]],
        'corr_len_baseline' : [corr_len_baseline[0, :, :]]}
    )
    return corr_len_df


def plot_corr_len(corr_len_df, output_dir):
    corr_len_test     = corr_len_df['corr_len_test'].iloc[0]
    corr_len_pred     = corr_len_df['corr_len_pred'].iloc[0]
    corr_len_baseline = corr_len_df['corr_len_baseline'].iloc[0]
    fig = plt.figure(figsize=[25, 8])
    axs = []
    for j in range(3):
        axs.append(fig.add_subplot(1, 3, j+1, projection=ccrs.PlateCarree()))
        axs[j].set_extent(utils.IMG_EXTENT)
        axs[j].coastlines(resolution='10m', color='black', linewidth=1)

    data = [corr_len_baseline, corr_len_pred, corr_len_test]
    images = []
    for j in range(3):
        images.append(axs[j].imshow(data[j], cmap="viridis", origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
        axs[j].label_outer()
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    axs[0].set_title("fullpos", fontdict={"fontsize": 20})
    axs[1].set_title("Unet", fontdict={"fontsize": 20})
    axs[2].set_title("Arome500m", fontdict={"fontsize": 20})
    fig.colorbar(images[0], ax=axs, label="correlation length [km]")
    plt.savefig(output_dir + 'correlation_length_maps.png', bbox_inches='tight')


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
        
        corr_len_df = compute_corr_len(results_df)
        plot_corr_len(corr_len_df, opt["path"]["results"])
