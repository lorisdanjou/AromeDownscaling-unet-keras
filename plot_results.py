import os
import argparse
import core.logger as logger

import results.load_results as lr
import results.pointwise_scores as ps
import results.WD as wd
import results.PSD as psd
import results.correlation as corr
import results.correlation_length as corr_len

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
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
        lr.plot_results(
            results_df,
            opt["path"]["results"],
            param=param,
            unit=opt["results"]["images"]["units"][i_p],
            cmap=opt["results"]["images"]["cmap"][i_p],
            n=opt["results"]["images"]["n"]
        )

        # pointwise scores
        pointwise_opt = opt["results"]["pointwise_scores"]
        if pointwise_opt["mae"]["enable"]:
            mae_df = ps.compute_score(results_df, ps.mae, "MAE")
            if pointwise_opt["mae"]["mode"] == "all":
                ps.plot_score_maps(
                    mae_df, 
                    output_dir=opt["path"]["results"], 
                    metric_name="MAE", 
                    unit=pointwise_opt["mae"]["units"][i_p], 
                    cmap=pointwise_opt["mae"]["cmap"]
                )
            elif pointwise_opt["mae"]["mode"] == "unique":
                ps.plot_unique_score_map(
                    mae_df, 
                    output_dir=opt["path"]["results"], 
                    metric_name="MAE", 
                    unit=pointwise_opt["mae"]["units"][i_p], 
                    cmap=pointwise_opt["mae"]["cmap"]
                )
            ps.plot_distrib(mae_df, "MAE", opt["path"]["results"])
        
        if pointwise_opt["mse"]["enable"]:
            mse_df = ps.compute_score(results_df, ps.mse, "MSE")
            if pointwise_opt["mse"]["mode"] == "all":
                ps.plot_score_maps(
                    mse_df, 
                    output_dir=opt["path"]["results"], 
                    metric_name="MSE", 
                    unit=pointwise_opt["mse"]["units"][i_p], 
                    cmap=pointwise_opt["mse"]["cmap"]
                )
            elif pointwise_opt["mse"]["mode"] == "unique":
                ps.plot_unique_score_map(
                    mse_df, 
                    output_dir=opt["path"]["results"], 
                    metric_name="MSE", 
                    unit=pointwise_opt["mse"]["units"][i_p], 
                    cmap=pointwise_opt["mse"]["cmap"]
                )
            ps.plot_distrib(mse_df, "MSE", opt["path"]["results"])

        if pointwise_opt["bias"]["enable"]:
            bias_df = ps.compute_score(results_df, ps.bias, "bias")
            if pointwise_opt["bias"]["mode"] == "all":
                ps.plot_score_maps(
                    bias_df, 
                    output_dir=opt["path"]["results"], 
                    metric_name="bias", 
                    unit=pointwise_opt["bias"]["units"][i_p], 
                    cmap=pointwise_opt["bias"]["cmap"]
                )
            elif pointwise_opt["mse"]["mode"] == "unique":
                ps.plot_unique_score_map(
                    bias_df, 
                    output_dir=opt["path"]["results"], 
                    metric_name="bias", 
                    unit=pointwise_opt["bias"]["units"][i_p], 
                    cmap=pointwise_opt["bias"]["cmap"]
                )
            ps.plot_distrib(bias_df, "bias", opt["path"]["results"])

        if pointwise_opt["ssim"]["enable"]:
            ssim_df = ps.compute_score(results_df, ps.ssim, "SSIM")
            if pointwise_opt["ssim"]["mode"] == "all":
                ps.plot_score_maps(
                    ssim_df, 
                    output_dir=opt["path"]["results"], 
                    metric_name="SSIM", 
                    unit=pointwise_opt["ssim"]["units"][i_p], 
                    cmap=pointwise_opt["ssim"]["cmap"]
                )
            elif pointwise_opt["ssim"]["mode"] == "unique":
                ps.plot_unique_score_map(
                    ssim_df, 
                    output_dir=opt["path"]["results"], 
                    metric_name="SSIM", 
                    unit=pointwise_opt["ssim"]["units"][i_p], 
                    cmap=pointwise_opt["ssim"]["cmap"]
                )
            ps.plot_distrib(ssim_df, "SSIM", opt["path"]["results"])

        # WD
        if opt["results"]["WD"]["enable"]:
            wd_df       = wd.compute_datewise_WD(results_df)
            wd_df_terre = wd.compute_datewise_WD_terre(results_df)
            wd_df_mer   = wd.compute_datewise_WD_mer(results_df)
            wd.plot_datewise_wasserstein_distance_distrib(wd_df, wd_df_terre, wd_df_mer, opt["path"]["results"])

        # PSD
        if opt["results"]["PSD"]["enable"]:
            psd_df = psd.PSD(results_df)
            psd.plot_PSDs(psd_df, opt["path"]["results"])

        # correlation
        if opt["results"]["correlation"]["enable"]:
            corr_df       = corr.correlation(results_df)
            corr_df_terre = corr.correlation_terre(results_df)
            corr_df_mer   = corr.correlation_mer(results_df) 
            corr.plot_corr_distrib(corr_df, corr_df_terre, corr_df_mer, opt["path"]["results"])  

        # correlation_length
        if opt["results"]["correlation_length"]["enable"]:
            corr_len_df = corr_len.compute_corr_len(results_df)
            corr_len.plot_corr_len(corr_len_df, opt["path"]["results"])

