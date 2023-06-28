from bronx.stdtypes.date import daterangex as rangex
import data.load_data as ld
import data.normalisations as norm
import data.patches as patches
import data.data_augmentation as da
import utils

def load_data(data_loading_opt):
    data_train_location  = data_loading_opt["data_train_location"]
    data_valid_location  = data_loading_opt["data_valid_location"]
    data_test_location   = data_loading_opt["data_test_location"]
    data_static_location = data_loading_opt["data_static_location"]
    dates_train          = rangex(data_loading_opt["dates_train"])
    dates_valid          = rangex(data_loading_opt["dates_valid"])
    dates_test           = rangex(data_loading_opt["dates_test"])
    echeances            = data_loading_opt["echeances"]
    params_in            = data_loading_opt["params_in"]
    params_out           = data_loading_opt["params_out"]
    static_fields        = data_loading_opt["static_fields"]
    interp               = data_loading_opt["interp"]

    if data_loading_opt["config"] == "optimisation": # the test dataset is not used
        from sklearn.model_selection import train_test_split   
        X_train_df = ld.load_X(
            dates_train, 
            echeances,
            params_in,
            data_train_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        X_test_df = ld.load_X(
            dates_valid, 
            echeances,
            params_in,
            data_valid_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        y_train_df = ld.load_y(
            dates_train,
            echeances,
            params_out,
            data_train_location
        )

        y_test_df = ld.load_y(
            dates_valid,
            echeances,
            params_out,
            data_valid_location
        )
        # split train set
        X_train_df, X_valid_df, y_train_df, y_valid_df = train_test_split(
            X_train_df, y_train_df, test_size=int(0.2*len(X_train_df)))

    elif data_loading_opt["config"] =="test": # the whole dataset is used
        X_train_df = ld.load_X(
            dates_train, 
            echeances,
            params_in,
            data_train_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        X_valid_df = ld.load_X(
            dates_valid, 
            echeances,
            params_in,
            data_valid_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        X_test_df = ld.load_X(
            dates_test, 
            echeances,
            params_in,
            data_test_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        y_train_df = ld.load_y(
            dates_train,
            echeances,
            params_out,
            data_train_location
        )

        y_valid_df = ld.load_y(
            dates_valid,
            echeances,
            params_out,
            data_valid_location
        )

        y_test_df = ld.load_y(
            dates_test,
            echeances,
            params_out,
            data_test_location
        )
    
    else:
        raise NotImplementedError

    return X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df

def  preprocess_data(preproc_opt, output_dir, X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df):

    # remove missing days
    X_train_df, y_train_df = ld.delete_missing_days(X_train_df, y_train_df)
    X_valid_df, y_valid_df = ld.delete_missing_days(X_valid_df, y_valid_df)
    X_test_df , y_test_df  = ld.delete_missing_days(X_test_df, y_test_df)

    # pad data (depends of the patching method)
    patches_opt = preproc_opt["patches"]
    if patches_opt["enable"]:
        if patches_opt["train_method"] == "random":
            # [X, y]_train don't need to be padded 
            if patches_opt["test_method"] is None:
                X_valid_df, y_valid_df = utils.pad(X_valid_df), utils.pad(y_valid_df)
                X_test_df , y_test_df  = utils.pad(X_test_df),  utils.pad(y_test_df)
            elif patches_opt["test_method"] == "patchify":
                X_valid_df, y_valid_df = patches.pad_for_patchify(X_valid_df), patches.pad_for_patchify(y_valid_df)
                X_test_df , y_test_df  = patches.pad_for_patchify(X_test_df) , patches.pad_for_patchify(y_test_df)

        elif patches_opt["train_method"] == "patchify":
            X_train_df, y_train_df = patches.pad_for_patchify(X_train_df), patches.pad_for_patchify(y_train_df)
            if patches_opt["test_method"] is None:
                X_valid_df, y_valid_df = utils.pad(X_valid_df), utils.pad(y_valid_df)
                X_test_df , y_test_df  = utils.pad(X_test_df),  utils.pad(y_test_df)
            elif patches_opt["test_method"] == "patchify":
                X_valid_df, y_valid_df = patches.pad_for_patchify(X_valid_df), patches.pad_for_patchify(y_valid_df)
                X_test_df , y_test_df  = patches.pad_for_patchify(X_test_df) , patches.pad_for_patchify(y_test_df)

    else:
        X_train_df, y_train_df = utils.pad(X_train_df), utils.pad(y_train_df)
        X_valid_df, y_valid_df = utils.pad(X_valid_df), utils.pad(y_valid_df)
        X_test_df , y_test_df  = utils.pad(X_test_df),  utils.pad(y_test_df)

    # normalisation:
    if preproc_opt["normalisation"] is not None:
        if preproc_opt["normalisation"] == "standardisation":
            norm.get_mean(X_train_df, output_dir)
            norm.get_std(X_train_df, output_dir)
            X_train_df, y_train_df = norm.standardisation(X_train_df, output_dir), norm.standardisation(y_train_df, output_dir)
            X_valid_df, y_valid_df = norm.standardisation(X_valid_df, output_dir), norm.standardisation(y_valid_df, output_dir)
            X_test_df , y_test_df  = norm.standardisation(X_test_df, output_dir) , norm.standardisation(y_test_df, output_dir)
        elif preproc_opt["normalisation"] == "normalisation":
            norm.get_max_abs(X_train_df, output_dir)
            X_train_df, y_train_df = norm.normalisation(X_train_df, output_dir), norm.normalisation(y_train_df, output_dir)
            X_valid_df, y_valid_df = norm.normalisation(X_valid_df, output_dir), norm.normalisation(y_valid_df, output_dir)
            X_test_df , y_test_df  = norm.normalisation(X_test_df, output_dir) , norm.normalisation(y_test_df, output_dir)
        elif preproc_opt["normalisation"] == "minmax":
            norm.get_min(X_train_df, output_dir)
            norm.get_max(X_train_df, output_dir)
            X_train_df, y_train_df = norm.min_max_norm(X_train_df, output_dir), norm.min_max_norm(y_train_df, output_dir)
            X_valid_df, y_valid_df = norm.min_max_norm(X_valid_df, output_dir), norm.min_max_norm(y_valid_df, output_dir)
            X_test_df , y_test_df  = norm.min_max_norm(X_test_df, output_dir) , norm.min_max_norm(y_test_df, output_dir)
        elif preproc_opt["normalisation"] == "mean":
            norm.get_min(X_train_df, output_dir)
            norm.get_max(X_train_df, output_dir)
            norm.get_mean(X_train_df, output_dir)
            X_train_df, y_train_df = norm.mean_norm(X_train_df, output_dir), norm.mean_norm(y_train_df, output_dir)
            X_valid_df, y_valid_df = norm.mean_norm(X_valid_df, output_dir), norm.mean_norm(y_valid_df, output_dir)
            X_test_df , y_test_df  = norm.mean_norm(X_test_df, output_dir) , norm.mean_norm(y_test_df, output_dir)
        else:
            raise NotImplementedError

    # if necessary, extract patches
    if patches_opt["enable"]:
        if patches_opt["train_method"] == "random":
            X_train_df, y_train_df = patches.extract_patches(X_train_df, y_train_df,
                patches_opt["patch_h"], patches_opt["patch_w"], patches_opt["n_patches"])
            if patches_opt["test_method"] == "patchify":
                X_valid_df = patches.extract_patches_patchify(X_valid_df, patches_opt["patch_size"])
                y_valid_df = patches.extract_patches_patchify(y_valid_df, patches_opt["patch_size"])
                X_test_df  = patches.extract_patches_patchify(X_test_df , patches_opt["patch_size"])
                y_test_df  = patches.extract_patches_patchify(y_test_df , patches_opt["patch_size"])

        elif patches_opt["train_method"] == "patchify":
            X_train_df = patches.extract_patches_patchify(X_train_df, patches_opt["patch_size"])
            y_train_df = patches.extract_patches_patchify(y_train_df, patches_opt["patch_size"])
            if patches_opt["test_method"] == "patchify":
                X_valid_df = patches.extract_patches_patchify(X_valid_df, patches_opt["patch_size"])
                y_valid_df = patches.extract_patches_patchify(y_valid_df, patches_opt["patch_size"])
                X_test_df  = patches.extract_patches_patchify(X_test_df , patches_opt["patch_size"])
                y_test_df  = patches.extract_patches_patchify(y_test_df , patches_opt["patch_size"])

    # Data augmentation
    da_opt = preproc_opt["data_augmentation"]
    if da_opt["enable_flip"]:
        X_train_df, y_train_df = da.random_flip(X_train_df, y_train_df, frac=da_opt["frac_flip"])
        if da_opt["enable_rot"]:
            X_train_df, y_train_df = da.random_rot(X_train_df, y_train_df, frac=da_opt["frac_flip"])

    return X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df


def postprocess_data(opt, y_pred_df):
    output_dir = opt["path"]["experiment"]

    # rebuild from patches
    patches_opt = opt["preprocessing"]["patches"]
    if patches_opt["enable"]:
        if patches_opt["test_method"] == "patchify":
            y_pred_df = patches.rebuild_from_patchify(y_pred_df, patches_opt["IMG_H"], patches_opt["IMG_W"])

    # denormalisation
    if opt["preprocessing"]["normalisation"] is not None:
        if opt["preprocessing"]["normalisation"] == "standardisation":
            y_pred_df = norm.destandardisation(y_pred_df, output_dir)
        elif opt["preprocessing"]["normalisation"] == "normalisation":
            y_pred_df = norm.denormalisation(y_pred_df, output_dir)
        elif opt["preprocessing"]["normalisation"] == "minmax":
            y_pred_df = norm.min_max_denorm(y_pred_df, output_dir)
        elif opt["preprocessing"]["normalisation"] == "mean":
            y_pred_df = norm.mean_denorm(y_pred_df, output_dir)

    # crop data
    if patches_opt["enable"]:
        if patches_opt["test_method"] == "patchify":
            y_pred_df = patches.crop_for_patchify(y_pred_df)
    else:
        y_pred_df = utils.crop(y_pred_df)

    return y_pred_df