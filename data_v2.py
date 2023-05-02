import numpy as np
from os.path import exists
import pandas as pd
from bronx.stdtypes.date import daterangex as rangex


"""
Useful functions to get the shape of a domain
"""
def highestPowerof2(n):
    res = 0
    for i in range(n, 0, -1):
        # If i is a power of 2
        if ((i & (i - 1)) == 0):
            res = i
            break
    return res

    
def get_shape_500m():
    field_500m = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/G9KP_2021-01-01T00:00:00Z_rr.npy')
    return field_500m[:, :, 0].shape


def get_highestPowerof2_500m():
    field_500m = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/G9KP_2021-01-01T00:00:00Z_rr.npy')
    geometry_500m = field_500m[:, :, 0].shape
    size_500m = min(geometry_500m)
    size_500m_crop = highestPowerof2(min(geometry_500m))
    return size_500m, size_500m_crop


def get_shape_2km5(resample='c'):
    if resample == 'c':
        field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_c_2021-01-01T00:00:00Z_rr.npy')
        return field_2km5[:, :, 0].shape
    else:
        field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_r_2021-01-01T00:00:00Z_rr.npy')
        return field_2km5[:, :, 0].shape


def get_highestPowerof2_2km5(resample='c'):
    if resample == 'c':
        field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_c_2021-01-01T00:00:00Z_rr.npy')
        geometry_2km5 = field_2km5[:, :, 0].shape
        size_2km5 = min(geometry_2km5)
        size_2km5_crop = highestPowerof2(min(geometry_2km5))
        return size_2km5, size_2km5_crop
    else:
        field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_r_2021-01-01T00:00:00Z_rr.npy')
        geometry_2km5 = field_2km5[:, :, 0].shape
        size_2km5 = min(geometry_2km5)
        size_2km5_crop = highestPowerof2(min(geometry_2km5))
        return size_2km5, size_2km5_crop


"""
Load dataset
"""
def load_X(dates, echeances, params, data_location, data_static_location='', static_fields=[], resample='r'):
    """
    Loads all the inputs of the NN in a pandas dataframe
    Inputs:
    Outputs : pansdas dataframe
    """
    data = pd.DataFrame(
        [], 
        columns = ['dates', 'echeances'] + params
    )
    domain_shape_in  = get_shape_2km5(resample=resample)
    domain_shape_out = get_shape_500m()
    for i_d, d in enumerate(dates):
        # chargement des données
        X_d = np.zeros([len(echeances), domain_shape_in[0], domain_shape_in[1], len(params) + len(static_fields)], dtype=np.float32)
        try:
            for i_p, p in enumerate(params):
                if resample == 'c':
                    filepath_X = data_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
                    X_d[:, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
                else:
                    filepath_X = data_location + 'oper_r_' + d.isoformat() + 'Z_' + p + '.npy'
                    X_d[:, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
        # champs statiques
            for i_s, s in enumerate(static_fields):
                if resample == 'r':
                    filepath_static = data_static_location + 'static_G9KP_' + s + '.npy'
                    # filepath_static = data_static_location + 'static_oper_r_' + s + '.npy'
                else:
                    filepath_static = data_static_location + 'static_oper_c_' + s + '.npy'
                for i_ech, ech in enumerate(echeances):
                    X_d[i_ech, :, :, len(params) + i_s] = np.load(filepath_static)
        except FileNotFoundError:
            print('missing day (X): ' + d.isoformat())
            X_d = None
        
        for i_ech, ech in enumerate(echeances):
            values_i = []
            try:
                for i_p, p in enumerate(params + static_fields):
                    values_i.append(X_d[i_ech, :, :, i_p])
                data_i = pd.DataFrame(
                    [[d.isoformat(), ech] + values_i], 
                    columns = ['dates', 'echeances'] + params + static_fields
                )
            except TypeError:
                for i_p, p in enumerate(params + static_fields):
                    values_i.append(None)
                data_i = pd.DataFrame(
                    [[d.isoformat(), ech] + values_i], 
                    columns = ['dates', 'echeances'] + params + static_fields
                )

            data = pd.concat([data, data_i])
    return data.reset_index(drop=True)


def load_y(dates, echeances, params, data_location):
    """
    Loads all the outputs of the NN in a pandas dataframe
    Inputs:
    Outputs : pansdas dataframe
    """
    data = pd.DataFrame(
        [], 
        columns = ['dates', 'echeances'] + params
    )
    domain_shape_out = get_shape_500m()
    for i_d, d in enumerate(dates):
        # chargement des données
        y_d = np.zeros([len(echeances), domain_shape_out[0], domain_shape_out[1], len(params)], dtype=np.float32)
        try:
            for i_p, p in enumerate(params):
                filepath_y = data_location + 'G9L1_' + d.isoformat() + 'Z_' + p + '.npy'
                if exists(filepath_y):
                    y_d[:, :, :, i_p] = np.load(filepath_y).transpose([2, 0, 1])
                else:
                    filepath_y = data_location + 'G9KP_' + d.isoformat() + 'Z_' + p + '.npy'
                    y_d[:, :, :, i_p] = np.load(filepath_y).transpose([2, 0, 1])
        except FileNotFoundError:
            print('missing day (y): ' + d.isoformat())
            y_d = None
        
        # création du dataframe
        for i_ech, ech in enumerate(echeances):
            values_i = []
            try:
                for i_p, p in enumerate(params):
                    values_i.append(y_d[i_ech, :, :, i_p])
                data_i = pd.DataFrame(
                    [[d.isoformat(), ech] + values_i], 
                    columns = ['dates', 'echeances'] + params
                )
            except TypeError:
                for i_p, p in enumerate(params):
                    values_i.append(None)
                data_i = pd.DataFrame(
                    [[d.isoformat(), ech] + values_i], 
                    columns = ['dates', 'echeances'] + params
                )

            data = pd.concat([data, data_i])
    return data.reset_index(drop=True)


def delete_missing_days(X_df, y_df):
    """
    Deletes all the missing days in both X and y
    Inputs :
        X_df : a pandas dataframe representing X 
        y_df : a pandas dataframe representing y 
    Outputs:
        X_df : a pandas dataframe representing X without missing days
        y_df : a pandas dataframe representing y without missing days
    """
    X_df_out = X_df.copy()
    y_df_out = y_df.copy()
    nan_indices_y = y_df[y_df.isna().any(axis=1)].index
    y_df_out = y_df_out.drop(index=nan_indices_y, axis = 0)
    X_df_out = X_df_out.drop(index=nan_indices_y, axis = 0)
    nan_indices_X = X_df[X_df.isna().any(axis=1)].index
    X_df_out = X_df_out.drop(index=nan_indices_X, axis = 0)
    y_df_out = y_df_out.drop(index=nan_indices_X, axis = 0)

    return X_df_out.reset_index(drop=True), y_df_out.reset_index(drop=True)


def get_arrays_cols(df):
    arrays_cols = []
    for c in df.columns:
        if type(df[c][0]) == np.ndarray:
            arrays_cols.append(c)
    return arrays_cols


def pad(df):
    """
    Pad the data in order to make its size compatible with the unet
    Input : dataframe
    Output : dataframe with padding
    """
    df_out = df.copy()
    arrays_cols = get_arrays_cols(df_out)
    for c in arrays_cols:
        for i in range(len(df_out)):
            df_out[c][i] = np.pad(df_out[c][i], ((5,5), (2,3)), mode='reflect')
    return df_out


def crop(df):
    """
    Crop the data
    Input : dataframe
    Output : cropped dataframe
    """
    df_out = df.copy()
    for c in df_out.columns:
        if type(df_out[c][0]) == np.ndarray:
            for i in range(len(df_out)):
                df_out[c][i] = df_out[c][i][5:-5, 2:-3]
    return df_out


def param_to_array(arrays_serie):
    """
    Transforms a pandas series into a big numpy array of shape B x H x W
    """
    array = np.zeros((len(arrays_serie), arrays_serie[0].shape[0], arrays_serie[0].shape[1]), dtype=np.float32)
    for i in range(len(arrays_serie)):
        array[i, :, :] = arrays_serie[i]
    return array


def df_to_array(df):
    """
    transforms a pandas dataframe into a big numpy array of shape B x H x W x C
    """
    arrays_cols = get_arrays_cols(df)
            
    array = np.zeros((len(df), df[arrays_cols[0]][0].shape[0], df[arrays_cols[0]][0].shape[1], len(arrays_cols)), dtype=np.float32)

    for i in range(len(df)):
        for i_c, c in enumerate(arrays_cols):
            array[i, :, :, i_c] = df[c][i]
    return array


###############################################################################################################################
# Normalisations
###############################################################################################################################

# Normalisation between -1 and 1
def get_max_abs(X_df, working_dir):
    """
    Get the absolute maximun of the columns containing arrays of X
    ! the channels of X must begin by the channels of y (in the same order)
    Input : X dataframe
    Output: List of absolute maximums 
    """
    max_abs_out = []
    arrays_cols = get_arrays_cols(X_df)
    for i_c, c in enumerate(arrays_cols):
        X_c = param_to_array(X_df[c])
        max_abs_out.append(np.abs(X_c).max())
    np.save(working_dir + 'max_abs_X.npy', max_abs_out, allow_pickle=True)

def normalisation(df, working_dir):
    """
    Normalise a dataframe between -1 and 1
    Input : X or y dataframe
    Output : normalised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    max_abs_X = np.load(working_dir + 'max_abs_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            df_norm[c][i] = df_norm[c][i] / max_abs_X[i_c]
    return df_norm

def denormalisation(df, working_dir):
    """
    Denormalise a dataframe
    Input : X or y dataframe
    Output : denormalised copy of the dataframe
    """
    df_den = df.copy()
    arrays_cols = get_arrays_cols(df_den)
    max_abs_X = np.load(working_dir + 'max_abs_X.npy')
    for i in range(len(df_den)):
        for i_c, c in enumerate(arrays_cols):
            df_den[c][i] = df_den[c][i] * max_abs_X[i_c]
    return df_den


# Standardisation
def get_mean(X_df, working_dir):
    """
    Get the mean of the columns containing arrays of X
    ! the channels of X must begin by the channels of y (in the same order)
    Input : X dataframe
    Output: List of means
    """
    mean_out_X = []
    arrays_cols_X = get_arrays_cols(X_df)
    for i_c, c in enumerate(arrays_cols_X):
        X_c = param_to_array(X_df[c])
        mean_out_X.append(X_c.mean())
    np.save(working_dir + 'mean_X.npy', mean_out_X, allow_pickle=True)


def get_std(X_df, working_dir):
    """
    Get the standard deviations of the columns containing arrays of X
    ! the channels of X must begin by the channels of y (in the same order)
    Input : X dataframe
    Output: List of standard deviations
    """
    std_out_X = []
    arrays_cols_X = get_arrays_cols(X_df)
    for i_c, c in enumerate(arrays_cols_X):
        X_c = param_to_array(X_df[c])
        std_out_X.append(X_c.std())
    np.save(working_dir + 'std_X.npy', std_out_X, allow_pickle=True)


def standardisation(X_df, working_dir):
    """
    Standardise a dataframe (* - mean) / std
    Input : X or y dataframe
    Output : standardised copy of the dataframe
    """
    X_df_norm = X_df.copy()
    arrays_cols = get_arrays_cols(X_df_norm)
    mean_X = np.load(working_dir + 'mean_X.npy')
    std_X = np.load(working_dir + 'std_X.npy')
    for i in range(len(X_df_norm)):
        for i_c, c in enumerate(arrays_cols):
            if std_X[i_c] < 1e-9:
                raise ValueError('std = 0') 
            X_df_norm[c][i] = (X_df_norm[c][i] - mean_X[i_c]) / std_X[i_c]
    return X_df_norm


def destandardisation(df, working_dir):
    """
    Destandardise a dataframe
    Input : X or y dataframe
    Output : destandardised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    mean_X = np.load(working_dir + 'mean_X.npy')
    std_X = np.load(working_dir + 'std_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            df_norm[c][i] = (df_norm[c][i] * std_X[i_c]) + mean_X[i_c]
    return df_norm


# def get_mean_both(X_df, y_df, working_dir):
#     mean_out_X = []
#     arrays_cols_X = get_arrays_cols(X_df)
#     for i_c, c in enumerate(arrays_cols_X):
#         X_c = param_to_array(X_df[c])
#         mean_out_X.append(X_c.mean())
#     np.save(working_dir + 'mean_X.npy', mean_out_X, allow_pickle=True)
#     mean_out_y = []
#     arrays_cols_y = get_arrays_cols(y_df)
#     for i_c, c in enumerate(arrays_cols_y):
#         y_c = param_to_array(y_df[c])
#         mean_out_y.append(y_c.mean())
#     np.save(working_dir + 'mean_y.npy', mean_out_y, allow_pickle=True)


# def get_std_both(X_df, y_df, working_dir):
#     std_out_X = []
#     arrays_cols_X = get_arrays_cols(X_df)
#     for i_c, c in enumerate(arrays_cols_X):
#         X_c = param_to_array(X_df[c])
#         std_out_X.append(X_c.std())
#     np.save(working_dir + 'std_X.npy', std_out_X, allow_pickle=True)
#     std_out_y = []
#     arrays_cols_y = get_arrays_cols(y_df)
#     for i_c, c in enumerate(arrays_cols_y):
#         y_c = param_to_array(y_df[c])
#         std_out_y.append(y_c.std())
#     np.save(working_dir + 'std_y.npy', std_out_y, allow_pickle=True)


# def standardisation_both(X_df, y_df, working_dir):
#     X_df_norm = X_df.copy()
#     arrays_cols_X = get_arrays_cols(X_df_norm)
#     mean_X = np.load(working_dir + 'mean_X.npy')
#     std_X = np.load(working_dir + 'std_X.npy')
#     for i in range(len(X_df_norm)):
#         for i_c, c in enumerate(arrays_cols_X):
#             if std_X[i_c] < 1e-9:
#                 raise ValueError('std = 0') 
#             X_df_norm[c][i] = (X_df_norm[c][i] - mean_X[i_c]) / std_X[i_c]
#     y_df_norm = y_df.copy()
#     arrays_cols_y = get_arrays_cols(y_df_norm)
#     mean_y = np.load(working_dir + 'mean_y.npy')
#     std_y = np.load(working_dir + 'std_y.npy')
#     for i in range(len(y_df_norm)):
#         for i_c, c in enumerate(arrays_cols_y):
#             if std_y[i_c] < 1e-9:
#                 raise ValueError('std = 0') 
#             y_df_norm[c][i] = (y_df_norm[c][i] - mean_y[i_c]) / std_y[i_c]
#     return X_df_norm, y_df_norm


# def destandardisation_both(X_df, y_df, working_dir):
#     X_df_norm = X_df.copy()
#     arrays_cols_X = get_arrays_cols(X_df_norm)
#     mean_X = np.load(working_dir + 'mean_X.npy')
#     std_X = np.load(working_dir + 'std_X.npy')
#     for i in range(len(X_df_norm)):
#         for i_c, c in enumerate(arrays_cols_X):
#             X_df_norm[c][i] = (X_df_norm[c][i] * std_X[i_c]) + mean_X[i_c]
#     y_df_norm = y_df.copy()
#     arrays_cols_y = get_arrays_cols(y_df_norm)
#     mean_y = np.load(working_dir + 'mean_y.npy')
#     std_y = np.load(working_dir + 'std_y.npy')
#     for i in range(len(y_df_norm)):
#         for i_c, c in enumerate(arrays_cols_y):
#             y_df_norm[c][i] = (y_df_norm[c][i] * std_y[i_c]) + mean_y[i_c]
#     return X_df_norm, y_df_norm


# def get_mean_df(df):
#     mean_out = []
#     arrays_cols = get_arrays_cols(df)
#     for i_c, c in enumerate(arrays_cols):
#         array_c = param_to_array(df[c])
#         mean_out.append(array_c.mean())
#     return mean_out


# def get_std_df(df):
#     std_out = []
#     arrays_cols = get_arrays_cols(df)
#     for i_c, c in enumerate(arrays_cols):
#         array_c = param_to_array(df[c])
#         std_out.append(array_c.std())
#     return std_out


# def standardisation_df(df, mean, std):
#     df_norm = df.copy()
#     arrays_cols = get_arrays_cols(df_norm)
#     for i in range(len(df_norm)):
#         for i_c, c in enumerate(arrays_cols):
#             if std[i_c] < 1e-9:
#                 raise ValueError('std = 0') 
#             df_norm[c][i] = (df_norm[c][i] - mean[i_c]) / std[i_c]
#     return df_norm


# def destandardisation_df(df, mean, std):
#     df_norm = df.copy()
#     arrays_cols = get_arrays_cols(df_norm)
#     for i in range(len(df_norm)):
#         for i_c, c in enumerate(arrays_cols):
#             if std[i_c] < 1e-9:
#                 raise ValueError('std = 0') 
#             df_norm[c][i] = df_norm[c][i] * std[i_c] + mean[i_c]
#     return df_norm


def standardisation_sample(df):
    df_norm = df.copy()
    columns_mean_std = []
    arrays_cols = get_arrays_cols(df_norm)
    for i_c, c in enumerate(arrays_cols):
        columns_mean_std.append('mean_' + c)
        columns_mean_std.append('std_' + c)
    df_mean_std = pd.DataFrame(
        [], 
        columns = columns_mean_std
    )
    for i in range(len(df_norm)):
        means_stds = []
        for i_c, c in enumerate(arrays_cols):
            mean_c = df_norm[c][i].mean()
            std_c  = df_norm[c][i].std()
            print(std_c)
            means_stds.append(mean_c)
            means_stds.append(std_c)
            df_norm[c][i] = (df_norm[c][i] - mean_c) / std_c
        df_mean_std_i = pd.DataFrame(
            [means_stds], 
            columns = columns_mean_std
        )
        df_mean_std = pd.concat([df_mean_std, df_mean_std_i])    
    df_mean_std = df_mean_std.reset_index(drop=True)
    df_norm = pd.concat([df_norm, df_mean_std], axis=1)
    return df_norm

def destandardisation_sample(df):
    df_den = df.copy()
    arrays_cols = get_arrays_cols(df_den)
    for i in range(len(df_den)):
        for i_c, c in enumerate(arrays_cols):
            mean_c = df_den['mean_' + c][i]
            std_c  = df_den['std_' + c][i]
            df_den[c][i] = df_den[c][i] * std_c + mean_c 
    cols_to_drop = []
    for c in arrays_cols:
        cols_to_drop.append('mean_' + c)
        cols_to_drop.append('std_' + c)
    return df_den.drop(cols_to_drop, axis=1)


# MinMax normalisation
def get_min(X_df, working_dir):
    """
    Get the min of the columns containing arrays of X
    ! the channels of X must begin by the channels of y (in the same order)
    Input : X dataframe
    Output: List of mins
    """
    min_out = []
    arrays_cols = get_arrays_cols(X_df)
    for i_c, c in enumerate(arrays_cols):
        X_c = param_to_array(X_df[c])
        min_out.append(X_c.min())
    np.save(working_dir + 'min_X.npy', min_out, allow_pickle=True)


def get_max(X_df, working_dir):
    """
    Get the max of the columns containing arrays of X
    ! the channels of X must begin by the channels of y (in the same order)
    Input : X dataframe
    Output: List of maxs
    """
    max_out = []
    arrays_cols = get_arrays_cols(X_df)
    for i_c, c in enumerate(arrays_cols):
        X_c = param_to_array(X_df[c])
        max_out.append(X_c.max())
    np.save(working_dir + 'max_X.npy', max_out, allow_pickle=True)


def min_max_norm(df, working_dir):
    """
    Normalise a dataframe
    Input : X or y dataframe
    Output : min-max normalised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    min_X = np.load(working_dir + 'min_X.npy')
    max_X = np.load(working_dir + 'max_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            if (max_X[i_c] - min_X[i_c]) < 1e-9:
                raise ValueError('min - max = 0') 
            df_norm[c][i] = (df_norm[c][i] - min_X[i_c]) / (max_X[i_c] - min_X[i_c])
    return df_norm


def min_max_denorm(df, working_dir):
    """
    Destandardise a dataframe
    Input : X or y dataframe
    Output : min-max denormalised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    min_X = np.load(working_dir + 'min_X.npy')
    max_X = np.load(working_dir + 'max_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            df_norm[c][i] = df_norm[c][i]  * (max_X[i_c] - min_X[i_c]) + min_X[i_c]
    return df_norm


# Mean normalisation

def mean_norm(df, working_dir):
    """
    Normalise a dataframe
    Input : X or y dataframe
    Output : mean normalised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    min_X = np.load(working_dir + 'min_X.npy')
    max_X = np.load(working_dir + 'max_X.npy')
    mean_X =np.load(working_dir + 'mean_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            if (max_X[i_c] - min_X[i_c]) < 1e-9:
                raise ValueError('min - max = 0') 
            df_norm[c][i] = (df_norm[c][i] - mean_X[i_c]) / (max_X[i_c] - min_X[i_c])
    return df_norm


def mean_denorm(df, working_dir):
    """
    Destandardise a dataframe
    Input : X or y dataframe
    Output : mean denormalised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    min_X = np.load(working_dir + 'min_X.npy')
    max_X = np.load(working_dir + 'max_X.npy')
    mean_X =np.load(working_dir + 'mean_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            df_norm[c][i] = df_norm[c][i]  * (max_X[i_c] - min_X[i_c]) + mean_X[i_c]
    return df_norm



###############################################################################################################################
# OLD
###############################################################################################################################

# def load_data(dates, echeances, params_in, params_out, data_location, data_static_location='', static_fields=[], resample='r'):
#     """
#     Creates a pandas dataframe containing the fields : dates, echeances, X and y
#     Inputs :
#     Output : A pandas dataframe with length = number of samples (dates * echeances)
#                 containing dates, echeances, X (size B x H x W x params_in) and y (size B x H x W x params_out))
#     """
#     data = pd.DataFrame(
#         {'dates' : [],
#         'echeances' : [],
#         'X' : [],
#         'y' : []}
#     )

#     domain_shape_in  = get_shape_2km5(resample=resample)
#     domain_shape_out = get_shape_500m()
#     for i_d, d in enumerate(dates):
#         # load X
#         X_i = np.zeros([len(echeances), domain_shape_in[0], domain_shape_in[1], len(params_in) + len(static_fields)], dtype=np.float32)
#         try:
#             for i_p, p in enumerate(params_in):
#                 if resample == 'c':
#                     filepath_X = data_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
#                     X_i[:, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
#                 else:
#                     filepath_X = data_location + 'oper_r_' + d.isoformat() + 'Z_' + p + '.npy'
#                     X_i[:, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
#         # load X static
#             for i_s, s in enumerate(static_fields):
#                 if resample == 'r':
#                     filepath_static = data_static_location + 'static_G9KP_' + s + '.npy'
#                     # filepath_static = data_static_location + 'static_oper_r_' + s + '.npy'
#                 else:
#                     filepath_static = data_static_location + 'static_oper_c_' + s + '.npy'
#                 for i_ech, ech in enumerate(echeances):
#                     X_i[i_ech, :, :, len(params_in) + i_s] = np.load(filepath_static)
#         except FileNotFoundError:
#             print('missing day (X): ' + d.isoformat())
#             X_i = None

#         # load y
#         y_i = np.zeros([len(echeances), domain_shape_out[0], domain_shape_out[1], len(params_out)], dtype=np.float32)
#         try:
#             for i_p, p in enumerate(params_out):
#                 filepath_y = data_location + 'G9L1_' + d.isoformat() + 'Z_' + p + '.npy'
#                 if exists(filepath_y):
#                     y_i[:, :, :, i_p] = np.load(filepath_y).transpose([2, 0, 1])
#                 else:
#                     filepath_y = data_location + 'G9KP_' + d.isoformat() + 'Z_' + p + '.npy'
#                     y_i[:, :, :, i_p] = np.load(filepath_y).transpose([2, 0, 1])
#         except FileNotFoundError:
#             print('missing day (y): ' + d.isoformat())
#             y_i = None

#         for i_ech, ech in enumerate(echeances):
#             try:
#                 data_i = pd.DataFrame(
#                     {'dates' : [d.isoformat()],
#                     'echeances' : [ech],
#                     'X' : [X_i[i_ech, :, :, :]],
#                     'y' : [y_i[i_ech, :, :, :]]}
#                 )
#             except TypeError:
#                 data_i = pd.DataFrame(
#                     {'dates' : [],
#                     'echeances' : [],
#                     'X' : [],
#                     'y' : []}
#                 )
#             data = pd.concat([data, data_i])
#     return data.reset_index(drop=True)


# def to_array(arrays_serie):
#     """
#     Transforms a pandas dataframe previously loaded in a big numpy array
#     Input : a pandas dataframe
#     output : array of size B x H x W x C
#     """
#     output = np.zeros([len(arrays_serie), arrays_serie[0].shape[0], arrays_serie[0].shape[1], arrays_serie[0].shape[2]], dtype = np.float32)
#     for i in range(len(arrays_serie)):
#         output[i, :, :, :] = arrays_serie[i]
#     return output


# """
# Preprocessing functions
# """
# def pad(data):
#     """
#     Pad the data in order to make its size compatible with the unet
#     Input : dataframe
#     Output : dataframe with padding
#     """
#     data_pad = data.copy()
#     for i in range(len(data)):
#         data_pad.X[i] = np.pad(data.X[i], ((5,5), (2,3), (0,0)), mode='reflect')
#         data_pad.y[i] = np.pad(data.y[i], ((5,5), (2,3), (0,0)), mode='reflect')
#     return data_pad


# def crop(data):
#     """
#     Reverse operation of paddind
#     Input : dataframe
#     output : dataframe
#     """
#     data_crop = data.copy()
#     for i in range(len(data)):
#         data_crop.X[i] = data.X[i][5:-5, 2:-3, :]
#         data_crop.y[i] = data.y[i][5:-5, 2:-3, :]
#     return data_crop


# """
# Global Normalisations
# """
# def get_max_abs(data, working_dir):
#     data_norm = data.copy() # ? Copie en mémoire non nécessaire (idem pour les fonctions get ... suivantes)
#     X = to_array(data.X)
#     y = to_array(data.y)
#     max_abs_X = np.abs(X, axis=(0, 1, 2), dtype=np.float64).max(axis=(0, 1, 2), dtype=np.float64)
#     max_abs_y = np.abs(y, axis=(0, 1, 2), dtype=np.float64).max(axis=(0, 1, 2), dtype=np.float64)
#     np.save(working_dir + 'max_abs_X.npy', max_abs_X)
#     np.save(working_dir + 'max_abs_y.npy', max_abs_y)

#     # # find the max in each channel of each array :
#     # max_abs_X = data_norm.X.apply(lambda x: np.abs(x).max(axis=(0, 1))).mean()
#     # max_abs_y = data_norm.y.apply(lambda x: np.abs(x).max(axis=(0, 1))).mean()
#     # np.save(working_dir + 'max_abs_X.npy', max_abs_X)
#     # np.save(working_dir + 'max_abs_y.npy', max_abs_y)


# def get_mean(data, working_dir):
#     data_norm = data.copy() 
#     mean_X = data_norm.X.apply(lambda x: x.mean(axis=(0, 1))).mean()
#     mean_y = data_norm.y.apply(lambda x: x.mean(axis=(0, 1))).mean()
#     np.save(working_dir + 'mean_X.npy', mean_X)
#     np.save(working_dir + 'mean_y.npy', mean_y)


# def get_mean(data, working_dir):
#     data_norm = data.copy()
#     std_X = data_norm.X.apply(lambda x: x.std(axis=(0, 1))).mean()
#     std_y = data_norm.y.apply(lambda x: x.std(axis=(0, 1))).mean()
#     np.save(working_dir + 'std_X.npy', std_X)
#     np.save(working_dir + 'std_y.npy', std_y)


# def get_min(data, working_dir):
#     data_norm = data.copy()
#     min_X = data_norm.X.apply(lambda x: x.min(axis=(0, 1))).mean()
#     min_y = data_norm.y.apply(lambda x: x.min(axis=(0, 1))).mean()
#     np.save(working_dir + 'min_X.npy', min_X)
#     np.save(working_dir + 'min_y.npy', min_y)


# def get_max(data, working_dir):
#     data_norm = data.copy()
#     max_X = data_norm.X.apply(lambda x: x.max(axis=(0, 1))).mean()
#     max_y = data_norm.y.apply(lambda x: x.max(axis=(0, 1))).mean()
#     np.save(working_dir + 'max_X.npy', max_X)
#     np.save(working_dir + 'max_y.npy', max_y)


# def global_normalisation(data, working_dir):
#     data_norm = data.copy()
#     max_abs_X = np.load(working_dir + 'max_abs_X.npy')
#     max_abs_y = np.load(working_dir + 'max_abs_y.npy')

#     # normalize the imputs :
#     for i in range(len(data_norm)):
#         for i_cx in range(data_norm.X[0].shape[2]):
#             data_norm.X[i][:, :, i_cx] = data_norm.X[i][:, :, i_cx] / max_abs_X[i_cx]
#         for i_cy in range(data_norm.y[0].shape[2]):
#             data_norm.y[i][:, :, i_cy] = data_norm.y[i][:, :, i_cy] / max_abs_y[i_cy]
#     return data_norm


# def global_denormalisation(data, working_dir):
#     data_den = data.copy()
#     max_abs_X = np.load(working_dir + 'max_abs_X.npy')
#     max_abs_y = np.load(working_dir + 'max_abs_y.npy')
#     for i in range(len(data_den)):
#         for i_cx in range(data_den.X[0].shape[2]):
#             data_den.X[i][:, :, i_cx] = data_den.X[i][:, :, i_cx] * max_abs_X[i_cx]
#         for i_cy in range(data_den.y[0].shape[2]):
#             data_den.y[i][:, :, i_cy] = data_den.y[i][:, :, i_cy] * max_abs_y[i_cy]
#     return data_den


# def global_standardisation(data, working_dir):
#     data_norm = data.copy()
#     mean_X = np.load(working_dir + 'mean_X.npy')
#     mean_y = np.load(working_dir + 'mean_y.npy')
#     std_X  = np.load(working_dir + 'std_X.npy')
#     std_y  = np.load(working_dir + 'std_y.npy')

#     for i in range(len(data_norm)):
#         for i_cx in range(data_norm.X[0].shape[2]):
#             data_norm.X[i][:, :, i_cx] = (data_norm.X[i][:, :, i_cx] - mean_X) / std_X[i_cx]
#         for i_cy in range(data_norm.y[0].shape[2]):
#             data_norm.y[i][:, :, i_cy] = (data_norm.y[i][:, :, i_cy] - mean_X) / std_y[i_cy]
#     return data_norm


# def global_destandardisation(data, working_dir):
#     data_norm = data.copy()
#     mean_X = np.load(working_dir + 'mean_X.npy')
#     mean_y = np.load(working_dir + 'mean_y.npy')
#     std_X  = np.load(working_dir + 'std_X.npy')
#     std_y  = np.load(working_dir + 'std_y.npy')

#     for i in range(len(data_norm)):
#         for i_cx in range(data_norm.X[0].shape[2]):
#             data_norm.X[i][:, :, i_cx] = data_norm.X[i][:, :, i_cx] * std_X[i_cx] + mean_X
#         for i_cy in range(data_norm.y[0].shape[2]):
#             data_norm.y[i][:, :, i_cy] = data_norm.y[i][:, :, i_cy] * std_y[i_cy] + mean_X
#     return data_norm




###############################################################################################################################
# OLD : Normalisations par sample
###############################################################################################################################

# ! Attention
# ! Les fonctions si-dessous ne sont pas à jour !
# TODO passer de (B x C x H x W) à (B x H x W x C) (premières fonctions seulement)
# TODO les rendres compatibles avec le nouveu chargement des données


# def standardisation(data):
#     data_standard = data.copy()
#     mean_X = []
#     mean_y = []
#     for i in range(len(data_standard)):
#         mean_X.append([np.mean(data_standard.X[i][i_p, :, :]) for i_p in range(data_standard.X[0].shape[0])])
#         mean_y.append([np.mean(data_standard.y[i][i_p, :, :]) for i_p in range(data_standard.y[0].shape[0])])
#     mean_X = pd.Series(mean_X, name='mean_X')        
#     mean_y = pd.Series(mean_y, name='mean_y')

#     std_X = []
#     std_y = []
#     for i in range(len(data_standard)):
#         std_X.append([np.std(data_standard.X[i][i_p, :, :]) for i_p in range(data_standard.X[0].shape[2])])
#         std_X.append([np.std(data_standard.X[i][i_p, :, :]) for i_p in range(data_standard.X[0].shape[2])])
#     std_X = pd.Series(std_X, name='std_X')        
#     std_y = pd.Series(std_y, name='std_y')

#     data_standard = pd.concat([data_standard, mean_X, mean_y, std_X, std_y], axis=1)

#     for i in range(len(data_standard)):
#         X_i = data_standard.X[i].copy()
#         y_i = data_standard.y[i].copy()
#         for i_p in range(data_standard.X[0].shape[2]):
#             X_i[i_p, :, :] = (data_standard.X[i][i_p, :, :] - data_standard.mean_X[i][i_p])/data_standard.std_X[i][i_p]
#             y_i[i_p, :, :] = (data_standard.y[i][i_p, :, :] - data_standard.mean_y[i][i_p])/data_standard.std_y[i][i_p]
#         data_standard.X[i] = X_i
#         data_standard.y[i] = y_i
#     return data_standard


# def destandardisation(data_standard):
#     data_des = data_standard.copy()
#     for i in range(len(data_des)):
#         X_i = data_des.X[i].copy()
#         y_i = data_des.y[i].copy()
#         for i_p in range(data_des.X[0].shape[0]):
#             X_i[i_p, :, :] = data_des.std_X[i][i_p] * X_i[i_p, :, :] + data_des.mean_X[i][i_p]
#             y_i[i_p, :, :] = data_des.std_y[i][i_p] * y_i[i_p, :, :] + data_des.mean_y[i][i_p]
#         data_des.X[i] = X_i
#         data_des.y[i] = y_i
#     return data_des.drop(columns=['mean_X', 'mean_y', 'std_X', 'std_y'])


# def normalisation(data):
#     data_norm = data.copy()
#     max_X = []
#     for i in range(len(data_norm)):
#         max_X.append([np.abs(data_norm.X[i][:, :, j]).max() for j in range(data_norm.X[0].shape[2])])
#     max_X = pd.Series(max_X, name='max_abs_X')        
#     max_y = pd.Series([np.abs(data_norm.y[i]).max() for i in range(len(data_norm))], name='max_abs_y')

#     data_norm = pd.concat([data_norm, max_X, max_y], axis=1)

#     for i in range(len(data_norm)):
#         data_norm.y[i] = data_norm.y[i] / data_norm.max_abs_y[i]
#         X_i = data_norm.X[i].copy()
#         for i_p in range(data_norm.X[0].shape[2]):
#             X_i[:, :, i_p] = X_i[:, :, i_p] / data_norm.max_abs_X[i][i_p]
#         data_norm.X[i] = X_i
#     return data_norm


# def denormalisation(data_norm):
#     data_den = data_norm.copy()
#     for i in range(len(data_den)):
#         data_den.y[i] = data_den.y[i] * data_den.max_abs_y[i]
#         X_i = data_den.X[i].copy()
#         for i_p in range(data_den.X[0].shape[2]):
#             X_i[:, :, i_p] = X_i[:, :, i_p] * data_den.max_abs_X[i][i_p]
#         data_den.X[i] = X_i
#     return data_den.drop(columns=['max_abs_X', 'max_abs_y'])


# def min_max_norm(data):
#     data_norm = data.copy()

#     min_X = []
#     for i in range(len(data_norm)):
#         min_X.append([data_norm.X[i][:, :, j].min() for j in range(data_norm.X[0].shape[2])])
#     min_X = pd.Series(min_X, name='min_X')        
#     min_y = pd.Series([data_norm.y[i].min() for i in range(len(data_norm))], name='min_y')

#     max_X = []
#     for i in range(len(data_norm)):
#         max_X.append([data_norm.X[i][:, :, j].max() for j in range(data_norm.X[0].shape[2])])
#     max_X = pd.Series(max_X, name='max_X')        
#     max_y = pd.Series([data_norm.y[i].max() for i in range(len(data_norm))], name='max_y')

#     data_norm = pd.concat([data_norm, min_X, min_y, max_X, max_y], axis=1)

#     for i in range(len(data_norm)):
#         data_norm.y[i] = (data_norm.y[i] - data_norm.min_y[i]) / (data_norm.max_y[i] - data_norm.min_y[i])
#         X_i = data_norm.X[i].copy()
#         for i_p in range(data_norm.X[0].shape[2]):
#             X_i[:, :, i_p] = (X_i[:, :, i_p] - data_norm.min_X[i][i_p]) / (data_norm.max_X[i][i_p] - data_norm.min_X[i][i_p])
#         data_norm.X[i] = X_i
#     return data_norm


# def min_max_denorm(data_norm):
#     data_den = data_norm.copy()
#     for i in range(len(data_den)):
#         data_den.y[i] = data_den.y[i] * (data_norm.max_y[i] - data_norm.min_y[i]) + data_norm.min_y[i]
#         X_i = data_den.X[i].copy()
#         for i_p in range(data_den.X[0].shape[2]):
#             X_i[:, :, i_p] = X_i[:, :, i_p] * (data_norm.max_X[i][i_p] - data_norm.min_X[i][i_p]) + data_norm.min_X[i][i_p]
#         data_den.X[i] = X_i
#     return data_den.drop(columns=['min_X', 'min_y', 'max_X', 'max_y'])


# def mean_norm(data):
#     data_norm = data.copy()

#     mean_X = []
#     for i in range(len(data_norm)):
#         mean_X.append([np.mean(data_norm.X[i][:, :, j]) for j in range(data_norm.X[0].shape[2])])
#     mean_X = pd.Series(mean_X, name='mean_X')        
#     mean_y = pd.Series([np.mean(data_norm.y[i]) for i in range(len(data_norm))], name='mean_y')

#     min_X = []
#     for i in range(len(data_norm)):
#         min_X.append([data_norm.X[i][:, :, j].min() for j in range(data_norm.X[0].shape[2])])
#     min_X = pd.Series(min_X, name='min_X')        
#     min_y = pd.Series([data_norm.y[i].min() for i in range(len(data_norm))], name='min_y')

#     max_X = []
#     for i in range(len(data_norm)):
#         max_X.append([data_norm.X[i][:, :, j].max() for j in range(data_norm.X[0].shape[2])])
#     max_X = pd.Series(max_X, name='max_X')        
#     max_y = pd.Series([data_norm.y[i].max() for i in range(len(data_norm))], name='max_y')

#     data_norm = pd.concat([data_norm, mean_X, mean_y, min_X, min_y, max_X, max_y], axis=1)

#     for i in range(len(data_norm)):
#         data_norm.y[i] = (data_norm.y[i] - data_norm.mean_y[i]) / (data_norm.max_y[i] - data_norm.min_y[i])
#         X_i = data_norm.X[i].copy()
#         for i_p in range(data_norm.X[0].shape[2]):
#             X_i[:, :, i_p] = (X_i[:, :, i_p] - data_norm.mean_X[i][i_p]) / (data_norm.max_X[i][i_p] - data_norm.min_X[i][i_p])
#         data_norm.X[i] = X_i
#     return data_norm


# def mean_denorm(data_norm):
#     data_den = data_norm.copy()
#     for i in range(len(data_den)):
#         data_den.y[i] = data_den.y[i] * (data_norm.max_y[i] - data_norm.min_y[i]) + data_norm.mean_y[i]
#         X_i = data_den.X[i].copy()
#         for i_p in range(data_den.X[0].shape[2]):
#             X_i[:, :, i_p] = X_i[:, :, i_p] * (data_norm.max_X[i][i_p] - data_norm.min_X[i][i_p]) + data_norm.mean_X[i][i_p]
#         data_den.X[i] = X_i
#     return data_den.drop(columns=['mean_X', 'mean_y', 'min_X', 'min_y', 'max_X', 'max_y'])
