import numpy as np
from os.path import exists
import pandas as pd
from bronx.stdtypes.date import daterangex as rangex


'''
Useful functions to get the shape of a domain
'''
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


'''
Load dataset
'''
def load_data(dates, echeances, data_location, data_static_location, params, static_fields=[], resample='r'):
    data = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'X' : [],
        'y' : []}
    )

    domain_shape = get_shape_2km5(resample=resample)
    for i_d, d in enumerate(dates):
        # load X
        X_i = np.zeros([len(echeances), domain_shape[0], domain_shape[1], len(params) + len(static_fields)])
        try:
            for i_p, p in enumerate(params):
                if resample == 'c':
                    filepath_X = data_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
                    X_i[:, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
                else:
                    filepath_X = filepath_X = data_location + 'oper_r_' + d.isoformat() + 'Z_' + p + '.npy'
                    X_i[:, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
        # load X static
            for i_s, s in enumerate(static_fields):
                if resample == 'r':
                    filepath_static = data_static_location + 'static_G9KP_' + s + '.npy'
                    # filepath_static = data_static_location + 'static_oper_r_' + s + '.npy'
                else:
                    filepath_static = data_static_location + 'static_oper_c_' + s + '.npy'
                for i_ech, ech in enumerate(echeances):
                    X_i[i_ech, :, :, i_p + i_s] = np.load(filepath_static)
        except FileNotFoundError:
            print('missing day (X): ' + d.isoformat())
            X_i = None

        # load y
        try:
            filepath_y = data_location + 'G9L1_' + d.isoformat() + 'Z_t2m.npy'
            if exists(filepath_y):
                y_i = np.load(filepath_y).transpose([2, 0, 1])
            else:
                filepath_y = data_location + 'G9KP_' + d.isoformat() + 'Z_t2m.npy'
                y_i = np.load(filepath_y).transpose([2, 0, 1])
        except FileNotFoundError:
            print('missing day (y): ' + d.isoformat())
            y_i = None

        for i_ech, ech in enumerate(echeances):
            try:
                data_i = pd.DataFrame(
                    {'dates' : [d.isoformat()],
                    'echeances' : [ech],
                    'X' : [X_i[i_ech, :, :, :]],
                    'y' : [y_i[i_ech, :, :]]}
                )
            except TypeError:
                data_i = pd.DataFrame(
                    {'dates' : [],
                    'echeances' : [],
                    'X' : [],
                    'y' : []}
                )
            data = pd.concat([data, data_i])
    return data.reset_index(drop=True)


def to_array(arrays_serie):
    if len(arrays_serie[0].shape) == 3: # C'est un X
        output = np.zeros([len(arrays_serie), arrays_serie[0].shape[0], arrays_serie[0].shape[1], arrays_serie[0].shape[2]])
        for i in range(len(arrays_serie)):
            output[i, :, :, :] = arrays_serie[i]
    else: # C'est un y
        output = np.zeros([len(arrays_serie), arrays_serie[0].shape[0], arrays_serie[0].shape[1]])
        for i in range(len(arrays_serie)):
            output[i, :, :] = arrays_serie[i]
    return output




# def to_array(array_serie, dates, echeances):
#     array_serie_na = array_serie.dropna()
#     if len(array_serie_na[0].shape) == 3:
#         output = np.zeros([len(dates)*len(echeances), array_serie_na[0].shape[0], array_serie_na[0].shape[1],  array_serie_na[0].shape[2]])
#         for i in range(len(array_serie_na)):
#             output[array_serie_na.index[i],: , :, :] = array_serie.iloc[i]
#     else:
#         output = np.zeros([len(dates)*len(echeances), array_serie_na[0].shape[0], array_serie_na[0].shape[1]])
#         for i in range(len(array_serie_na)):
#             output[array_serie_na.index[i], : , :] = array_serie.iloc[i]
#     return output


'''
Pre/postprocessing functions
'''
def pad(data):
    data_pad = data.copy()
    for i in range(len(data)):
        data_pad.X[i] = np.pad(data.X[i], ((5,5), (2,3), (0,0)), mode='reflect')
        data_pad.y[i] = np.pad(data.y[i], ((5,5), (2,3)), mode='reflect')
    return data_pad


# def pad(data_na):
#     data = data_na.dropna()
#     data_pad = data.copy()
#     for i in range(len(data)):
#         data_pad.X.iloc[i] = np.pad(data.X[i], ((5,5), (2,3), (0,0)), mode='reflect')
#         data_pad.y.iloc[i] = np.pad(data.y[i], ((5,5), (2,3)), mode='reflect')
#     return data_pad

def crop(data):
    data_crop = data.copy()
    for i in range(len(data)):
        data_crop.X[i] = data.X[i][5:-5, 2:-3, :]
        data_crop.y[i] = data.y[i][5:-5, 2:-3]
    return data_crop


def standardisation(data):
    data_standard = data.copy()
    mean_X = []
    for i in range(len(data_standard)):
        mean_X.append([np.mean(data_standard.X[i][:, :, j]) for j in range(data_standard.X[0].shape[2])])
    mean_X = pd.Series(mean_X, name='mean_X')        
    mean_y = pd.Series([np.mean(data_standard.y[i]) for i in range(len(data_standard))], name='mean_y')

    std_X = []
    for i in range(len(data_standard)):
        std_X.append([np.std(data_standard.X[i][:, :, i_p]) for i_p in range(data_standard.X[0].shape[2])])
    std_X = pd.Series(std_X, name='std_X')        
    std_y = pd.Series([np.std(data_standard.y[i]) for i in range(len(data_standard))], name='std_y')

    data_standard = pd.concat([data_standard, mean_X, mean_y, std_X, std_y], axis=1)

    for i in range(len(data_standard)):
        data_standard.y[i] = (data_standard.y[i] - data_standard.mean_y[i])/data_standard.std_y[i]
        X_i = data_standard.X[i].copy()
        for i_p in range(data_standard.X[0].shape[2]):
            X_i[:, :, i_p] = (data_standard.X[i][:, :, i_p] - data_standard.mean_X[i][i_p])/data_standard.std_X[i][i_p]
        data_standard.X[i] = X_i
    return data_standard


def destandardisation(data_standard):
    data_des = data_standard.copy()
    for i in range(len(data_des)):
        data_des.y[i] = data_des.std_y[i] * data_des.y[i] + data_des.mean_y[i]
        X_i = data_des.X[i].copy()
        for i_p in range(data_des.X[0].shape[2]):
            X_i[:, :, i_p] = data_des.std_X[i][i_p] * X_i[:, :, i_p] + data_des.mean_X[i][i_p]
        data_des.X[i] = X_i
    return data_des.drop(columns=['mean_X', 'mean_y', 'std_X', 'std_y'])


def normalisation(data):
    data_norm = data.copy()
    max_X = []
    for i in range(len(data_norm)):
        max_X.append([np.abs(data_norm.X[i][:, :, j]).max() for j in range(data_norm.X[0].shape[2])])
    max_X = pd.Series(max_X, name='max_abs_X')        
    max_y = pd.Series([np.abs(data_norm.y[i]).max() for i in range(len(data_norm))], name='max_abs_y')

    data_norm = pd.concat([data_norm, max_X, max_y], axis=1)

    for i in range(len(data_norm)):
        data_norm.y[i] = data_norm.y[i] / data_norm.max_abs_y[i]
        X_i = data_norm.X[i].copy()
        for i_p in range(data_norm.X[0].shape[2]):
            X_i[:, :, i_p] = X_i[:, :, i_p] / data_norm.max_abs_X[i][i_p]
        data_norm.X[i] = X_i
    return data_norm


def denormalisation(data_norm):
    data_den = data_norm.copy()
    for i in range(len(data_den)):
        data_den.y[i] = data_den.y[i] * data_den.max_abs_y[i]
        X_i = data_den.X[i].copy()
        for i_p in range(data_den.X[0].shape[2]):
            X_i[:, :, i_p] = X_i[:, :, i_p] * data_den.max_abs_X[i][i_p]
        data_den.X[i] = X_i
    return data_den.drop(columns=['max_abs_X', 'max_abs_y'])


def min_max_norm(data):
    data_norm = data.copy()

    min_X = []
    for i in range(len(data_norm)):
        min_X.append([data_norm.X[i][:, :, j].min() for j in range(data_norm.X[0].shape[2])])
    min_X = pd.Series(min_X, name='min_X')        
    min_y = pd.Series([data_norm.y[i].min() for i in range(len(data_norm))], name='min_y')

    max_X = []
    for i in range(len(data_norm)):
        max_X.append([data_norm.X[i][:, :, j].max() for j in range(data_norm.X[0].shape[2])])
    max_X = pd.Series(max_X, name='max_X')        
    max_y = pd.Series([data_norm.y[i].max() for i in range(len(data_norm))], name='max_y')

    data_norm = pd.concat([data_norm, min_X, min_y, max_X, max_y], axis=1)

    for i in range(len(data_norm)):
        data_norm.y[i] = (data_norm.y[i] - data_norm.min_y[i]) / (data_norm.max_y[i] - data_norm.min_y[i])
        X_i = data_norm.X[i].copy()
        for i_p in range(data_norm.X[0].shape[2]):
            X_i[:, :, i_p] = (X_i[:, :, i_p] - data_norm.min_X[i][i_p]) / (data_norm.max_X[i][i_p] - data_norm.min_X[i][i_p])
        data_norm.X[i] = X_i
    return data_norm


def min_max_denorm(data_norm):
    data_den = data_norm.copy()
    for i in range(len(data_den)):
        data_den.y[i] = data_den.y[i] * (data_norm.max_y[i] - data_norm.min_y[i]) + data_norm.min_y[i]
        X_i = data_den.X[i].copy()
        for i_p in range(data_den.X[0].shape[2]):
            X_i[:, :, i_p] = X_i[:, :, i_p] * (data_norm.max_X[i][i_p] - data_norm.min_X[i][i_p]) + data_norm.min_X[i][i_p]
        data_den.X[i] = X_i
    return data_den.drop(columns=['min_X', 'min_y', 'max_X', 'max_y'])


def mean_norm(data):
    data_norm = data.copy()

    mean_X = []
    for i in range(len(data_norm)):
        mean_X.append([np.mean(data_norm.X[i][:, :, j]) for j in range(data_norm.X[0].shape[2])])
    mean_X = pd.Series(mean_X, name='mean_X')        
    mean_y = pd.Series([np.mean(data_norm.y[i]) for i in range(len(data_norm))], name='mean_y')

    min_X = []
    for i in range(len(data_norm)):
        min_X.append([data_norm.X[i][:, :, j].min() for j in range(data_norm.X[0].shape[2])])
    min_X = pd.Series(min_X, name='min_X')        
    min_y = pd.Series([data_norm.y[i].min() for i in range(len(data_norm))], name='min_y')

    max_X = []
    for i in range(len(data_norm)):
        max_X.append([data_norm.X[i][:, :, j].max() for j in range(data_norm.X[0].shape[2])])
    max_X = pd.Series(max_X, name='max_X')        
    max_y = pd.Series([data_norm.y[i].max() for i in range(len(data_norm))], name='max_y')

    data_norm = pd.concat([data_norm, mean_X, mean_y, min_X, min_y, max_X, max_y], axis=1)

    for i in range(len(data_norm)):
        data_norm.y[i] = (data_norm.y[i] - data_norm.mean_y[i]) / (data_norm.max_y[i] - data_norm.min_y[i])
        X_i = data_norm.X[i].copy()
        for i_p in range(data_norm.X[0].shape[2]):
            X_i[:, :, i_p] = (X_i[:, :, i_p] - data_norm.mean_X[i][i_p]) / (data_norm.max_X[i][i_p] - data_norm.min_X[i][i_p])
        data_norm.X[i] = X_i
    return data_norm


def mean_denorm(data_norm):
    data_den = data_norm.copy()
    for i in range(len(data_den)):
        data_den.y[i] = data_den.y[i] * (data_norm.max_y[i] - data_norm.min_y[i]) + data_norm.mean_y[i]
        X_i = data_den.X[i].copy()
        for i_p in range(data_den.X[0].shape[2]):
            X_i[:, :, i_p] = X_i[:, :, i_p] * (data_norm.max_X[i][i_p] - data_norm.min_X[i][i_p]) + data_norm.mean_X[i][i_p]
        data_den.X[i] = X_i
    return data_den.drop(columns=['mean_X', 'mean_y', 'min_X', 'min_y', 'max_X', 'max_y'])
