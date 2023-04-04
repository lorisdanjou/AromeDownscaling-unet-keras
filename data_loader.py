import numpy as np
from os.path import exists

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
Data Loader
'''
def load_X_y(dates, echeances, data_location, data_static_location, params, static_fields=[], resample='r'):
    
    shape_500m = get_shape_500m()
    shape_2km5 = get_shape_2km5(resample=resample)

    # initial shape of the data:
        # X[date, ech, x, y, param]
        # y[date, ech, x, y]
    X = np.zeros(shape=[len(dates), len(echeances), shape_2km5[0], shape_2km5[1], len(params) + len(static_fields)], dtype=np.float32)
    y = np.zeros(shape=[len(dates), len(echeances), shape_500m[0], shape_500m[1]], dtype=np.float32)
    static = np.zeros(shape=[shape_2km5[0], shape_2km5[1], len(static_fields)])

    for i_d, d in enumerate(dates):
        try:
            filepath_y = data_location + 'G9L1_' + d.isoformat() + 'Z_t2m.npy'
            if exists(filepath_y):
                y[i_d, :, :, :] = np.load(filepath_y).transpose([2, 0, 1])
            else:
                filepath_y = data_location + 'G9KP_' + d.isoformat() + 'Z_t2m.npy'
                y[i_d, :, :, :] = np.load(filepath_y).transpose([2, 0, 1])

            for i_p, p in enumerate(params):
                if resample == 'c':
                    filepath_X = data_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
                    X[i_d, :, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
                else:
                    filepath_X = data_location + 'oper_r_' + d.isoformat() + 'Z_' + p + '.npy'
                    X[i_d, :, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
            for i_s, s in enumerate(static_fields):
                for i_ech, ech in enumerate(echeances):
                    if resample == 'r':
                        filepath_static = data_static_location + 'static_G9KP_' + s + '.npy'
                        # filepath_static = data_static_location + 'static_oper_r_' + s + '.npy'
                    else:
                        filepath_static = data_static_location + 'static_oper_c_' + s + '.npy'
                    X[i_d, i_ech, :, :, len(params)+i_s] = np.load(filepath_static)
        except FileNotFoundError:
            print('missing day')

    print('initial X shape : ' + str(X.shape))
    print('initial y shape : ' + str(y.shape))

    # new shape of the data :
        # X[date/ech, x, y, param]
        # y[date/ech, x, y]
    X = X.reshape((-1, shape_2km5[0], shape_2km5[1], len(params)+len(static_fields)))
    y = y.reshape((-1, shape_500m[0], shape_500m[1]))

    print('reshaped X shape : ' + str(X.shape))
    print('reshaped y shape : ' + str(y.shape))

    return X, y


def load_X_y_r(dates, echeances, data_location, data_static_location, params, static_fields=[]):

    X1, y1 = load_X_y(dates, echeances, data_location, data_static_location, params, static_fields, resample='r')

    X = np.pad(X1, ((0,0), (5,5), (2,3), (0,0)), mode='reflect')
    y = np.pad(y1, ((0,0), (5,5), (2,3)), mode='reflect')

    return X, y


