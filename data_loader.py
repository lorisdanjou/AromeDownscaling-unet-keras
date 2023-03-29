import numpy as np
from make_unet import*

'''
Useful functions to get the shape of a domain
'''
def get_shape_500m():
    field_500m = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/G9KP_2021-01-01T00:00:00Z_rr.npy')
    return field_500m[:, :, 0].shape


def get_size_500m():
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



def get_size_2km5(resample='c'):
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
def load_X_y(dates, echeances, data_location, params, resample='c'):
    shape_500m = get_shape_500m()
    shape_2km5 = get_shape_2km5(resample=resample)

    # initial shape of the data:
        # X[date, ech, x, y, param]
        # y[date, ech, x, y]
    X = np.zeros(shape=[len(dates), len(echeances), shape_2km5[0], shape_2km5[1], len(params)], dtype=np.float32)
    y = np.zeros(shape=[len(dates), len(echeances), shape_500m[0], shape_500m[1]], dtype=np.float32)

    for i_d, d in enumerate(dates):
        try:
            try:
                filepath_y = data_location + 'G9L1_' + d.isoformat() + 'Z_t2m.npy'
                y[i_d, :, :, :] = np.load(filepath_y).transpose([2, 0, 1])
            except:
                filepath_y = data_location + 'G9KP_' + d.isoformat() + 'Z_t2m.npy'
                y[i_d, :, :, :] = np.load(
                    filepath_y).transpose([2, 0, 1])
            for i_p, p in enumerate(params):
                if resample == 'c':
                    filepath_X = data_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
                    X[i_d, :, :, :, i_p] = np.load(
                        filepath_X).transpose([2, 0, 1])
                else:
                    filepath_X = data_location + 'oper_r_' + d.isoformat() + 'Z_' + p + '.npy'
                    X[i_d, :, :, :, i_p] = np.load(
                        filepath_X).transpose([2, 0, 1])
        except:
            print('missing day : ' + d)

    print('initial X shape : ' + str(X.shape))
    print('initial y shape : ' + str(y.shape))

    # new shape of the data :
        # X[date/ech, x, y, param]
        # y[date/ech, x, y]
    X = X.reshape((-1, shape_2km5[0], shape_2km5[1], len(params)))
    y = y.reshape((-1, shape_500m[0], shape_500m[1]))

    print('reshaped X shape : ' + str(X.shape))
    print('reshaped y shape : ' + str(y.shape))

    # final shape of the data:
        # X[date/exh, new_x, new_y, param]
        # y[date/ech, new_x, new_y, param]
    X = X[:, int(0.5 * (shape_2km5[0] - get_size_2km5(resample=resample)[1])):int(0.5 * (shape_2km5[0] + get_size_2km5(resample=resample)[1])), 
            int(0.5 * (shape_2km5[1] - get_size_2km5(resample=resample)[1])):int(0.5 * (shape_2km5[1] + get_size_2km5(resample=resample)[1])), :]
    y = y[:, int(0.5 * (shape_500m[0] - get_size_500m()[1])):int(0.5 * (shape_500m[0] + get_size_500m()[1])), 
            int(0.5 * (shape_500m[1] - get_size_500m()[1])):int(0.5 * (shape_500m[1] + get_size_500m()[1]))]

    print('final (cropped) X shape : ' + str(X.shape))
    print('final (cropped) y shape : ' + str(y.shape))

    return X, y
