import numpy as np
from make_unet import*


def get_shape_500m():
    field_500m = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/G9KP_2021-01-01T00:00:00Z_rr.npy')
    return field_500m[:,:,0].shape

def get_size_500m():
    field_500m = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/G9KP_2021-01-01T00:00:00Z_rr.npy')
    geometry_500m = field_500m[:,:,0].shape
    size_500m = min(geometry_500m)
    size_500m_crop = highestPowerof2(min(geometry_500m))
    return size_500m, size_500m_crop

def get_shape_2km5():
    field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_c_2021-01-01T00:00:00Z_rr.npy')
    return field_2km5[:,:,0].shape

def get_size_2km5():
    field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_c_2021-01-01T00:00:00Z_rr.npy')
    geometry_2km5 = field_2km5[:,:,0].shape
    size_2km5 = min(geometry_2km5)
    size_2km5_crop = highestPowerof2(min(geometry_2km5))
    return size_2km5, size_2km5_crop

def load_data(dates_train, dates_valid, echeances, data_train_location, data_valid_location, params):
    shape_500m = get_shape_500m()
    shape_2km5 = get_shape_2km5()

    # initial shape of the data:
        # X[date, ech, x, y, param]
        # y[date, ech, x, y]
    X_train = np.zeros(shape=[len(dates_train), len(echeances), shape_2km5[0], shape_2km5[1], len(params)], dtype=np.float32)
    y_train = np.zeros(shape=[len(dates_train), len(echeances), shape_500m[0], shape_500m[1]], dtype=np.float32)
    X_valid = np.zeros(shape=[len(dates_valid), len(echeances), shape_2km5[0], shape_2km5[1], len(params)], dtype=np.float32)
    y_valid = np.zeros(shape=[len(dates_valid), len(echeances), shape_500m[0], shape_500m[1]], dtype=np.float32)

    for i_d, d in enumerate(dates_train):
        try:
            filepath_y_train = data_train_location + 'G9L1_' + d.isoformat() + 'Z_t2m.npy'
            y_train[i_d, :, :, :] = np.load(filepath_y_train).transpose([2,0,1])
        except :
            filepath_y_train = data_train_location + 'G9KP_' + d.isoformat() + 'Z_t2m.npy'
            y_train[i_d, :, :, :] = np.load(filepath_y_train).transpose([2,0,1])
        for i_p, p in enumerate(params):
            filepath_X_train = data_train_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
            X_train[i_d, :, :, :, i_p] = np.load(filepath_X_train).transpose([2,0,1])

    for i_d, d in enumerate(dates_valid):
        filepath_y_valid = data_valid_location + 'G9L1_' + d.isoformat() + 'Z_t2m.npy'
        y_valid[i_d, :, :, :] = np.load(filepath_y_valid).transpose([2,0,1])
        for i_p, p in enumerate(params):
            filepath_X_valid = data_valid_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
            X_valid[i_d, :, :, :, i_p] = np.load(filepath_X_valid).transpose([2,0,1])

    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)

    # new shape of the data : 
        # X[date/ech, x, y, param]
        # y[date/ech, x, y]
    X_train = X_train.reshape((-1, shape_2km5[0], shape_2km5[1], len(params)))
    y_train = y_train.reshape((-1, shape_500m[0], shape_500m[1]))
    X_valid = X_valid.reshape((-1, shape_2km5[0], shape_2km5[1], len(params)))
    y_valid = y_valid.reshape((-1, shape_500m[0], shape_500m[1]))

    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)

    # final shape of the data:
        # X[date/exh, new_x, new_y, param]
        # y[date/ech, new_x, new_y, param]
    X_train = X_train[:, int(0.5* (shape_2km5[0] - get_size_2km5()[1])):int(0.5* (shape_2km5[0] + get_size_2km5()[1])), int(0.5* (shape_2km5[1] - get_size_2km5()[1])):int(0.5* (shape_2km5[1] + get_size_2km5()[1])), :]
    y_train = y_train[:, int(0.5* (shape_500m[0] - get_size_500m()[1])):int(0.5* (shape_500m[0] + get_size_500m()[1])), int(0.5* (shape_500m[1] - get_size_500m()[1])):int(0.5* (shape_500m[1] + get_size_500m()[1]))]
    X_valid = X_valid[:, int(0.5* (shape_2km5[0] - get_size_2km5()[1])):int(0.5* (shape_2km5[0] + get_size_2km5()[1])), int(0.5* (shape_2km5[1] - get_size_2km5()[1])):int(0.5* (shape_2km5[1] + get_size_2km5()[1])), :]
    y_valid = y_valid[:, int(0.5* (shape_500m[0] - get_size_500m()[1])):int(0.5* (shape_500m[0] + get_size_500m()[1])), int(0.5* (shape_500m[1] - get_size_500m()[1])):int(0.5* (shape_500m[1] + get_size_500m()[1]))]

    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)

    return X_train, y_train, X_valid, y_valid