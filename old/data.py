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
class Data():

    def __init__(self, dates, echeances, data_location, data_static_location, params, static_fields=[], missing_days=[]):
        self.dates = dates
        self.echeances = echeances
        self.data_location = data_location
        self.data_static_location = data_static_location
        self.params = params
        self.static_fields = static_fields
        self.missing_days = missing_days


    def copy(self):
        copy = Data(self.dates, self.echeances, self.data_location, self.data_static_location, self.params, self.static_fields, self.resample)
        return copy



class X(Data):

    def __init__(self, dates, echeances, data_location, data_static_location, params, static_fields=[], resample='r', missing_days=[]):
        super().__init__(dates, echeances, data_location, data_static_location, params, static_fields=static_fields, missing_days=missing_days)
        self.resample = resample
        self.domain_shape = get_shape_2km5(resample=self.resample)
        self.X = np.zeros(shape=[len(self.dates), len(self.echeances), self.domain_shape[0], self.domain_shape[1], len(self.params) + len(self.static_fields)], dtype=np.float32)

    def load(self):
        # initial shape of the data: X[date, ech, x, y, param]
        for i_d, d in enumerate(self.dates):
            try:
                for i_p, p in enumerate(self.params):
                    if self.resample == 'c':
                        filepath_X = self.data_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
                        self.X[i_d, :, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
                    else:
                        filepath_X = self.data_location + 'oper_r_' + d.isoformat() + 'Z_' + p + '.npy'
                        self.X[i_d, :, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
                for i_s, s in enumerate(self.static_fields):
                    for i_ech, ech in enumerate(self.echeances):
                        if self.resample == 'r':
                            filepath_static = self.data_static_location + 'static_G9KP_' + s + '.npy'
                            # filepath_static = data_static_location + 'static_oper_r_' + s + '.npy'
                        else:
                            filepath_static = self.data_static_location + 'static_oper_c_' + s + '.npy'
                        self.X[i_d, i_ech, :, :, len(self.params)+i_s] = np.load(filepath_static)
            except FileNotFoundError:
                print('missing day : ' + d.isoformat())
                self.missing_days.append(i_d)
        # print('initial X shape : ' + str(X.shape))

    def delete_missing_days(self, y):
        for i_d in y.missing_days:
            self.X[i_d, :, :, :, :] = np.zeros((self.X.shape[1], self.X.shape[2], self.X.shape[3], self.X.shape[4]))
            if i_d not in self.missing_days:
                self.missing_days.append(i_d)


    def reshape_4(self):
        # new shape of the data : X[date/ech, x, y, param]
        self.X = self.X.reshape((-1, self.domain_shape[0], self.domain_shape[1], len(self.params)+len(self.static_fields)))
        # print('reshaped (4) X shape : ' + str(X.shape))


    def reshape_5(self):
        # new shape of the data : X[date/ech, x, y, param]
        self.X = self.X.reshape((len(self.dates), len(self.echeances), self.domain_shape[0], self.domain_shape[1], len(self.params)+len(self.static_fields)))
        # print('reshaped (5) X shape : ' + str(X.shape))


    def copy(self):
        copy = X(self.dates, self.echeances, self.data_location, self.data_static_location, self.params, self.static_fields, self.resample, self.missing_days)
        copy.domain_shape = self.domain_shape
        copy.X = self.X
        return copy


    def pad(self): # adapte l'entrée pour un réseau à 4 convolutions + resample
        X1 = self.X

        if self.resample == 'r':
            X = np.pad(X1, ((0,0), (5,5), (2,3), (0,0)), mode='reflect')
            self.X = X
        else:
            print('data not resampled')

    def crop(self):
        X= self.X
        if self.resample == 'r':
            X1 = X[:, 5:-5, 2:-3, :]
        self.X = X1


    def normalize(self):
        X = self.X
        maxs_X = np.zeros(X[:, 0, 0, :].shape)
        for i_ech in range(X.shape[0]):
            for i_p in range(X.shape[3]):
                max_X = np.max(np.abs(X[i_ech, :, :, i_p]))
                maxs_X[i_ech, i_p] = max_X
                if max_X > 1e-6:
                    X[i_ech, :, :, i_p] = X[i_ech, :, :, i_p] / max_X 
        self.X = X
        return maxs_X


    def standardize(self):
        X = self.X
        means_X = np.zeros(X[:, 0, 0, :].shape)
        stds_X = np.zeros(X[:, 0, 0, :].shape)

        for i_ech in range(X.shape[0]):
            for i_p in range(X.shape[3]):
                mean_X = np.mean(X[i_ech, :, :, i_p])
                means_X[i_ech, i_p] = mean_X
                X[i_ech, :, :, i_p] = X[i_ech, :, :, i_p] - mean_X
                std_X = np.std(X[i_ech, :, :, i_p])
                stds_X[i_ech, i_p] = std_X
                if std_X > 1e-6:
                    X[i_ech, :, :, i_p] = X[i_ech, :, :, i_p] / std_X
        self.X = X
        return means_X, stds_X


    def denormalize(self, maxs_X):
        X = self.X
        for i_ech in range(X.shape[0]):
            for i_p in range(X.shape[3]):
                max_X = maxs_X[i_ech, i_p]
                if max_X > 1e-6:
                    X[i_ech, :, :, i_p] = X[i_ech, :, :, i_p] * max_X 
        self.X = X


    def destandardize(self, means_X, stds_X):
        X = self.X
        for i_ech in range(X.shape[0]):
            for i_p in range(X.shape[3]):
                mean_X = means_X[i_ech, i_p]
                std_X = stds_X[i_ech, i_p]
                if std_X > 1e-6:
                    X[i_ech, :, :, i_p] = X[i_ech, :, :, i_p] * std_X
                    X[i_ech, :, :, i_p] = X[i_ech, :, :, i_p] + mean_X

        self.X = X



class y(Data):
    
    def __init__(self, dates, echeances, data_location, data_static_location, params, static_fields=[], missing_days=[], base=False):
        super().__init__(dates, echeances, data_location, data_static_location, params, static_fields=static_fields, missing_days=missing_days)
        self.domain_shape = get_shape_500m()
        self.y = np.zeros(shape=[len(self.dates), len(self.echeances), self.domain_shape[0], self.domain_shape[1]], dtype=np.float32)
        self.base = base

    def load(self):
        # initial shape of the data: y[date, ech, x, y]
        if self.base:
            for i_d, d in enumerate(self.dates):
                try:
                    filepath_y = self.data_location + 'GG9B_' + d.isoformat() + 'Z_t2m.npy'
                    if exists(filepath_y):
                        self.y[i_d, :, :, :] = np.load(filepath_y).transpose([2, 0, 1])
                    else:
                        filepath_y = self.data_location + 'GG8A_' + d.isoformat() + 'Z_t2m.npy'
                        self.y[i_d, :, :, :] = np.load(filepath_y).transpose([2, 0, 1])
                except FileNotFoundError:
                    print('missing day : ' + d.isoformat())
                    self.missing_days.append(d)
        else:
            for i_d, d in enumerate(self.dates):
                try:
                    filepath_y = self.data_location + 'G9L1_' + d.isoformat() + 'Z_t2m.npy'
                    if exists(filepath_y):
                        self.y[i_d, :, :, :] = np.load(filepath_y).transpose([2, 0, 1])
                    else:
                        filepath_y = self.data_location + 'G9KP_' + d.isoformat() + 'Z_t2m.npy'
                        self.y[i_d, :, :, :] = np.load(filepath_y).transpose([2, 0, 1])
                except FileNotFoundError:
                    print('missing day : ' + d.isoformat())
                    self.missing_days.append(i_d)
        # print('initial y shape : ' + str(y.shape))
    
    def delete_missing_days(self, y):
        for i_d in y.missing_days:
            self.y[i_d, :, :, :] = np.zeros((self.y.shape[1], self.y.shape[2], self.y.shape[3]))
            if i_d not in self.missing_days:
                self.missing_days.append(i_d)

    def delete_missing_days(self, X):
        for i_d in X.missing_days:
            self.y[i_d, :, :, :] = np.zeros((self.y.shape[1], self.y.shape[2], self.y.shape[3]))
            if i_d not in self.missing_days:
                self.missing_days.append(i_d)

    def reshape_3(self):
        # new shape of the data : y[date/ech, x, y]
        self.y = self.y.reshape((-1, self.domain_shape[0], self.domain_shape[1]))
        # print('reshaped y shape : ' + str(y.shape))

    def reshape_4(self):
        self.y = self.y.reshape((len(self.dates), len(self.echeances), self.domain_shape[0], self.domain_shape[1]))

    def copy(self):
        copy = y(self.dates, self.echeances, self.data_location, self.data_static_location, self.params, self.static_fields, self.missing_days, base=self.base)
        copy.domain_shape = self.domain_shape
        copy.y = self.y
        return copy


    def pad(self): # adapte l'entrée pour un réseau à 4 convolutions + resample
        y1 = self.y
        y = np.pad(y1, ((0,0), (5,5), (2,3)), mode='reflect')
        self.y = y


    def crop(self):
        y = self.y
        y1 = y[:, 5:-5, 2:-3]
        self.y = y1


    def normalize(self):
        y = self.y
        maxs_y = np.zeros(y[:, 0, 0].shape)
        for i_ech in range(y.shape[0]):
            max_y = np.max(np.abs(y[i_ech, :, :]))
            maxs_y[i_ech] = max_y
            if max_y > 1e-6:
                y[i_ech, :, :] = y[i_ech, :, :] / max_y
        self.y = y
        return maxs_y


    def standardize(self):
        y = self.y
        means_y = np.zeros(y[:, 0, 0].shape)
        stds_y = np.zeros(y[:, 0, 0].shape)

        for i_ech in range(y.shape[0]):
            mean_y = np.mean(y[i_ech, :, :])
            means_y[i_ech] = mean_y
            y[i_ech, :, :] = y[i_ech, :, :] - mean_y
            std_y = np.std(y[i_ech, :, :])
            stds_y[i_ech] = std_y
            if std_y > 1e-6:
                y[i_ech, :, :] = y[i_ech, :, :] / std_y
        self.y = y
        return means_y, stds_y

    def denormalize(self, maxs_y):
        y = self.y
        for i_ech in range(y.shape[0]):
            max_y = maxs_y[i_ech]
            if max_y > 1e-6:
                y[i_ech, :, :] = y[i_ech, :, :] * max_y 
        self.y = y


    def destandardize(self, means_y, stds_y):
        y = self.y
        
        for i_ech in range(y.shape[0]):
            mean_y = means_y[i_ech]
            std_y = stds_y[i_ech]
            if std_y > 1e-6:
                y[i_ech, :, :] = y[i_ech, :, :] * std_y
            y[i_ech, :, :] = y[i_ech, :, :] + mean_y
        self.y = y