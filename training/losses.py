import numpy as np
import tensorflow as tf
import keras


def rmse_k(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))

def mse_k(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1)


def mse_terre_mer_k(y_true, y_pred): # fonctionne pour des inputs complets adaptés à la grille de sortie (interpolation nearest, bl ou bc)
    frac = 0.6
    shape = y_true.get_shape()[1:4]
    ind_terre_mer = np.load('/cnrm/recyf/Data/users/danjoul/dataset/static_G9KP_SURFIND.TERREMER.npy', allow_pickle=True)
    ind_terre_mer = np.pad(ind_terre_mer, ((5,5), (2,3)), mode='reflect')
    ind_terre_mer_numpy = np.zeros(shape)
    for i in range(shape[2]):
        ind_terre_mer_numpy[:, :, i] = ind_terre_mer
    ind_terre_mer_tf = tf.convert_to_tensor(ind_terre_mer_numpy,  dtype=tf.float32)
    y_true_terre = y_true * ind_terre_mer_tf
    y_true_mer   = y_true * (1 - ind_terre_mer_tf)
    y_pred_terre = y_pred * ind_terre_mer_tf
    y_pred_mer   = y_pred * (1 - ind_terre_mer_tf)

    return frac * mse_k(y_true_terre, y_pred_terre) + (1 - frac) * mse_k(y_true_mer, y_pred_mer)
