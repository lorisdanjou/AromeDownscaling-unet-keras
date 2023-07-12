import numpy as np
import tensorflow as tf
import keras
from time import perf_counter


def rmse_k(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))


def mse_k(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1)


def mse_terre_mer_k(y_true, y_pred, shape, frac=0.5): # fonctionne pour des inputs complets adaptés à la grille de sortie (interpolation nearest, bl ou bc)
    """
    Computes MSE between two tensors, but sea and land domains have a different weight.

    Args:
        y_true (Tensor): 
        y_pred (Tensor): 
        shape (tuple): shape of the inputs
        frac (float, optional): weight of land domain. Defaults to 0.5.

    Returns:
        Tensor: 
    """
    shape = shape[1:4]
    # print(shape)
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


def mse_terre_mer(shape, frac=0.5):
    """
    Calls mse_terre_mer_k loss function without y_true and y_pred in arguments
    """
    return lambda y_true, y_pred : mse_terre_mer_k(y_true, y_pred, shape, frac=frac)


def modified_mse_k(y_true, y_pred, shape, tau, eps):
    """
    Computes a modified MSE (https://www.researchgate.net/publication/359211825_Wind-Topo_Downscaling_near-surface_wind_fields_to_high-resolution_topography_in_highly_complex_terrain_with_deep_learning)

    Args:
        y_true (Tensor): 
        y_pred (Tensor): 
        shape (tuple): shape of the inputs
        tau (float): parameter tau needed to define the loss function
        eps (float): parameter epsilon needed to define the loss

    Returns:
        Tensor: 
    """
    shape = shape[0:3]
    # beta coefficients: 
    b = (eps + y_true)/(eps + y_pred)
    weighted_mse = tf.math.reduce_sum(((y_pred - b * y_true)**2), axis=3)
    # pinball function (tau) : 
    norm_true = tf.math.reduce_sum(y_true**2, axis=3)
    norm_pred = tf.math.reduce_sum(y_pred**2, axis=3)
    norm_true = tf.math.sqrt(norm_true)
    norm_pred = tf.math.sqrt(norm_pred)

    bias = norm_pred - norm_true
    zeros = tf.zeros(shape)
    if tf.math.reduce_min(tf.abs(bias)) <= 1e-15:
        t = tau
    else:
        t = tau * tf.math.reduce_max([bias, zeros], axis = 0) / (bias) + (1 - tau) * tf.math.reduce_min([bias, zeros], axis = 0) / (bias)
    loss =  tf.math.reduce_mean(t * weighted_mse)
    # return mean : 
    return loss


def modified_mse(shape, tau, eps):
    """
    Calls modified_mse__k loss function without y_true and y_pred in arguments
    """
    return lambda y_true, y_pred : modified_mse_k(y_true, y_pred, shape, tau, eps)