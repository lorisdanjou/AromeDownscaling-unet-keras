import keras
import tensorflow as tf 
from keras.layers import Conv2D, Conv2DTranspose, concatenate, BatchNormalization, Activation


# ========== For classic Unet
def block_conv(conv, filters):
    conv = Conv2D(filters, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(filters, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv


def block_up_conc(conv, filters, conv_conc):
    conv = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv = concatenate([conv,conv_conc])
    conv = block_conv(conv, filters)
    return conv


def block_up(conv, filters):
    conv = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv = block_conv(conv, filters)
    return conv


# ========== For UResNet
def block_res_conv(conv, filters):
    conv = Conv2D(filters, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv_add = Conv2D(conv.shape[3], 3, padding='same')(conv)
    conv_add = BatchNormalization()(conv_add)
    conv_add = Activation('relu')(conv_add)
    conv_add = Conv2D(conv.shape[3], 3, padding='same')(conv_add)
    conv_add = BatchNormalization()(conv_add)
    conv_add = Activation('relu')(conv_add)

    return conv + conv_add


def block_res_up_conc(conv, filters, conv_conc):
    conv = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv = concatenate([conv,conv_conc])
    conv = block_res_conv(conv, filters)
    return conv


def block_res_up(conv, filters):
    conv = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv = block_res_conv(conv, filters)
    return conv