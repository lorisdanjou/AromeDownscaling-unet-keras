import keras
import tensorflow as tf 
from keras.layers import Conv2D, Conv2DTranspose, concatenate, BatchNormalization, Activation



def block_conv(conv, filters):
    conv = Conv2D(filters, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(filters, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv


def block_up_conc(conv, filters,conv_conc):
    conv = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv = concatenate([conv,conv_conc])
    conv = block_conv(conv, filters)
    return conv


def block_up(conv, filters):
    conv = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv = block_conv(conv, filters)
    return conv