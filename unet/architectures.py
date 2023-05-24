import tensorflow as tf
import keras
from keras.layers import MaxPooling2D
from unet.blocks import *


def unet_maker(shape_input, output_channels=1, layers = 4, filters = 32):
    inputs_list=[]
    inputs = keras.Input(shape = shape_input)
    inputs_list.append(inputs)
    conv_down=[]

    prev = inputs

    for i in range(layers):
        conv=block_conv(prev, filters*2**i)
        pool=MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv)
        conv_down.append(conv)
        prev=pool
        print('down : ', i)
    up=block_conv(prev, filters*int(pow(2,i)))

    for i in range(layers-1, -1, -1):
        up=block_up_conc(up,filters*2**i,conv_down[i])
        print('up conc : ', i)

    last=Conv2D(output_channels, 1, padding='same')(up)
    return (keras.models.Model(inputs=inputs_list, outputs=last))


def UResNet_maker(shape_input, output_channels=1, layers = 4, filters = 32):
    inputs_list=[]
    inputs = keras.Input(shape = shape_input)
    inputs_list.append(inputs)
    conv_down=[]

    prev = inputs

    for i in range(layers):
        conv=block_res_conv(prev, filters*2**i)
        pool=MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv)
        conv_down.append(conv)
        prev=pool
        print('down : ', i)
    up=block_res_conv(prev, filters*int(pow(2,i)))

    for i in range(layers-1, -1, -1):
        up=block_res_up_conc(up,filters*2**i,conv_down[i])
        print('up conc : ', i)

    last=Conv2D(output_channels, 1, padding='same')(up)
    return (keras.models.Model(inputs=inputs_list, outputs=last))


def ResUNet_maker(shape_input, output_channels=1, layers = 4, filters = 32):
    inputs_list=[]
    inputs = keras.Input(shape = shape_input)
    inputs_list.append(inputs)


    # classic Unet :
    conv_down=[]
    prev = inputs

    for i in range(layers):
        conv=block_conv(prev, filters*2**i)
        pool=MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv)
        conv_down.append(conv)
        prev=pool
        print('down : ', i)
    up=block_conv(prev, filters*int(pow(2,i)))

    for i in range(layers-1, -1, -1):
        up=block_up_conc(up,filters*2**i,conv_down[i])
        print('up conc : ', i)

    last_unet=Conv2D(output_channels, 1, padding='same')(up)

    # res net :
    inputs_conv = Conv2D(output_channels, 1, padding='same')(inputs)
    last = inputs_conv + last_unet

    return (keras.models.Model(inputs=inputs_list, outputs=last))
