import os
import argparse
import data as Data
import utils
import training as Training
import matplotlib.pyplot as plt
import unet as Unet
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizer_v2.adam import Adam
from time import perf_counter
import core.logger as logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_example.jsonc',
                        help='JSON file for configuration')
    args = parser.parse_args()
    opt = logger.parse(args)

    # load & preprocess data
    t0 = perf_counter()

    X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df = Data.load_data(opt["data"])
    X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df = Data.preprocess_data(
        opt["preprocessing"],
        opt["path"]["experiment"],
        X_train_df,
        y_train_df,
        X_valid_df,
        y_valid_df,
        X_test_df,
        y_test_df
    )

    # generators
    training_opt = opt["training"]
    train_generator, valid_generator, X_test , y_test = Training.set_generators(
        X_train_df,
        y_train_df,
        X_valid_df,
        y_valid_df,
        X_test_df,
        y_test_df,
        training_opt
    )

    t1 = perf_counter()
    print("Loading & preprocessing time: {:.2f} s".format(t1 - t0))

    # model definition
    unet = Unet.define_model(opt["model"], (None, None, X_test.shape[3]), output_channels=len(opt["data"]["params_out"]))
    Unet.load_weights(opt["path"], unet)
    print("unet definition: ok")

    t2 = perf_counter()
    print("Model definition time: {:.2f} s".format(t2 - t1))

    unet.summary()

    # inference
    y_pred = unet.predict(X_test)
    print("y_pred shape : {}".format(y_pred.shape))

    y_pred_df = y_test_df.copy()
    arrays_cols = utils.get_arrays_cols(y_pred_df)
    for i in range(len(y_pred_df)):
        for i_c, c in enumerate(arrays_cols):
            y_pred_df[c][i] = y_pred[i, :, :, i_c]

    t3 = perf_counter()
    print("Inference time: {:.2f} s".format(t3 - t2))

    # postprocessing
    y_pred_df = Data.postprocess_data(opt, y_test_df)

    # save
    y_pred_df.to_pickle(opt["path"]["experiment"] + 'y_pred.csv')

    t4 = perf_counter()
    print("Postprocessing time: {:.2f} s".format(t4 - t3))
    print("Total time: {:.2f} s".format(t4 - t0))