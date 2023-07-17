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
import pandas as pd


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
    train_generator, valid_generator, test_generator = Training.set_generators(
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
    n_channels = len(utils.get_arrays_cols(X_test_df.head(1)))
    unet = Unet.define_model(opt["model"], (None, None, n_channels), output_channels=len(opt["data"]["params_out"]))
    Unet.load_weights(opt["path"], unet)
    print("unet definition: ok")

    t2 = perf_counter()
    print("Model definition time: {:.2f} s".format(t2 - t1))

    unet.summary()

    # inference
    y_pred = unet.predict(test_generator)
    print("y_pred shape : {}".format(y_pred.shape))

    y_pred_df = pd.DataFrame(
        [],
        columns=y_test_df.columns
    )
    arrays_cols = utils.get_arrays_cols(y_test_df)
    for i in range(y_pred.shape[0]):
        y_pred_df.loc[len(y_pred_df)] = [y_test_df.dates.iloc[i], y_test_df.echeances.iloc[i]] + \
            [y_pred[i, :, :, i_c] for  i_c in range(len(arrays_cols))]
    print("length of y_pred_df : ", len(y_pred_df))

    t3 = perf_counter()
    print("Inference time: {:.2f} s".format(t3 - t2))

    # postprocessing
    y_pred_df = Data.postprocess_data(opt, y_pred_df)

    # save
    y_pred_df.to_pickle(opt["path"]["experiment"] + 'y_pred.csv')

    t4 = perf_counter()
    print("Postprocessing time: {:.2f} s".format(t4 - t3))
    print("Total time: {:.2f} s".format(t4 - t0))