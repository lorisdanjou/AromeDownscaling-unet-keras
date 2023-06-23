import os
import argparse
import data as Data
from data.load_data import get_arrays_cols
import training as Training
import matplotlib.pyplot as plt
import unet as Unet
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizer_v2.adam import Adam
from time import perf_counter
import json
from collections import OrderedDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    args = parser.parse_args()
    opt_path = args.config
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

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

    # training

    ## pb avec shape
    shape = (
        opt["training"]["batch_size"],
        X_train_df[opt["data"]["params_in"][0]].iloc[0].shape[0],
        X_train_df[opt["data"]["params_in"][0]].iloc[0].shape[1],
        len(opt["data"]["params_out"])
    )
    loss = Training.set_loss(training_opt, shape)
    unet.compile(optimizer=Adam(learning_rate=training_opt["learning_rate"]), loss=loss, run_eagerly=training_opt["run_eagerly"])  
    print('compilation ok')

    checkpoint_path = os.path.join(opt["path"]["experiment"], "checkpoint")
    os.makedirs(checkpoint_path, exist_ok = True)
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, verbose=1),
        ModelCheckpoint(os.path.join(checkpoint_path, "weights.{epoch:02d}.h5"), monitor='val_loss', verbose=1, save_best_only=True)
    ]

    t2 = perf_counter()
    print("Model definition & training preparation time: {:.2f} s".format(t2 - t1))

    history = unet.fit(
        train_generator, 
        batch_size=training_opt["batch_size"],
        epochs=training_opt["n_epochs"],  
        validation_data=valid_generator, 
        callbacks = callbacks,
        verbose=2,
        shuffle=True
    )

    unet.summary()
    print(history.history.keys())

    # summarize history for loss
    loss_curve = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.semilogy()
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(opt["path"]["experiment"] + 'Loss_curve.png')

    t3 = perf_counter()
    print("Training time: {:.2f} s".format(t3 - t2))

    # inference
    y_pred = unet.predict(X_test)
    print("y_pred shape : {}".format(y_pred.shape))

    y_pred_df = y_test_df.copy()
    arrays_cols = get_arrays_cols(y_pred_df)
    for i in range(len(y_pred_df)):
        for i_c, c in enumerate(arrays_cols):
            y_pred_df[c][i] = y_pred[i, :, :, i_c]

    t4 = perf_counter()
    print("Inference time: {:.2f} s".format(t4 - t3))

    # postprocessing
    y_pred_df = Data.postprocess_data(opt, y_test_df)

    # save
    y_pred_df.to_pickle(opt["path"]["experiment"] + 'y_pred.csv')

    t5 = perf_counter()
    print("Postprocessing time: {:.2f} s".format(t5 - t4))
    print("Total time: {:.2f} s".format(t5 - t0))

    ### TODO:
    # reproduire plusieurs configurations avec les fichiers jsonc associés et les tester !
    # répertoire utils avec get_arrays_cols, etc.
    # ajouter un logger
    # faire un dernier tri dans les fichiers