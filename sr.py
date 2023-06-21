import argparse
import data as Data
from data.load_data import df_to_array, get_arrays_cols
import training as Training
import matplotlib.pyplot as plt

from unet.architectures import unet_maker
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
    X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df = Data.load_data(opt["data"])
    X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df = Data.preprocess_data(
        opt["preprocessing"],
        opt["working_dir"],
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

    # model definition
    unet = unet_maker((None, None, X_test.shape[3]), output_channels=len(opt["data"]["params_out"]))
    print("unet definition: ok")

    # training
    loss = Training.set_loss(training_opt)
    unet.compile(optimizer=Adam(learning_rate=training_opt["learning_rate"]), loss=loss, run_eagerly=training_opt["run_eagerly"])  
    print('compilation ok')

    model_name = 'weights.{epoch:02d}.h5'
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, verbose=1),
        ModelCheckpoint(opt["working_dir"] + model_name, monitor='val_loss', verbose=1, save_best_only=True)
    ]

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
    plt.savefig(opt["working_dir"] + 'Loss_curve.png')


    # prediction
    y_pred = unet.predict(X_test)
    print("y_pred shape : {}".format(y_pred.shape))

    y_pred_df = y_test_df.copy()
    arrays_cols = get_arrays_cols(y_pred_df)
    for i in range(len(y_pred_df)):
        for i_c, c in enumerate(arrays_cols):
            y_pred_df[c][i] = y_pred[i, :, :, i_c]

    # postprocessing
    y_pred_df = Data.postprocess_data(
        opt,
        X_train_df,
        y_train_df,
        X_valid_df,
        y_valid_df,
        X_test_df,
        y_test_df
    )

    # save
    y_pred_df.to_pickle(opt["working_dir"] + 'y_pred.csv')

    ### TODO:
    # implémenter postprocess_data
    # mode benchmark avec perf counter
    # reproduire plusieurs configurations avec les fichiers jsonc associés et les tester !
    # ajouter un logger
    # faire un dernier tri dans les fichiers