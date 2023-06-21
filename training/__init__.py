from training.generator import DataGenerator
import data.load_data as ld
import training.losses as losses



def set_generators(
    X_train_df,
    y_train_df,
    X_valid_df,
    y_valid_df,
    X_test_df,
    y_test_df,
    training_opt
):
    """
    Creates 2 DataGenerators for training and validation (to avoid memory errors)
    and converts the test dataset into big numpy arrays.
    Inputs : X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df (pandas Dataframes)
        and training_opt (dict)
    Outputs: train_generator, valid_generator, (DataGenerators) X_test and y_test (numpy arrays)
    """
    train_generator = DataGenerator(X_train_df, y_train_df, training_opt["batch_size"])
    valid_generator = DataGenerator(X_valid_df, y_valid_df, training_opt["batch_size"])
    X_test , y_test = ld.df_to_array(X_test_df) , ld.df_to_array(y_test_df)

    return train_generator, valid_generator, X_test, y_test


def set_loss(training_opt):
    """
    Returns the loss that will be used for training.
    """
    if training_opt["loss"] == "mse":
        loss = "mse"
    elif training_opt["loss"] == "hybrid":
        loss = losses.mse_terre_mer(training_opt["frac"])
    elif training_opt["loss"] == "hybrid":
        loss = losses.modified_mse(training_opt["tau"], training_opt["eps"])
    else:
        raise NotImplementedError
    
    return loss