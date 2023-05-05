import random
import numpy as np
import pandas as pd
from preprocessing.load_data import get_arrays_cols, param_to_array


# Dans le cas de deux df de même tailles
def extract_patches(X_df, y_df, patch_h, patch_w, n_patches):
    arrays_cols_X = get_arrays_cols(X_df)
    arrays_cols_y = get_arrays_cols(y_df)
    img_h = X_df[arrays_cols_X[0]][0].shape[0]
    img_w = X_df[arrays_cols_X[0]][0].shape[1]

    X_df_out = pd.DataFrame(
        [],
        columns = X_df.columns
    )
    y_df_out = pd.DataFrame(
        [],
        columns = y_df.columns
    )

    X = np.zeros((len(arrays_cols_X), img_h, img_w))
    y = np.zeros((len(arrays_cols_y), img_h, img_w))
    for i in range(len(X_df)):
        for i_x in range(len(arrays_cols_X)):
            X[i_x, :, :] = X_df[arrays_cols_X[i_x]][i]
        for i_y in range(len(arrays_cols_y)):
            y[i_y, :, :] = y_df[arrays_cols_y[i_y]][i]
        
        for k in range(n_patches):
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            patch_X = X[:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_y = y[:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]

            # ajout d'une ligne dans le dataset:
            values_X = []
            values_y = []
            for j in range(patch_X.shape[0]):
                values_X.append(patch_X[j, :, :].reshape((patch_X.shape[1], patch_X.shape[2])))
            for j in range(patch_y.shape[0]):
                values_y.append(patch_y[j, :, :].reshape((patch_y.shape[1], patch_y.shape[2])))
            X_df_i = pd.DataFrame(
                [[X_df.dates[i], X_df.echeances[i]] + values_X],
                columns=X_df.columns
            )
            y_df_i = pd.DataFrame(
                [[y_df.dates[i], y_df.echeances[i]] + values_y],
                columns=y_df.columns
            )
            X_df_out = pd.concat([X_df_out, X_df_i])
            y_df_out = pd.concat([y_df_out, y_df_i])
    return X_df_out.reset_index(drop=True), y_df_out.reset_index(drop=True)


