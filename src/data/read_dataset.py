import logging

import numpy as np
import pandas as pd


def read_data(Train_Path: str, Test_Path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    :param Train_Path: path to the Train set
    :param Test_Path: path to the Test set
    :return: Train and Test dataframes
    """
    Train = pd.read_csv(Train_Path)
    Test = pd.read_csv(Test_Path)

    Train.set_index("date", inplace=True)
    Train.index = pd.to_datetime(Train.index).date
    Test.set_index("date", inplace=True)
    Test.index = pd.to_datetime(Test.index).date
    logging.info(Train.head())
    Train.info()
    Train.describe()
    return Train, Test


def unscaled(pred: np.array, X: np.array, scaler):
    """
    Inverse Transformation and unscaled the prediction to plot the unscaled result
    :param pred: the prediction result
    :param X: the data array
    :param scaler: the scaler used for MinMax scaling, used for unscaling the result
    :return: the unscaled data arrays
    """
    pred_test_set = []
    for i in range(len(pred)):
        pred_test_set.append(
            np.concatenate(
                (
                    np.array(pred[i]).reshape(
                        1,
                    ),
                    X[i],
                )
            )
        )

    unscaled_pred_test_set = scaler.inverse_transform(np.array(pred_test_set))
    return unscaled_pred_test_set
