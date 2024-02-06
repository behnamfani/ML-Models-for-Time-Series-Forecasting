import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def FE(
    Train: pd.DataFrame, Test: pd.DataFrame, which: str
) -> (pd.DataFrame, pd.DataFrame, MinMaxScaler):
    """
    Based on the choice,create some features and prepare the datesets for the models by scaling it into (-1, 1)
    :param Train: the Train dataset
    :param Test: the Test dataset
    :param which: add lag, window, or date to the datasets
    :return:
    """
    Trainset = Train[["meantemp"]].copy()
    Testset = Test[["meantemp"]].copy()

    if which == "date":
        Trainset["month"] = pd.to_datetime(Train.index).month
        Trainset["Week_n"] = pd.to_datetime(Train.index).isocalendar().week
        Trainset["Day"] = pd.to_datetime(Train.index).day
        Testset["month"] = pd.to_datetime(Test.index).month
        Testset["Week_n"] = pd.to_datetime(Test.index).isocalendar().week
        Testset["Day"] = pd.to_datetime(Test.index).day
        plt.figure(figsize=(6, 4))
        sns.heatmap(Trainset.corr())
        plt.show()
        Trainset = Trainset.to_numpy()
        Testset = Testset.to_numpy()
        scaler = ""

    else:
        if which == "lag":
            # Add lags (The past values are known as lags) to the dataset. PACF plays an important role to get the
            # right number of the lags
            for inc in range(1, 6):
                field_name = "lag_" + str(inc)
                Trainset[field_name] = Trainset["meantemp"].shift(inc)
                Testset[field_name] = Testset["meantemp"].shift(inc)

            Trainset = Trainset.dropna().reset_index(drop=True)
            Testset = Testset.dropna().reset_index(drop=True)

        elif which == "window":
            rolling_mean = Trainset.rolling(window=5).mean()
            rolling_max = Trainset.rolling(window=5).max()
            rolling_min = Trainset.rolling(window=5).min()
            (
                Trainset["rolling_mean"],
                Trainset["rolling_max"],
                Trainset["rolling_min"],
            ) = (rolling_mean, rolling_max, rolling_min)
            rolling_mean = Testset.rolling(window=5).mean()
            rolling_max = Testset.rolling(window=5).max()
            rolling_min = Testset.rolling(window=5).min()
            Testset["rolling_mean"], Testset["rolling_max"], Testset["rolling_min"] = (
                rolling_mean,
                rolling_max,
                rolling_min,
            )
            Trainset = Trainset.dropna().reset_index(drop=True)
            Testset = Testset.dropna().reset_index(drop=True)

        plt.figure(figsize=(6, 4))
        sns.heatmap(Trainset.corr())
        plt.show()
        # Scale the dataset
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(Trainset)
        Trainset = scaler.transform(Trainset)
        Testset = scaler.transform(Testset)

    return Trainset, Testset, scaler
