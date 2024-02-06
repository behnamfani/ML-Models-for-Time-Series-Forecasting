from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import src.models.models as model
import src.visualization.visualize as vis
from src.data.EDA import *
from src.data.read_dataset import *
from src.features.build_features import *

if __name__ == "__main__":
    # Read the dataset and plot its features
    folder_path = (
        r"B:/Codes/Python/CODE/ML-Models-for-Time-Series-Forecasting-main/ml-models-for-time-series"
        r"-forecasting"
    )
    Train_path = f"{folder_path}/data/DailyDelhiClimateTrain.csv"
    Test_path = f"{folder_path}/data/DailyDelhiClimateTest.csv"
    Train, Test = read_data(Train_path, Test_path)
    vis.plot_train_df(Train)

    # Analyze the dataset and remove the outliers
    Train = EDA(Train)
    # plot the linear trend, the periodic (seasonal) component, and random residuals
    Seasonal_Decomposition(Train)
    # Check the stationarity of the dataset
    Train = check_Stationarity(Train)

    # AFC and PACF: the autocorrelation and partial autocorrelation functions for meantemp
    # plots for finding the values p, q and r for Autoregressive(AR) and Moving Average (MA) models for AMIRA and SAMIRA
    logging.info("meantemp")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    acf = plot_acf(Train["meantemp"], lags=30, ax=axes[0][0])
    pacf = plot_pacf(Train["meantemp"], lags=30, method="ywm", ax=axes[0][1])
    if "meantemp diff" in Train.columns:
        acf = plot_acf(Train["meantemp diff"].dropna(), lags=30, ax=axes[1][0])
        pacf = plot_pacf(
            Train["meantemp diff"].dropna(), lags=30, method="ywm", ax=axes[1][1]
        )
    plt.show()
    logging.info("meantemp diff")

    # Create extra features such as lag, window, or date
    Trainset, Testset, scaler = FE(
        Train, Test, which="lag"
    )  # which = 'lag' or 'date' or 'window'

    # Models
    model.Train = Train
    model.Test = Test
    vis.Test = Test
    # Linear Regression
    model.linearRegression(Trainset, Testset, scaler)
    # Random Forest
    model.RandomForest(Trainset, Testset, scaler, Max_depth=5)
    # XGboost
    model.XGBoost(Trainset, Testset, scaler, Max_depth=3, lr=0.05)
    # Prophet
    model.Prophet_FB()
    # Compare results
    df_result = model.df_result
    logging.info(df_result)
    fig = plt.figure(figsize=(12, 4))
    for i in df_result.drop(columns=["Model"]).columns:
        plt.plot(df_result["Model"], df_result[i], label=i)
    plt.xlabel("Models")
    plt.xticks(rotation=60)
    plt.ylabel("Error")
    plt.legend()
    plt.show()
