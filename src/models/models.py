import time

import xgboost as xgb
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data.read_dataset import *
from src.visualization.visualize import *

Test, Train = [None] * 2
df_result = pd.DataFrame(
    columns=["Model", "MAE_Train", "RMSE_Train", "MAE_Test", "RMSE_Test"]
)


# MAE and RMSE scores
def compare_results(model, preds_Train, preds_Test, train, test, scaler):
    """
    Compute the mean absolute error and root mean squared error for both train and test sets
    Save the result in df_result
    :param model: name of the model
    :param preds_Train: prediction array for the train
    :param preds_Test: prediction array for the test
    :param train: actual values for train
    :param test: actual values for test
    :param scaler: the scaler used for MinMax scaling, used for unscaling the result
    """
    global df_result, Test, Train

    if scaler != "":
        preds_Train = unscaled(preds_Train, train, scaler)
        preds_Test = unscaled(preds_Test, test, scaler)
        d_Train = Train.iloc[-len(preds_Train) :]["meantemp"].values - preds_Train[:, 0]
        d_Test = Test.iloc[-len(preds_Test) :]["meantemp"].values - preds_Test[:, 0]
    else:
        d_Train = train - preds_Train
        d_Test = test - preds_Test
    mae_Train = np.mean(abs(d_Train))
    rmse_Train = np.sqrt(np.mean(d_Train**2))
    mae_Test = np.mean(abs(d_Test))
    rmse_Test = np.sqrt(np.mean(d_Test**2))
    df_result.loc[len(df_result)] = [model, mae_Train, rmse_Train, mae_Test, rmse_Test]
    logging.info(
        f"{model}: MAE_Train: {mae_Train} - RMSE_Train {rmse_Train} - MAE_Test {mae_Test} - RMSE_Test {rmse_Test}"
    )


def linearRegression(Trainset: np.array, Testset: np.array, scaler):
    """
    linear Regression model, plot the results, and calculate the errors
    :param Trainset: the Train numpy array
    :param Testset: the Test numpy array
    :param scaler: the scaler used for MinMax scaling
    """
    X = np.array(Trainset[:, 1:])
    Y = np.array(Trainset[:, 0:1]).reshape(len(Trainset[:, 0:1]))
    X_test = np.array(Testset[:, 1:])
    Y_test = np.array(Testset[:, 0:1]).reshape(len(Testset[:, 0:1]))
    # Train model
    mlr = LinearRegression()
    mlr.fit(X, Y)
    # Trainset
    preds_Train = mlr.predict(X)
    # Testset
    preds_Test = mlr.predict(X_test)
    # plot the predictions
    result_plot("Linear Regression", Y, Y_test, preds_Train, preds_Test, X_test, scaler)
    # scores
    now = time.localtime()
    t = str(now.tm_min) + "." + str(now.tm_sec)
    compare_results(
        f"Linear Regression ({t})", preds_Train, preds_Test, X, X_test, scaler
    )


def RandomForest(Trainset: np.array, Testset: np.array, scaler, Max_depth: int):
    """
    Random Forest model, plot the results, and calculate the errors
    :param Trainset: the Train numpy array
    :param Testset: the Test numpy array
    :param scaler: the scaler used for MinMax scaling
    :param Max_depth: Maximum depth of the trees
    """
    X = np.array(Trainset[:, 1:])
    Y = np.array(Trainset[:, 0:1]).reshape(len(Trainset[:, 0:1]))
    X_test = np.array(Testset[:, 1:])
    Y_test = np.array(Testset[:, 0:1]).reshape(len(Testset[:, 0:1]))
    # Train model
    rf = RandomForestRegressor(max_depth=Max_depth)
    rf.fit(X, Y)
    # Trainset
    preds_Train = rf.predict(X)
    # Testset
    preds_Test = rf.predict(X_test)
    # plot the predictions
    result_plot("Random Forest", Y, Y_test, preds_Train, preds_Test, X_test, scaler)
    # scores
    now = time.localtime()
    t = str(now.tm_min) + "." + str(now.tm_sec)
    compare_results(
        f"Random Forest max_depth={Max_depth} ({t})",
        preds_Train,
        preds_Test,
        X,
        X_test,
        scaler,
    )


def XGBoost(Trainset, Testset, scaler, lr, Max_depth):
    """
    XGBoost model, plot the results, and calculate the errors
    :param Trainset: the Train numpy array
    :param Testset: the Test numpy array
    :param scaler: the scaler used for MinMax scaling
    :param Max_depth: Maximum depth of the trees
    :param lr: learning rate
    """
    X = np.array(Trainset[:, 1:])
    Y = np.array(Trainset[:, 0:1]).reshape(len(Trainset[:, 0:1]))
    X_test = np.array(Testset[:, 1:])
    Y_test = np.array(Testset[:, 0:1]).reshape(len(Testset[:, 0:1]))
    # Train model
    XGb = xgb.XGBRegressor(
        objective="reg:squarederror", learning_rate=lr, max_depth=Max_depth
    )
    XGb.fit(X, Y)
    # Trainset
    preds_Train = XGb.predict(X)
    # Testset
    preds_Test = XGb.predict(X_test)
    # plot the predictions
    result_plot("XGBoost", Y, Y_test, preds_Train, preds_Test, X_test, scaler)
    # scores
    now = time.localtime()
    t = str(now.tm_min) + "." + str(now.tm_sec)
    compare_results(
        f"XGBoost lr={lr} & max_depth={Max_depth} ({t})",
        preds_Train,
        preds_Test,
        X,
        X_test,
        scaler,
    )


def SARIMA():
    global Train, Test

    df = Train[["meantemp"]]
    df.index = pd.DatetimeIndex(Train.index).to_period("D")
    model = SARIMAX(
        df["meantemp"], order=(1, 1, 3), seasonal_order=(1, 1, 1, 365)
    ).fit()
    logging.info(model.summary())
    # Diagnosing the model residuals
    model.plot_diagnostics(figsize=(10, 8))
    forecast = model.forecast(Test.index[-1])
    logging.info(forecast)


def Prophet_FB():
    global Train, Test
    # Dataframe for Prophet should only have ds and y columns
    df = Train[["meantemp"]]
    df.rename(columns={"meantemp": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(Train.index)
    future = pd.DataFrame()
    future["ds"] = pd.to_datetime(Test.index)

    # Train Prophet
    model = Prophet(weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    forecast_1 = model.predict(df)
    forecast_2 = model.predict(future)
    model.plot_components(forecast_2)
    # plot the result
    fig, axes = plt.subplots(
        1, 2, figsize=[22, 6], gridspec_kw={"width_ratios": [2, 1]}
    )
    # Trainset
    model.plot(forecast_1, xlabel="Date", ylabel="Temp", ax=axes[0])
    axes[0].set_title(f"Prophet Trainset", fontweight="bold")
    # Testset
    model.plot(forecast_2, xlabel="Date", ylabel="Temp", ax=axes[1])
    axes[1].set_title(f"Prophet Testset", fontweight="bold")
    plt.show()
    # Scores
    now = time.localtime()
    t = str(now.tm_min) + "." + str(now.tm_sec)
    compare_results(
        f"Prophet ({t})",
        forecast_1["yhat"].values,
        forecast_2["yhat"],
        df["y"].values,
        Test["meantemp"].values,
        "",
    )
