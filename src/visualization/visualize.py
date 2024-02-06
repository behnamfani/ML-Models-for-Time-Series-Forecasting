import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.data.read_dataset import unscaled

Test = None


def result_plot(model, Y, Y_test, preds_Train, preds_Test, X_test, scaler):
    """
    Plot the actual vs prediction results
    :param model: name of the model
    :param Y: actual train values
    :param Y_test: actual test values
    :param preds_Train: prediction array for the train
    :param preds_Test: prediction array for the test
    :param X_test: X values of the test set
    :param scaler: the scaler used for MinMax scaling, used for unscaling the result
    """
    global Test

    # Plot the scaled results
    fig, axes = plt.subplots(
        1, 2, figsize=[22, 4], gridspec_kw={"width_ratios": [2, 1]}
    )
    # Trainset
    axes[0].plot(Y, "darkslategray", label="Original")
    axes[0].plot(preds_Train, "aqua", label="Predicted")
    axes[0].set_ylabel("scaled_meantemp")
    axes[0].set_xlabel("Days from 01.01.2013")
    axes[0].legend()
    axes[0].set_title(f"{model} Trainset", fontweight="bold")
    # Testset
    axes[1].plot(Y_test, "darkslategray", label="Original")
    axes[1].plot(preds_Test, "aqua", label="Predicted")
    axes[1].set_ylabel("scaled_meantemp")
    axes[1].set_xlabel("Days from 01.01.2017")
    axes[1].legend()
    axes[1].set_title(f"{model} Testset scaled if used", fontweight="bold")
    plt.show()

    if scaler != "":  # Plot the unscale values for Test as well
        unscaled_pred_test_set = unscaled(preds_Test, X_test, scaler)
        df_result = pd.DataFrame({"meantemp": unscaled_pred_test_set[:, 0]})
        df_result.index = Test.index[-len(unscaled_pred_test_set) :]
        plt.figure(figsize=(8, 4))
        ax = plt.subplot()
        ax.plot(df_result["meantemp"], label="Predicted")
        ax.plot(Test["meantemp"], label="Original")
        ax.set_ylabel("meantemp")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b.%Y"))
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_title(f"{model} Unscaled Test actual vs prediction")
        ax.legend()
        plt.show()


def plot_train_df(Train: pd.DataFrame):
    """
    Plot different features, their rolling averages and standard deviations
    :param Train: the train dataset
    """
    fig, axes = plt.subplots(4, 1, figsize=(20, 24))
    for counter, i in enumerate(Train.columns):
        axes[counter].plot(Train[i], "g", marker="o", label=i)
        axes[counter].plot(
            Train[i].rolling(window=30).mean(), "red", label="Rolling Mean"
        )
        axes[counter].plot(
            Train[i].rolling(window=30).std(),
            "gold",
            label="Rolling Standard deviation",
        )
        # Define x-axis to be month and year
        axes[counter].xaxis.set_major_locator(mdates.MonthLocator())
        axes[counter].xaxis.set_major_formatter(mdates.DateFormatter("%b.%Y"))
        axes[counter].tick_params(axis="x", labelrotation=45)
        axes[counter].legend()
    plt.show()
