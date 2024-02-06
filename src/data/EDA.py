import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from colorama import Fore, Style
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def EDA(Train: pd.DataFrame):
    """
    Exploratory data analysis, remove the outliers for each month w.r.t meantemp values and using interquartile range
    outlier removal
    :param Train: the Train dataset
    :return: Train set without outliers
    """
    df_temp = Train.copy()
    df_temp["Year"] = pd.to_datetime(df_temp.index).year
    df_temp["Month"] = pd.to_datetime(df_temp.index).month
    df_temp["Month Name"] = pd.to_datetime(df_temp.index).month_name()
    df_temp["Month-Year"] = pd.to_datetime(df_temp.index).strftime("%b.%Y")

    # Countplots
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    figure.suptitle("Countplot")
    sns.countplot(data=df_temp, x="Year", ax=axes[0])
    axes[0].set_title("Year", fontweight="bold")
    sns.countplot(data=df_temp, x="Month", ax=axes[1])
    axes[1].set_title("Month", fontweight="bold")
    axes[1].tick_params(axis="x", labelrotation=45)
    plt.show()

    # Drop duplicates
    df_temp.drop_duplicates(inplace=True)

    # Fill nan elements with the previous values
    logging.info("\nNumber of NaN values in each column:")
    logging.info(df_temp.isnull().sum())
    df_temp.fillna(method="ffill", inplace=True)

    # features correlations
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_temp.corr(numeric_only=True))
    plt.title("Correlation between different columns:", fontweight="bold")
    plt.show()

    # Boxplot
    plt.figure(figsize=(20, 4))
    sns.boxplot(
        data=df_temp, hue="Month-Year", y="meantemp", palette="Blues", legend=False
    )
    plt.title("Boxplot of meantemp", fontweight="bold")
    plt.xticks(rotation=60)
    plt.show()

    # Remove outliers of each month
    logging.info("\nRemoving outliers...")
    logging.info(
        f"Length of the Train dataset before removing the outliers: {len(df_temp)}"
    )
    for i in df_temp["Month-Year"].unique():
        dff_temp = df_temp.loc[df_temp["Month-Year"] == i].copy()
        Q1 = dff_temp["meantemp"].quantile(0.25)
        Q3 = dff_temp["meantemp"].quantile(0.75)
        IQR = Q3 - Q1

        indx = dff_temp.loc[
            (dff_temp["meantemp"] < (Q1 - 1.5 * IQR))
            | (dff_temp["meantemp"] > (Q3 + 1.5 * IQR))
        ].index.tolist()
        df_temp = df_temp.loc[~df_temp.index.isin(indx)]
    logging.info(
        f"Length of the Train dataset after removing the outliers: {len(df_temp)}"
    )

    return df_temp[Train.columns]


def Seasonal_Decomposition(Train: pd.DataFrame):
    """
    Plot the linear trend, the periodic (seasonal) component, and random residuals using statsmodels lib
    :param Train: the Train dataset
    """
    decomposition = seasonal_decompose(x=Train["meantemp"], model="additive", period=6)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    dic = {"Trend": trend, "Seasonal": seasonal, "Residual": residual}
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))
    for counter, i in enumerate(dic.keys()):
        axes[counter].plot(dic[i], label=i)
        # Define x-axis to be month and year
        axes[counter].xaxis.set_major_locator(mdates.MonthLocator())
        axes[counter].xaxis.set_major_formatter(mdates.DateFormatter("%b.%Y"))
        axes[counter].tick_params(axis="x", labelrotation=45)
        axes[counter].legend()
        axes[counter].set_title(i)
    plt.xlabel("Days from 01.01.2013 to 01.01.2017")
    plt.show()


def check_Stationarity(Train: pd.DataFrame, add_diff: bool = True):
    """
    The mean of a stationary series does not show any significant trend or fluctuation as it is important for reliable
    analysis and modeling. A non-stationary time series can be converted to a stationary time series through differencing
    (here we use first order differencing if the add_diff is true)
    :param add_diff: whether to add the difference of the non-stationary features
    :param Train: the Train dataset
    :return:
    """
    # Dickey-Fuller test
    for i in Train.columns:
        if "diff" not in i:
            logging.info(f" --> Dickey-Fuller test on {i}")
            adf, pval, usedlag, nobs, crit_vals, icbest = adfuller(Train[i].values)
            logging.info("ADF test statistic:", adf)
            logging.info("ADF p-values:", pval)
            logging.info("ADF number of lags used:", usedlag)
            logging.info("ADF number of observations:", nobs)
            logging.info("ADF critical values:", crit_vals)
            logging.info("ADF best information criterion:", icbest)
            # Check the conditions
            if (pval <= 0.05) & (crit_vals["5%"] > adf):
                logging.info(f"{Fore.GREEN}\033[1mStationary\033[0m{Style.RESET_ALL}")
            elif add_diff:
                logging.info(f"{Fore.RED}\033[1mNon-Stationary\033[0m{Style.RESET_ALL}")
                cl = i + " diff"
                Train[cl] = Train[i].diff()

            return Train
