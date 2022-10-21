# ML-Models-for-Time-Series-Forecasting

Time series data is a sequence of data points indexed in time order and methods for predicting them are different than normal data. In this project, I try different feature engineering methods and different models to see their performances in forecasting the mean temperature of Dehli [kaggle](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data). For more description, figures, and other parts like Exploratory data analysis see the script.

Models:
1. Linear Regression
2. Random Forest
3. XGBoost
4. [Prophet](https://facebook.github.io/prophet/) and [NeuralProphet](https://github.com/ourownstory/neural_prophet)
5. SARIMA

Dataset:
The dataset contains meantemp, humidity, wind_speed, meanpressure, and the dates. I use and predict the meantemp feature. For the first 3 models, I create 3 different datasets with different features based on [medium](https://medium.com/data-science-at-microsoft/introduction-to-feature-engineering-for-time-series-forecasting-620aa55fcab0)

The first method is to simply use different date attributes:
```python
    Trainset['month'] = pd.to_datetime(Train.index).month
    Trainset['Week_n'] = pd.to_datetime(Train.index).isocalendar().week
    Trainset['Day'] = pd.to_datetime(Train.index).day
    Testset['month'] = pd.to_datetime(Test.index).month
    Testset['Week_n'] = pd.to_datetime(Test.index).isocalendar().week
    Testset['Day'] = pd.to_datetime(Test.index).day
```

The second method is to use lags. Lag features are values at prior timesteps that are considered useful because they are created on the assumption that what happened in the past can influence or contain a sort of intrinsic information about the future.
```python
      # Add lags (The past values are known as lags) to the dataset. PACF plays an important role to get the right number of the lags
      for inc in range(1,6):
          field_name = 'lag_' + str(inc)
          Trainset[field_name] = Trainset['meantemp'].shift(inc)
          Testset[field_name] = Testset['meantemp'].shift(inc)
```

The third method is to use a rolling window. The main goal of building and using rolling window statistics in a time series dataset is to compute statistics on the values from a given data sample by defining a range that includes the sample itself as well as some specified number of samples before and after the sample used.
```python
      rolling_mean = Trainset.rolling(window=5).mean()
      rolling_max = Trainset.rolling(window=5).max()
      rolling_min = Trainset.rolling(window=5).min()
      Trainset['rolling_mean'], Trainset['rolling_max'], Trainset['rolling_min'] = rolling_mean, rolling_max, rolling_min
      rolling_mean = Testset.rolling(window=5).mean()
      rolling_max = Testset.rolling(window=5).max()
      rolling_min = Testset.rolling(window=5).min()
      Testset['rolling_mean'], Testset['rolling_max'], Testset['rolling_min'] = rolling_mean, rolling_max, rolling_min
```

E.g. using dataset created by the lags
|meantemp|	lag_1|	lag_2|	lag_3|	lag_4|	lag_5|
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
0|	7.000000|	6.000000|	8.666667|	7.166667|	7.400000|	10.000000|
1|	7.000000|	7.000000|	6.000000|	8.666667|	7.166667|	7.400000|
2|	8.857143|	7.000000|	7.000000|	6.000000|	8.666667|	7.166667|
3|	14.000000|	8.857143|	7.000000|	7.000000|	6.000000|	8.666667|
4|	11.000000|	14.000000|	8.857143|	7.000000|	7.000000|	6.000000|
