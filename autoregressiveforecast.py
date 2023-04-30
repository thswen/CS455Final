# Author Thomas Swenson
# Date: 4/22/23
# autoregressiveforecast.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
# importing for forecasting
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster

# importing data from csv file
cwd = os.getcwd()
df = pd.read_csv(cwd + '/data/AAPL.csv')
# removing unnecessary columns
#df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop(['Volume'], axis=1)
df = df.drop(['Dividends'], axis=1)
df = df.drop(['Stock Splits'], axis=1)

# data preparation
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
df.set_index('Date')
df.asfreq('B')  # data was recorded on a business day basis
df = df.sort_index()  # makes sure the data is in chronological order


steps = 260  # average business days in a year

df_train = df[:-steps]
df_test = df[-steps:]

print(f"Train dates : {df_train['Date'].min()} --- {df_train['Date'].max()}")
print(f"Test dates  : {df_test['Date'].min()} --- {df_test['Date'].max()}")

forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=steps)
metric, backtest_predictions = backtesting_forecaster(
                              forecaster=forecaster,
                              y=df['Close'],
                              exog= df[['Open', 'High', 'Low']],  # not technically exog variables but reduces error
                              initial_train_size=(len(df_train)),
                              fixed_train_size=False,
                              steps=2,
                              refit=True,
                              metric='mean_absolute_percentage_error',
                              verbose=False
                          )
print("MAPE: ", metric)

# creating the backtest plot
fig, ax = plt.subplots(figsize=(11, 5))
df_test.loc[:, 'Close'].plot(ax=ax, linewidth=2, label='Test')
df_train.loc[:, 'Close'].plot(ax=ax, linewidth=2, label='Train')
backtest_predictions.plot(ax=ax, linewidth=2, label='Predicted')
ax.set_title('Close Price vs Predictions (Test Data)')
ax.set_ylabel('Price (USD)')
ax.legend()
plt.show()

# predicting future values
forecaster.fit(y=df['Close'])
future_predictions = forecaster.predict(steps=30)  # 30 days of forecast

# plotting the future forecast
fig1, ax1 = plt.subplots(figsize=(11,5))
df.loc[:, 'Close'].plot(ax=ax1, linewidth=2, label='Train')
future_predictions.plot(ax=ax1, linewidth=2, label='Future Predictions')
ax1.set_title('Historical Data vs Predicted')
ax1.set_ylabel('Price (USD)')
ax1.legend()
plt.show()
