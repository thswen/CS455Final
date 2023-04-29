# Author Sean Huber
# Date: 2020-03-20
# regression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os
import time
import datetime
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve,LearningCurveDisplay
from sklearn.metrics._plot.confusion_matrix import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score
#importing data from csv file
cwd = os.getcwd()
df = pd.read_csv(cwd + '/data/SONY.csv')
#df.head()
#removing unnecessary columns
#df = df.drop(['Unnamed: 0'], axis=1)
dftime = df.loc[:,'Date']
df = df.drop(['Date'], axis=1)
df = df.drop(['Volume'], axis=1)
df = df.drop(['Dividends'], axis=1)
df = df.drop(['Stock Splits'], axis=1)
print(df.describe())
#splitting data into training and testing sets

X = df.loc[:,['Open','High','Low']].values
y = df.loc[:, 'Close'].values
X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(X, y,dftime, test_size=0.3,shuffle=False)

#scaling data

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting linear regression model

model = LinearRegression()
model.fit(X_train, y_train)

#predicting values

y_pred = model.predict(X_test)

#calculating RMSE

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

#calculating MAE

mae = np.mean(np.abs(y_pred - y_test))

#calculating R^2

r2 = sklearn.metrics.r2_score(y_test, y_pred)

#calculating scores

score = model.score(X_test, y_test)


print("RMSE: ", rmse)
print("MAE: ", mae)
print("r2: ", r2)
print("score: ", score)



#plotting results
def make_plot(): # Did this to easily comment out the plot to test other things
    pplot = pd.DataFrame(y_test, columns=['Actual'], index=time_test)
    pplot['Predicted'] = y_pred
    pplot.loc[:, 'Date'] = time_test
    pplot[['Actual','Predicted']].plot(figsize=(10,8),
    title="Linear Regression for SONY",
    legend=True,
    xlabel="Date",
    ylabel="Price",
    )
    plt.show()
    
make_plot()

#dftime.iloc[-1] last day in data holder
def days_from_last_date(last_date):
    today=datetime.datetime.utcnow()
    last_day_in_data=datetime.datetime.strptime(last_date,"%Y-%m-%d %H:%M:%S%z").replace(tzinfo=None)
    delta=today-last_day_in_data
    days = divmod(delta.total_seconds(), 86400)[0]
    return days
def predict_future_values(num_of_days):
    X_new=np.zeros(shape=(num_of_days,4)) #use previous data to predict future values recursive
    y_future=model.predict(X_new)
    return y_future
#days = int(days_from_last_date(dftime.iloc[-1]))
#print(days)
#y_future = predict_future_values(days)
#print(y_future)
def plot_future_values():
    plt.plot(np.zeros(days), y_future, color='blue') # Predicted values
    plt.title('Linear Regression fro APPL')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.figsize=(1600,1000)
    plt.show()
