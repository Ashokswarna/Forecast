# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:54:19 2019

@author: ashok.swarna
"""

from pmdarima.arima import auto_arima
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

os.getcwd()
df = pd.read_excel("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\My_python\\Texas.xlsx")

train_period = [
    ['1-1-2014', '10-31-2015']
]

train_period = [[datetime.strptime(y,'%m-%d-%Y') for y in x] for x in train_period]

validation_period = [
    ['11-1-2015', '12-31-2015']
]

validation_period = [[datetime.strptime(y,'%m-%d-%Y') for y in x] for x in validation_period]

begin = 0
end = 1

for tp in train_period:
    df_train_period = df[(df['Order_date'] >= tp[begin]) & (df['Order_date'] <= tp[end])]
    df_validation_period = df[(df['Order_date'] >= '11-1-2015')]
    y = df_train_period['Sales'].reset_index(drop=True)
    valid = df_validation_period['Sales'].reset_index(drop=True)
    y =y.astype('float32')
    length = len (y) + len (valid) + 30
    # Fit a simple auto_arima model
    arima = auto_arima(y, error_action='ignore', trace=1,
                       seasonal=True, m=7)
    # Plot actual test vs. forecasts:
    y_predict_log = arima.predict(n_periods=length)
    y_length = len(valid)
    y_pred = y_predict_log[-y_length:]
    plt.figure()
    plt.plot(valid)
    plt.plot(y_pred, color = 'Red')
    plt.show()
    mse = mean_squared_error(valid, y_pred)
 
# Now add the actual samples to the model and create NEW forecasts
arima.add_new_observations(valid)
new_preds, new_conf_int = arima.predict(n_periods=7, return_conf_int=True)
new_x_axis = np.arange(y.shape[0] + 10)

plt.plot(new_x_axis[:y_pred.shape[0]], y_pred, alpha=0.75)
plt.scatter(new_x_axis[y_pred.shape[0]:], new_preds, alpha=0.4, marker='o')
plt.fill_between(new_x_axis[-new_preds.shape[0]:],
                     new_conf_int[:, 0],
                     new_conf_int[:, 1],
                     alpha=0.1, color='g')
plt.set_title("Added new observed values with new forecasts")
plt.show()