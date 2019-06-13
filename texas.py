# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:26:30 2019

@author: ashok.swarna
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA, _arma_predict_out_of_sample
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from datetime import datetime
from sklearn.metrics import mean_squared_error
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

#y = df.Sales
#plt.figure()
#y.plot()
#plt.figure()
#y.hist()

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob


for tp in train_period:
    df_train_period = df[(df['Order_date'] >= tp[begin]) & (df['Order_date'] <= tp[end])]
    df_validation_period = df[(df['Order_date'] >= '11-1-2015')]
    y = df_train_period['Sales'].reset_index(drop=True)
    valid = df_validation_period['Sales'].reset_index(drop=True)
    y =y.astype('float32')
    
    plt.figure()
    plot_acf(y, ax=plt.gca())
    plt.figure()
    plot_pacf(y, ax=plt.gca())
    
    y1 = difference(y, 1)
    plt.figure()
    plt.plot(y1)
    plt.show()
    
    plt.figure()
    plot_acf(y1, ax=plt.gca())
    plt.figure()
    plot_pacf(y1, ax=plt.gca())
    
    model = ARIMA(y1, order=(0, 0, 0))
    results_ARIMA = model.fit(disp=-1) 
    y_predict_log = results_ARIMA.predict(start=0, end=365)
    
    # invert the difference
    inverted = [inverse_difference(y[i], y1[i]) for i in range(len(y1))]
    plt.plot(inverted, color = 'black')
    plt.show()
    
    y_length = len(valid)
    y_pred = inverted[-y_length:]
    mse = mean_squared_error(y_pred, valid)
    plt.figure()
    plt.plot(valid)
    plt.plot(y_pred, color = 'Red')
    plt.show()
 