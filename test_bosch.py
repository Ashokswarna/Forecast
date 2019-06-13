# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:34:49 2019

@author: ashok.swarna
"""

import warnings; warnings.simplefilter('ignore')
# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sns
import time
import pandas_profiling as prof
import pivottablejs as pvt
import datetime as dt
import os
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf
#import keras.backend as K
#import tensorflow as tf
import statistics
from math import sqrt

df=pd.read_csv(r'C:\Users\ashok.swarna\bosch_agg.csv')
df.head()

def Custom_loss2(y_true, y_pred, df_for_loss, price, sample_weight=None, multioutput='uniform_average'):
    
   # y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
   # check_consistent_length(y_true, y_pred, sample_weight)
    df_for_loss['y_pred']=y_pred
    df_for_loss['agg_closing_stock'] = np.where(df_for_loss.agg_closing_stock> 0, (price/2)*df_for_loss.agg_closing_stock, df_for_loss.agg_closing_stock)
    
    df_for_loss['agg_closing_stock'] = np.where(df_for_loss.agg_closing_stock== 0, price*df_for_loss.y_pred, df_for_loss.agg_closing_stock)
    df_for_loss.agg_closing_stock= df_for_loss.agg_closing_stock.astype('float64', raise_on_error = False)
    data=tf.convert_to_tensor(df_for_loss.agg_closing_stock)
    y_pred=K.tf.math.multiply(y_pred, data)
    #ch
    #cs
    output_errors = np.average((y_true - y_pred) ** 2, axis=0,
                               weights=sample_weight)
    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)

def Custom_loss(y_true, y_pred, df_for_loss, price, sample_weight=None, multioutput='uniform_average'):
    
   # y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
   # check_consistent_length(y_true, y_pred, sample_weight)
    y_pred.reset_index(inplace=True, drop=True)
    df_for_loss['y_pred']=y_pred
    df_for_loss['agg_closing_stock'] = np.where(df_for_loss.agg_closing_stock> 0, (price/2)*df_for_loss.agg_closing_stock, df_for_loss.agg_closing_stock)
    
    df_for_loss['agg_closing_stock'] = np.where(df_for_loss.agg_closing_stock== 0, price*df_for_loss.y_pred, df_for_loss.agg_closing_stock)
    df_for_loss.agg_closing_stock= df_for_loss.agg_closing_stock.astype('float32', raise_on_error = False)
   # y_pred= y_pred.astype('float32', raise_on_error = False)
   # data=tf.convert_to_tensor(df_for_loss.agg_closing_stock)
   # y_pred=K.tf.math.multiply(y_pred,data)
    #ch
    #cs
    y_pred= y_pred * df_for_loss.agg_closing_stock
    output_errors = np.average((y_true - y_pred) ** 2, axis=0,
                               weights=sample_weight)
   # if isinstance(multioutput, string_types):
    #    if multioutput == 'raw_values':
     #       return output_errors
      #  elif multioutput == 'uniform_average':
       #      pass None as weights to np.average: uniform mean
        #     multioutput = None

    #return np.average(output_errors, weights=multioutput)
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return statistics.mean(sqrt(y_pred - y_true), axis=-1)
    return output_errors


df['timestamp'] = pd.to_datetime(df['To_Date'], format = '%m/%d/%Y')
print(df.describe())

material = 'M303.160.117'
# Select SKU to train & validate model
df_mat = df[df['Material'].isin([material])]


df_for_loss = df_mat[['agg_closing_stock','Total_Issue_quantities']]
df_for_loss = df_for_loss.reset_index(drop=True)
df_for_loss.head()


from statsmodels.tsa.stattools import adfuller
result = adfuller(df_mat.Total_Issue_quantities)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
model = ARIMA(df_mat.Total_Issue_quantities, order=(5,0,0))
model_fit = model.fit(disp=0)

#y_predict_log = model.predict(start= 1, end=24, exog=None, dynamic=False)
y_predict_log = model_fit.predict(start=1, end=24, exog=None, dynamic=False)

y_pred = y_predict_log

price = 168.39
mse = Custom_loss(df_mat.Total_Issue_quantities, y_pred, df_for_loss,price)

