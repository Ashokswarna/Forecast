# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:31:48 2019

@author: ashok.swarna
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA, _arma_predict_out_of_sample
from datetime import datetime
from sklearn.model_selection import train_test_split

os.getcwd()
df = pd.read_excel("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\My_python\\ISCP_final_Dataset.xlsx")

sku_group = df.groupby('Product SKU', as_index=False)
sku_list = sku_group.groups.keys()

region_group = df.groupby('Region', as_index=False)
region_list = region_group.groups.keys()

state_group = df.groupby('State', as_index=False)
state_list = state_group.groups.keys()

#3 Regions
#4 States
#2 SKU


y = df.Sales
x = df.Order_date     

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



for Region in region_list:
    for state in state_list:
        for sku in sku_list:
            print()
            print(sku, state, Region)
            df_sku = df[df['Product SKU'].isin([sku])]
            period_index = 0
            y = df_sku.Sales
            model = ARIMA(y_train, order=(2, 2, 0))
            results_ARIMA = model.fit(disp=-1)
            plt.figure()
            plt.plot(X_test)
            plt.plot(results_ARIMA.fittedvalues, color = 'Red')
            plt.show()
            
df['Order_date'] = pd.to_datetime(df['Order_date'])
df['Day'] = df['Order_date'].dt.day
df['Month'] = df['Order_date'].dt.month
df['Year'] = df['Order_date'].dt.year

import calendar
df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[x])


date = pd.concat([df['Day'], df['Month'], df['Year'] ], axis=1)
date = df['Day'] + df['Month'] + df['Year']


df.insert(loc = 8, column = 'Date', value = date)
