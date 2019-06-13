# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:35:44 2019

@author: ashok.swarna
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

os.getcwd()
os.chdir("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\My_python")
new_df= pd.read_excel("Data_set_forecasting.xlsx")

new_df.drop(['Row ID', 'Order ID', 'Customer ID', 'Customer Name',
         'Postal Code'], axis = 1, inplace = True)


# Show top 10 Stores , max and min sales   

res_df = new_df.groupby(['Store no.']).sum()

res_df.sort_values('Sales',ascending=False,inplace=True)
res_df.head()
plt.bar(res_df.index.values, res_df['Sales'])
plt.title('Store V/S Sales')
plt.show()

# Show top 10 stores with profit

res_df.sort_values('Profit',ascending=False,inplace=True)
res_df.head()
plt.bar(res_df.index.values, res_df['Profit'])
plt.title('Store V/S Profit')
plt.show()

#showtop 10 stores with quantity
res_df.sort_values('Quantity',ascending=False,inplace=True)
res_df.head()
plt.bar(res_df.index.values, res_df['Quantity'])
plt.title('Store V/S Quantity')
plt.show()

#Month and year wise representation

new_df['Order Date'] = pd.to_datetime(new_df['Order Date'])
new_df['ordered_year'] = new_df['Order Date'].dt.year
new_df['ordered_Month'] = new_df['Order Date'].dt.month

date_df =  new_df[['Sales','Quantity','Discount','Profit',
                  'ordered_Month', 'ordered_year' ]].copy()

new_df.drop(['Store no.', 'ordered_year', 'ordered_Month'], axis = 1, 
             inplace = True)

date_df = date_df.groupby(['ordered_year','ordered_Month']).sum()
date_df = date_df.sort_values('Sales',ascending=False)

date_df.plot(kind='bar',figsize=(10,5),title='Graph show total sales and quatities at each month')
plt.ylabel('Quantity, total Amount')

#which ship mode most sales and profits

mode_df = new_df.groupby('Ship Mode').sum()

#segment wise sales and profits

segment_df = new_df.groupby('Segment').sum()

#citywise and state wise sales

city_df= new_df.groupby('City').sum()

state_df = new_df.groupby('State').sum()

#region wise 

region_df = new_df.groupby('Region').sum()

#category

category_df = new_df.groupby('Category').sum()




