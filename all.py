# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:12:11 2019

@author: ashok.swarna
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.getcwd()
df = pd.read_excel("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\My_python\\ISCP_DATASET.xlsx")

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


sku_group = df.groupby('Product SKU', as_index=False)
sku_list = sku_group.groups.keys()

region_group = df.groupby('Region', as_index=False)
region_list = region_group.groups.keys()

state_group = df.groupby('State', as_index=False)
state_list = state_group.groups.keys()

begin = 0
end = 1                 
for Region in region_list:
    for state in state_list:
        for sku in sku_list:
            print()
            print(sku)

        # Select SKU to train & validate model
            df_sku = df[df['Product SKU'].isin([sku]) & df['State'].isin([state]) &
                        df['Region'].isin([Region])]
            

            # Select sales data for training & validation
            y = df_sku['Sales'].reset_index(drop=True)
            
            y1 = difference(y, 1)
            plt.figure()
            plt.plot(y)
            plt.plot(y1, color = 'Red')
            plt.show()
            
            
            
            
