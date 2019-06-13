# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:34:17 2019

@author: ashok.swarna
"""

def add_sub_mul(str1, str2):
    add1 = str1 + str2
    sub1 = str1 - str2
    mul1 = str1 * str2
    print ('result =  ' )
    print (add1, sub1, mul1)
    return add1 , sub1, mul1;

import sys
import numpy as np
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
        print (np.array(diff))
    return np.array(diff)

difference (d)
len(d)

d = (1,2,3,4,5,6,7,8,9)
import pands as pd
def _monthly_sku_forecast(df, category, country):
    sku_list = df.groupby('sku', as_index=False).groups.keys()
    print (sku_list)
    monthly_results = []

    for sku in sku_list:
        print (sku)
        df_sku = df[df['sku'].isin([sku])]
        # df_sku['forecastWeek'] = pd.to_datetime(df_sku['forecastWeek'].astype(str) + '1', format='%Y%W%w')
        df_sku['forecastWeek'] = df_sku.apply(lambda x: get_date_from_year_week(x['forecastWeek']), axis=1)

        df_sku = df_sku.groupby(['sku', df_sku['forecastWeek'].dt.to_period('m')]).sum().reset_index()
        df_sku['category'] = category
        df_sku['market'] = country
        df_sku.rename(columns={'forecastWeek': 'month'}, inplace=True)
        df_sku['month'] = df_sku['month'].apply(lambda x: str(x))
        monthly_results.append(df_sku)

    res_df = pd.concat(monthly_results)
    return res_df

_monthly_sku_forecast (d,'Tea', 'UK')
train_period = [
    [201422, 201621],
    [201427, 201626],
    [201431, 201630],
    [201435, 201634],
    [201440, 201639]
]
train_period[1]

import pickle

import os

from configparser import ConfigParser
def read_config(filename='config.ini', section='settings'):
    os.chdir('../config')
    parser = ConfigParser()
    parser.read(filename)
    configurations = {}

    if parser.has_section(section):
        items = parser.items(section)
        for item in items:
            configurations[item[0]] = item[1]
    else:
        raise Exception('{0} not found in {1} file'.format(section, filename))
    return configurations

import pandas as pd

df = pd.read_csv("C:\\Users\\ashok.swarna\\Downloads\\Unilever_Tea_data.csv")
print (df.describe())

def main():
    train(df)
    print(file_path)
    
os.getcwd()
os.chdir("C:\\Users\\ashok.swarna\\Downloads\\config")
    

def train(filename):
    """
        Trains ARIMA model post least MSE per sku & selects the best model and saves it
    :return: None
    """

    app_settings = read_config()
    data_path = app_settings['data_path']
    print (data_path, filename)
   # file_path = os.path.join(data_path, filename)
    #print(file_path)
    return filename




if __name__ == '__main__':
    main()
    























