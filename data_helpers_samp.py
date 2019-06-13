# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:40:16 2019

@author: ashok.swarna
"""
project_home = u'C:\\Users\\ashok.swarna\\Downloads'
import os
os.getcwd()
os.chdir("C:\\Users\\ashok.swarna\\Downloads")


import pandas as pd
import numpy as np
import pickle
from os import path
from app_settings import read_config


def load_data(filename, skus, country, category):
    app_settings = read_config()
    data_path = app_settings['data_path']
    data_file_path = path.join(data_path, filename)

    df = pd.read_csv(data_file_path)

    df.rename(columns={'Sku': 'sku', 'Sales': 'actualVolume', 'Week': 'forecastWeek',
                       'Retailer': 'accountPlanningGroupCode', 'Market': 'market',
                       'Category': 'category'}, inplace=True)

    cols = ['sku', 'actualVolume', 'forecastWeek', 'accountPlanningGroupCode', 'market', 'category']

    df = df[df['market'] == country]
    df = df[df['category'] == category]
    df = df[df['sku'].isin(skus)]

    df = df[cols]

    df_sku_sales = df.groupby(['sku', 'forecastWeek'], as_index=False)['actualVolume'].sum()
    df_sku_sales['category'] = category
    df_sku_sales['market'] = country

    return df_sku_sales, df


def load_model():
    app_settings = read_config()
    model_path = app_settings['model_path']
    model_file_name = app_settings['model_file']
    model_file_path = path.join(model_path, model_file_name)
    model = pickle.load(open(model_file_path, 'rb'))
    return model
