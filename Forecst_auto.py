
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:20:39 2019

@author: ashok.swarna
"""
import os
import pandas as pd
import datetime

os.chdir("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\My_python")
os.listdir(os.getcwd())
df = pd.read_excel('Dataset.Xlsm')


furniture = df.loc[df['Category'] == 'Furniture']

office_supplies = df.loc[df['Category'] == 'Office Supplies']

technology = df.loc[df['Category'] == 'Technology']