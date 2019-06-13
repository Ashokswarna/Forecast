# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:28:29 2019

@author: ashok.swarna
"""
import os
import pandas as pd

os.chdir("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\ISCP")

df = pd.read_csv("train.Csv")

df2 = df.head(1000000)
#df2.to_csv('Pred.Csv', index=False)

df2.isnull().sum().sort_values(ascending=False)
corr = df2.corr()
corr
df.tail(5)
