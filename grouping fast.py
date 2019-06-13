# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:44:35 2019

@author: ashok.swarna
"""

import pandas as pd
import os

os.chdir("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\BOSCH")
df = pd.read_excel("fast_pace_bosch.xlsx")

df2 = df.groupby(['Material','From Date', 'To Date'])['          Opening Stock'].sum().reset_index()
df3 = df.groupby(['Material','From Date', 'To Date'])['    Total Receipt Qties'].sum().reset_index()
df4 = df.groupby(['Material','From Date', 'To Date'])[' Total Issue Quantities'].sum().reset_index()
df5 = df.groupby(['Material','From Date', 'To Date'])['          Closing Stock'].sum().reset_index()


df3.drop(['Material','From Date', 'To Date'], axis = 1, inplace = True)
df4.drop(['Material','From Date', 'To Date'], axis = 1, inplace = True)
df5.drop(['Material','From Date', 'To Date'], axis = 1, inplace = True)

df2.rename({'          Opening Stock':'opening_stock'}, axis= 1, inplace = True)
df3.rename({'    Total Receipt Qties':'receipt_quantities'},axis= 1, inplace = True)
df4.rename({' Total Issue Quantities':'total_issue_quantities'}, axis= 1, inplace = True)
df5.rename({'          Closing Stock':'closing_stock'}, axis= 1, inplace = True)

#a=df2.agg_opening_stock
b=df3.receipt_quantities
c=df4.total_issue_quantities
d=df5.closing_stock

df2['receipt_quantities'] = b
df2['total_issue_quantities'] = c
df2['closing_stock'] = d


df2.to_csv('fast_pace_bosch.Csv', index=False)