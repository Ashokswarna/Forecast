# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:37:42 2019

@author: ashok.swarna
"""
import fitter
import pandas as pd
import os

os.chdir("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\BOSCH")
df = pd.read_excel("Bosch_data.xlsx")

df2 = df.groupby(['Material','From Date', 'To Date'])['          Opening Stock'].sum().reset_index()
df3 = df.groupby(['Material','From Date', 'To Date'])['    Total Receipt Qties'].sum().reset_index()
df4 = df.groupby(['Material','From Date', 'To Date'])[' Total Issue Quantities'].sum().reset_index()
df5 = df.groupby(['Material','From Date', 'To Date'])['          Closing Stock'].sum().reset_index()

#df2.drop(['Material','From Date', 'To Date'], axis = 1, inplace = True)
df3.drop(['Material','From Date', 'To Date'], axis = 1, inplace = True)
df4.drop(['Material','From Date', 'To Date'], axis = 1, inplace = True)
df5.drop(['Material','From Date', 'To Date'], axis = 1, inplace = True)

df2.rename({'          Opening Stock':'agg_opening_stock'}, axis= 1, inplace = True)
df3.rename({'    Total Receipt Qties':'total_receipt_quantities'},axis= 1, inplace = True)
df4.rename({' Total Issue Quantities':'agg_total_issue_quantities'}, axis= 1, inplace = True)
df5.rename({'          Closing Stock':'agg_closing_stock'}, axis= 1, inplace = True)

#a=df2.agg_opening_stock
b=df3.total_receipt_quantities
c=df4.agg_total_issue_quantities
d=df5.agg_closing_stock

df2['total_receipt_quantities'] = b
df2['agg_total_issue_quantities'] = c
df2['agg_closing_stock'] = d


df2.to_csv('bosch_agg.Csv', index=False)

from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

y_ = df2.agg_total_issue_quantities
y_ = abs(y_)

#Organize Data
SR_y = pd.Series(y_, name="y_ (Target Vector Distribution)")


#Plot Data
fig, ax = plt.subplots()
sns.distplot(SR_y, bins=25, color="g", ax=ax)
plt.show()


#group by material
material_group = df2.groupby('Material', as_index=False)
material_list = material_group.groups.keys()
material_list


for material in material_list:
    df1 = df2[df2['Material'].isin ([material])].reset_index(drop=True)
    df1.head(5)
    y = df1.agg_total_issue_quantities
    y = abs(y)
    #Organize Data
    SR_y = pd.Series(y, name="y (Target Vector Distribution)")
    #Plot Data
    fig, ax = plt.subplots()
    sns.distplot(SR_y, bins=25, color="g", ax=ax, label= material)
    plt.legend()
    plt.show()

