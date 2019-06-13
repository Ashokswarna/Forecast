# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:56:58 2019

@author: ashok.swarna
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
os.getcwd()
df = pd.read_excel("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\My_python\\Dataset_forecasting.xlsx")


comments = ((df['positive_comments']/(df['positive_comments'] + df['negative_comments']))*100)
df.insert(loc = 30, column = 'Comments', value = comments)
df = df.assign(Reviews=pd.cut(df.Comments,[0,40,53,100],
                              labels=['Negative','Neutral','Positive']))

df.drop(['Order ID','Customer ID', 'Customer Name',
         'Product Name', 'Product ID', 'Store no.',
         'Ship Date', 'Country', 'Postal Code', 'positive_comments',
        'negative_comments', 'Comments' ], axis = 1, inplace = True)

df = df.dropna()

#region
df_region = pd.DataFrame(df, columns = ['Region', 'Sales'])
region = df_region.groupby('Region').count()
plt.bar(region.index.values, region['Sales'])

#Product SKU
df_sku = pd.DataFrame(df, columns = ['Product SKU', 'Sales'])
sku = df_sku.groupby('Product SKU').count()
plt.bar(sku.index.values, sku['Sales'])


#category
df_cat= pd.DataFrame(df,columns=['Category', 'Sales'])
cat = df_cat.groupby('Category').count()
plt.bar(cat.index.values, cat['Sales'])

# Discount

df_disc = pd.DataFrame(df,columns=['Discount','Sales'])
disc = df_disc.groupby("Discount").count()
plt.plot(disc.index.values, disc['Sales'])
plt.xlabel('Sales', fontsize=5)
plt.ylabel('group of discount', fontsize=5)
plt.title('Brand wise sales')
plt.show()

#Brand wise sales analysis
df_brand = pd.DataFrame(df,columns=['Brand','Sales'])
brand = df_brand.groupby("Brand").count()
plt.bar(brand.index.values, brand['Sales'])
plt.xlabel('Sales', fontsize=5)
plt.ylabel('Count of brands', fontsize=5)
plt.xticks(brand.index.values, fontsize=8, rotation=45)
plt.title('Brand wise sales')
plt.show()

#state
df_state = pd.DataFrame(df, columns = ['State', 'Sales'])
state = df_state.groupby('State').count()
plt.bar(state.index.values, state['Sales'])
plt.xlabel('Sales', fontsize=5)
plt.ylabel('Count of city wise sales', fontsize=5)
plt.xticks(state.index.values, fontsize=8, rotation=45)
plt.title('wise sales')
plt.show()



