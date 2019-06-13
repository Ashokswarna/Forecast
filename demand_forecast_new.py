# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:58:34 2019

@author: ashok.swarna
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
os.getcwd()
os.chdir("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\My_python")

df = pd.read_csv("Dataset.Csv")
comments = ((df['positive_comments']/(df['positive_comments'] + df['negative_comments']))*100)
df.insert(loc = 30, column = 'Comments', value = comments)
df = df.assign(Reviews=pd.cut(df.Comments,[0,40,53,100],
                              labels=['Negative','Neutral','Positive']))

df.drop(['Order ID','Customer ID', 'Customer Name',
         'Product Name', 'Product ID', 'Store no.',
         'Ship Date', 'Country', 'Postal Code', 'positive_comments',
        'negative_comments', 'Comments' ], axis = 1, inplace = True)

df.rename(columns={'Order Date':'Order_date'}, inplace=True)
df['Order_date'] = pd.to_datetime(df['Order_date'], errors='coerce')

df.info()
df.isnull().sum().sort_values(ascending=False)
#Missing values are present
#as missing values are 6 and 5 only droping them
df = df.dropna()

#shipment_mode
df_ship = pd.DataFrame(df,columns=['Ship Mode', 'Sales'])
ship_mode = df_ship.groupby('Ship Mode').count()
plt.bar(ship_mode.index.values, ship_mode['Sales'])


#segment
df_seg = pd.DataFrame(df,columns=['Segment', 'Sales'])
seg = df_seg.groupby('Segment').count()
plt.bar(seg.index.values, seg['Sales'])

#city
df_city = pd.DataFrame(df,columns = ['City', 'Sales'])
city = df_city.groupby('City').count()
plt.bar(city.index.values, city['Sales'])
plt.xlabel('Sales', fontsize=5)
plt.ylabel('Count of city wise sales', fontsize=5)
plt.xticks(city.index.values, fontsize=8, rotation=45)
plt.title('wise sales')
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

#sub category wise sales
df_sub = pd.DataFrame(df,columns=['Sub-Category', 'Sales'])
sub_category = df_sub.groupby('Sub-Category').count()
plt.bar(sub_category.index.values, sub_category['Sales'])
plt.xlabel('Sales', fontsize=5)
plt.ylabel('Count of sub quantity', fontsize=5)
plt.xticks(sub_category.index.values, fontsize=8, rotation=45)
plt.title('Brand wise sales')
plt.show()

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

#extract month and year

#Frequency based on Brand and count of observations
df_freq = pd.DataFrame(df,columns=['Brand','Order_date','Sales'])
df_freq.Order_date = df.Order_date.dt.to_period("M")

brand_pivot = df_freq.pivot_table(values='Sales', index='Brand', columns='Order_date', 
                    aggfunc=lambda x: len(x.unique()))

#product SKU wise and count of observations

df_prod = pd.DataFrame(df,columns=['Product SKU','Order_date','Sales'])
df_prod.Order_date = df.Order_date.dt.to_period("M")

prod_pivot = df_prod.pivot_table(values='Sales', index='Product SKU', columns='Order_date', 
                    aggfunc=lambda x: len(x.unique()))

#City wise and count of observations
city_df = pd.DataFrame(df, columns = ['City', 'Order_date', 'Sales'])
city_df.Order_date = df.Order_date.dt.to_period('M')

city_pivot = city_df.pivot_table(values='Sales', index='City', columns='Order_date', 
                                    aggfunc=lambda x: len(x.unique()))

#State wise and count of observations
state_df = pd.DataFrame(df, columns = ['State', 'Order_date', 'Sales'])
state_df.Order_date = df.Order_date.dt.to_period('M')

state_pivot = state_df.pivot_table(values='Sales', index='State', columns='Order_date', 
                                    aggfunc=lambda x: len(x.unique()))

#region wise and count of observations
region_df = pd.DataFrame(df, columns = ['Region', 'Order_date', 'Sales'])
region_df.Order_date = df.Order_date.dt.to_period('M')

region_pivot = region_df.pivot_table(values='Sales', index='Region', columns='Order_date', 
                                    aggfunc=lambda x: len(x.unique()))

#discover patterns for sales , profit

df.Sales.describe()
sales = df['Sales']
print("Median of data-set is : % s "
        % (statistics.median(sales)))

df.Profit.describe()

#correlation 

corr = df.corr()
#check for correlation
np.corrcoef(df.quantity, df.movement_type)

plt.plot(sales)

#Seperating categorical and continuous data
cols = df.columns
num_cols = df._get_numeric_data().columns
cat_cols = df._get_categori_data().columns


##########Pattern 

#Plot for Product SKU level
new_df = pd.DataFrame(df,columns=['Product SKU','Sales'])

new_df.groupby('Product SKU').plot(legend=True)

#Plot for brand level

new_df = pd.DataFrame(df,columns=['Brand','Sales'])

new_df.groupby('Brand').plot(legend=True)

# plot data for state level

df_state.groupby('State').plot(legend=True)
plt.xlabel('Sales', fontsize=5)
plt.ylabel('State', fontsize=5)