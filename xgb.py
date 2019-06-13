# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:44:27 2019

@author: ashok.swarna
"""

import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
os.chdir("C:\\Users\\ashok.swarna\\Documents")
df_train = pd.read_csv("train.Csv")
df_test = pd.read_csv("test.Csv")

def time_extract(a):
    df_train['Dates'] = pd.to_datetime(df_train['Dates'])
    df_train['Year'] = df_train['Dates'].dt.year
    df_train['Month'] = df_train['Dates'].dt.month
    df_train['Week'] = df_train['Dates'].dt.week
    df_train['Day'] = df_train['Dates'].dt.day
    df_train['Hour'] = df_train['Dates'].dt.hour
    return df_train

def time_extract_test(a):
    df_test['Dates'] = pd.to_datetime(df_test['Dates'])
    df_test['Year'] = df_test['Dates'].dt.year
    df_test['Month'] = df_test['Dates'].dt.month
    df_test['Week'] = df_test['Dates'].dt.week
    df_test['Day'] = df_test['Dates'].dt.day
    df_test['Hour'] = df_test['Dates'].dt.hour
    return df_train

def featureToInteger(data, feature):
   notnan = data[feature].dropna(how='all')
   unique = notnan.unique()
   data[feature+'_1'] = np.nan
   for cls in unique:
      cls_ind = np.where(unique==cls)[0][0]
      data[feature+'_1'][data[feature]==cls] = cls_ind
   return data

def makeClass(data):
   categories = data['Category'].dropna(how='all')
   classes = categories.unique()
   data['class'] = np.nan
   for cls in classes:
      cls_ind = np.where(classes==cls)[0][0]
      data['class'][data['Category']==cls] = cls_ind
   df_classes = pd.DataFrame(classes)
   return classes, data

if __name__ == '__main__':
    
    time_extract(df_train)
    time_extract_test(df_test)
    df_train.rename(columns={'X':'Latitude', 'Y':'Longitude'})
    df_test.rename(columns={'X':'Latitude', 'Y':'Longitude'})
    df_train.drop(['Dates', 'Address', 'Descript', 'Resolution'], axis = 1, inplace = True)
    df_test.drop(['Dates', 'Address'], axis = 1, inplace = True)
    features = ['DayOfWeek', 'PdDistrict', 'Category' ]
    for feature in features:
        print (feature)
        df_train = featureToInteger(df_train, feature)
    classes, df_train = makeClass(df_train)
    features1 = ['DayOfWeek', 'PdDistrict' ]
    for feature in features1:
        print (feature)
        df_test = featureToInteger(df_test, feature)
    #df_test = makeClass(df_test)
    features_drop = ['DayOfWeek', 'PdDistrict', 'Resolution']
    features_dr = ['DayOfWeek', 'PdDistrict']
    features_left = list(set(df_train.columns)-set(features_drop))
    features_le = list(set(df_test.columns)-set(features_dr))
    features_left_test = list(set(df_test.columns)-set(features_dr))
    new_df = df_train[features_left]
    new_df_test = df_test[features_le]
    dtypes = new_df.dtypes
       
       #split the data
    train_data, validate_data = train_test_split(new_df, test_size=0.2, random_state=42)

    train_X = train_data.drop('Category_1', 1)
    train_Y = train_data.Category_1
    validate_X = validate_data.drop('Category_1', 1)
    validate_Y = validate_data.Category_1
    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dtest = xgb.DMatrix(validate_X, label=validate_Y)
    print ('matrices created')
  # setup parameters for xgboost
    param = {}
  # use softprob multi-class classification
    param['objective'] = 'multi:softprob'
    param['eta'] = 1
  #param['eta'] = 0.8
    param['max_depth'] = 8
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = len(classes)
    param['max_delta_step'] = 1
    num_round = 10
    watchlist = [ (dtrain,'train'), (dtest, 'eval') ]
    bst = xgb.train(param, dtrain, num_round, watchlist);
    print ('model built')
    yprob = bst.predict(dtest).reshape( validate_Y.shape[0], param['num_class'])
    ylabel = np.argmax(yprob, axis=1)
    print ('prediction done')
    
    
    df_pred = pd.DataFrame(yprob)
    df_pred.describe()
    df_pred.columns = classes
    df_pred = df_pred.reindex_axis(sorted(df_pred.columns), axis=1)
    df_pred.to_csv('submission.csv', index=False,  float_format='%.6f') 