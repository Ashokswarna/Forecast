# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:45:08 2019

@author: ashok.swarna
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
os.chdir("C:\\Users\\ashok.swarna\\Documents")
df_train = pd.read_csv("train.Csv")
df_test = pd.read_csv("test.Csv")
#new_df = pd.DataFrame()
#new_df = new_df.append(df)
#new_df.dtypes
#new_df.drop(['Dates', 'Descript', 'Address'], axis = 1, inplace = True)

#merge the train and test datasets function
def merge_data(a,b):
#get the columns names in test and training
    train_cols = list(df_train.columns.values)
    test_cols = list(df_test.columns.values)
#add all these columns into one list
    all_cols = set(train_cols + test_cols)
#get the columns missing in the test and train set and add them
    add_to_train = list(all_cols - set(train_cols))
    add_to_test = list(all_cols - set(test_cols))
#added the missing columns and replaced them with NAN    
    for col in add_to_train:
        df_train[col] = np.nan
    for col in add_to_test:
        df_test[col] = np.nan
    df_test.drop(['Id'], axis = 1, inplace = True)
    df_train.drop(['Id'], axis = 1, inplace = True)
    print(df_test.shape)
    print(df_train.shape)
    df_merged = pd.concat([df_train, df_test], axis = 0)
    print(df_merged.shape)
    return df_merged
    
#https://www.kaggle.com/nikolayburlutskiy/xgboost-sf-crime
#https://github.com/yinniyu/kaggle_SFCrime/blob/master/SF%20crime%20(2).ipynb
#https://www.kaggle.com/keldibek/xgboost-crime-classification
def time_extract(a):
    df_merged['Dates'] = pd.to_datetime(df_merged['Dates'])
    df_merged['Year'] = df_merged['Dates'].dt.year
    df_merged['Month'] = df_merged['Dates'].dt.month
    df_merged['Week'] = df_merged['Dates'].dt.week
    df_merged['Day'] = df_merged['Dates'].dt.day
    df_merged['Hour'] = df_merged['Dates'].dt.hour
    return df_merged

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
       df_merged = merge_data(df_train, df_test)
       time_extract(df_merged)    
       df_merged.drop(['Dates', 'Address', 'Descript'], axis = 1, inplace = True)
       
       #convert features from textual to numeric
       features = ['Resolution', 'DayOfWeek', 'PdDistrict', 'Category' ]
       for feature in features:
           print (feature)
           df_merged = featureToInteger(df_merged, feature)
       classes, df_merged = makeClass(df_merged)
       features_drop = ['DayOfWeek', 'PdDistrict', 'Resolution']
       features_left = list(set(df_merged.columns)-set(features_drop))
       new_df = df_merged[features_left]
       dtypes = new_df.dtypes
       
       #split the data
       new_df.drop(['Category'], axis = 1, inplace = True)
       y = new_df.Category_1
       y = y.astype(int)
       new_df['Category_1'].unique()
       x = new_df.drop('Category_1', axis = 1)      
       X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
       
       #fit the model
       train_xg = xgb.DMatrix(X_train, label=y_train)
       test_xg = xgb.DMatrix(X_test, label=y_test)
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
       bst = xgb.train(param, train_xg, num_round)
       print ('model built')
  # get prediction
       pred = bst.predict(test_xg)
       print ('prediction done')
  #format the prediction
       df_pred = pd.DataFrame(pred)
       df_pred.columns = classes
       df_pred = df_pred.reindex_axis(sorted(df_pred.columns), axis=1)
       df_pred = pd.concat([data['x_test']['Id'], df_pred],axis=1)
       df_pred['Id'] = df_pred['Id'].astype(int)
       
       
       #model = xgb.XGBClassifier(missing=np.nan, max_depth=6, 
       #                 n_estimators=5, learning_rate=0.15, 
        #                subsample=1, colsample_bytree=0.9, seed=1400)
       #model.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="logloss")
#clf.fit(X_fit, y_fit, early_stopping_rounds=50, eval_metric="logloss", eval_set=[(X_eval, y_eval)])
