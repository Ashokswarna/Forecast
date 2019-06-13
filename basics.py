# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:39:29 2019

@author: ashok.swarna
"""
import pandas as pd
import numpy as np

df2 = pd.DataFrame([1, '', ''], ['a', 'b', 'c'])
df2.replace("" , np.nan, inplace = True)

df2.replace(np.nan, 0 , inplace = True)
df2.isnull().sum()

df.groupby(['gender']).agg({'sales':'sum'})

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 
# Cannot use Rank 1 matrix in scikit learn
X = X.reshape((m, 1))
# Creating Model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)
 
# Calculating R2 Score
r2_score = reg.score(X, Y)
 
print(r2_score)


