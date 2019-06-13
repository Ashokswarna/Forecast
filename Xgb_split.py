# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:58:02 2019

@author: ashok.swarna
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 

os.chdir("C:\\Users\\ashok.swarna\\Documents")
df = pd.read_csv('train.Csv')

df['Dates'] = pd.to_datetime(df['Dates'])
Hour = df['Dates'].dt.hour

#drop Description, Address resolution and Dates

df.drop(['Dates', 'Descript', 'Address', 'Resolution'], axis = 1, inplace = True)

c1_range = list(range (0,7,1))
c2_range = list(range (7,13,1))
c3_range = list(range (13,19,1))
c4_range = list(range (19,25,1))

#split time into 4 categories
for i in Hour:
    if zip ((i >= 0) & (i <=6)):
        df['Day_range'] = 'C1'
    elif zip ((i >=7) & (i <=12)):
        df['Day_range'] = 'C2'
    elif zip ((i >= 13) & (i <=18)):
        df['Day_range'] = 'C3'
    elif zip ((i >= 19) & (i <=24)):
        df['Day_range'] = 'C4'

X = df.iloc[: , 1:6 ].values
Y = df.iloc[:,0].values

labelencoder_X_0 = LabelEncoder()
X[:,0] = labelencoder_X_0.fit_transform(X[:,0])

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

onehotencoder = OneHotEncoder(categorical_features = [0,1])
X = onehotencoder.fit_transform(X).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size=0.2, random_state=42)

#ETA learning rate

#XGBClassifier(max_depth=3, learning_rate=0.1, 
#n_estimators=100, silent=True, 
#objective='binary:logistic', 
#booster='gbtree', n_jobs=1, 
#nthread=None, gamma=0, 
#min_child_weight=1, 
#max_delta_step=0, subsample=1, 
#colsample_bytree=1, colsample_bylevel=1, 
#reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 
#base_score=0.5, random_state=0, seed=None, 
#missing=None, **kwargs) #

#classifier = XGBClassifier(eta = 0.4, max_depth = 5)

classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

Predicted = pd.DataFrame(y_pred)

Real = pd.DataFrame(Y_test)

cm = confusion_matrix(Y_test, y_pred)

true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos
false_neg = np.sum(cm, axis=1) - true_pos

precision = np.sum(true_pos / true_pos+false_pos)
recall = np.sum(true_pos / true_pos + false_neg)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred)) 

#decision tree prediction
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, Y_train)
y_pred1 = dt_classifier.predict(X_test)
Predicted1 = pd.DataFrame(y_pred1)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))


