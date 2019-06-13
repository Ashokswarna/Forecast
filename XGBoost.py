# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:37:09 2019

@author: ashok.swarna
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from xgboost import plot_importance


os.chdir("C:\\Users\\ashok.swarna\\Documents")
df = pd.read_csv('train.Csv')

df['Dates'] = pd.to_datetime(df['Dates'])
df['Hour'] = df['Dates'].dt.hour

df = df.assign(session=pd.cut(df.Hour,[-1,5,11,17,21,24],
                              labels=['Mid-Night','Morning','Afternoon','Evening', 'Night']))

#drop Description, Address resolution and Dates

df.drop(['Dates', 'Descript', 'Address', 'Resolution', 'Hour'], 
        axis = 1, inplace = True)

X = df.iloc[: , 1:6 ].values
Y = df.iloc[:,0].values

labelencoder_X_0 = LabelEncoder()
X[:,0] = labelencoder_X_0.fit_transform(X[:,0])

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_4 = LabelEncoder()
X[:,4] = labelencoder_X_4.fit_transform(X[:,4])

onehotencoder = OneHotEncoder(categorical_features = [0,1,4])
X = onehotencoder.fit_transform(X).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size=0.2, random_state=42)

#ETA learning rate

#classifier = XGBClassifier(eta = 0.4, max_depth = 6,
#                           colsample_bytree=0.9, seed=1400)
#Accuracy 27.07
classifier = XGBClassifier(eta = 0.5, max_depth = 7,
                           colsample_bytree=0.9, seed=1400)
#classifier = XGBClassifier()
classifier.fit(X_train, Y_train, eval_metric="auc")

#27.42

y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)

true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos
false_neg = np.sum(cm, axis=1) - true_pos

precision = np.sum(true_pos / true_pos+false_pos)
recall = np.sum(true_pos / true_pos + false_neg)

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred)) 

plot_importance(classifier, )

df.to_csv('SF_crime1.Csv')
