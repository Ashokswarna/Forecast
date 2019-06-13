# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:40:04 2019

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
df = pd.read_csv('SF_crime2.Csv')

#df.drop(['X', 'Y'],axis = 1, inplace = True)

X = df.iloc[: , 1:6 ].values
Y = df.iloc[:,0].values

labelencoder_X_0 = LabelEncoder()
X[:,0] = labelencoder_X_0.fit_transform(X[:,0])

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])


onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
X = onehotencoder.fit_transform(X).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size=0.2, random_state=42)

#Fit the model

classifier = XGBClassifier(eta = 0.5, max_depth = 7,
                           colsample_bytree=0.9, seed=1400)
#classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred)) 

plot_importance(classifier, )
#0.26025682629456215 with 10000 condition

