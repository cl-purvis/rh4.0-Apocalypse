# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:16:53 2018

@author: Yash Bhati
"""

import numpy as np
import pandas as pd
labelencoder_X.fit_transform
dt=pd.read_csv('book2.csv')

X = dt.iloc[:, :].values
y = dt.iloc[:, 4].values
val = dt['Production'].mean()

dt.fillna(value = val, axis = 1, inplace = True)
dt.drop(['State_Name', 'District_Name'], axis = 1, inplace = True)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_1 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])
le_2 = LabelEncoder()
X[:, 2] = le_2.fit_transform(X[:, 2])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
y = y.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)

y_pred = linreg.predict(X)
   
from sklearn.metrics import r2_score
score = r2_score(y, y_pred)
