# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:45:35 2018

@author: Remilia
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree, linear_model
from sklearn.decomposition import PCA

input_file = "https://raw.githubusercontent.com/grbruns/cst495/master/winequality-white.csv"
df = pd.read_csv(input_file, sep=";")

df.info()

y = df['quality'].astype(float).values
df.drop('quality', axis=1, inplace=True)

feature_names = df.columns
X = df.values
X.shape

X_train, X_test, y_train, y_test = train_test_split(X,y)
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
rmse

scaler = StandardScaler()
X = scaler.fit_transform(X) 

pca = PCA(n_components = 0.90)
X_reduced = pca.fit_transform(X)

print(X_reduced.shape[1])

print(pca.components_[0])

