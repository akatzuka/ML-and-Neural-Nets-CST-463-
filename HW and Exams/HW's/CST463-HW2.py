# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:25:34 2018

@author: Bret Stine
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline 
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__ (self, attribute_names):
        self.attribute_names = attribute_names
    def fit (self, X, y = None):
        return self
    def transform (self, X):
        return X[self.attribute_names]

class CreateDummies(TransformerMixin):
    def transform(self, X, **transformparams):
        return pd.get_dummies(X).copy().values
    def fit(self, X, y=None, **fitparams):
        return self

class Preprep(BaseEstimator, TransformerMixin):
    def transform(self, X, **transformparams):
        le = preprocessing.LabelEncoder()
        X['Title'] = X['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
        X['Title'] = le.fit_transform(dat['Title'])
        trans = X.drop(X['Cabin', 'Ticket', 'Name'], axis=1).copy()
        return trans
    def fit(self, X, y=None, **fitparams):
        return self

input_file_titanic = "C:/Users/Bret Stine/Desktop/463 - Machine Learning things/train.csv"
dat = pd.read_csv(input_file_titanic)

plt.style.use('seaborn-whitegrid')

sns.distplot(dat['Age'].dropna())
plt.title('Histogram of Age')

sns.distplot(dat['Fare'])
plt.title('Density of Fare (ticket price)')

sns.jointplot(x='Age', y='Fare', data=dat)

sns.violinplot(x='Age', y='Sex', data=dat)
plt.title('Violin plot of Age and Sex')

sns.barplot(x='Sex', y='Survived', data=dat)
plt.title('Sex of those who Survived')

dat['Title'] = dat['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
dat.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

num_attribs = dat.drop(['Title', 'Sex', 'Embarked'], axis=1)
cat_attribs = ['Title', 'Sex', 'Embarked']

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        ])
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('imputer', Imputer(stratagy="most_frequent")),
        (),
        ])
full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
        ])
dat_prepared = cat_pipeline.fit_transform(dat)
