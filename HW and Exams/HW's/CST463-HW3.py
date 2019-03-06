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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier

# Selector to get the attributes for the given columns. Either numeric or categorical.
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__ (self, attribute_names):
        self.attribute_names = attribute_names
    def fit (self, X, y = None):
        return self
    def transform (self, X):
        return X[self.attribute_names]

# Custom get_dummies() class for categorical columns using a pipeline.
class CreateDummies(TransformerMixin):
    def __init__(self):
        """
        """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.get_dummies(X)

# Custom transformer to strip the names column of titles and replace.
class Preprepp(TransformerMixin):
    def __init__(self):
        """
        """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['Name'] = X['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
        X['Name'] = X.apply(replace_titles, axis=1)
        return X

# Custom Imputer class to fill nans for categorical columns.
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

# Simple transformer to convert pandas dataframe to numpy matrix.
class DataFrameToMatrix(TransformerMixin):
    def __init__(self):
        """
        """
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.values

# Function to clean up the new 'Name' column.
# Code credit found at: https://www.kaggle.com/manuelatadvice/feature-engineering-titles
def replace_titles(x):
    title = x['Name']
    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:
        return 'Mr'
    elif title in ['the Countess', 'Mme', 'Lady']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
input_file_train = "C:/Users/Bret Stine/Desktop/463 - Machine Learning things/train.csv"
dat = pd.read_csv(input_file_train)

X = dat[['Pclass', 'Age', 'SibSp', 'Fare', 'Name', 'Sex', 'Embarked']]
y = dat['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

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

num_attribs = ['Pclass', 'Age', 'SibSp', 'Fare']
cat_attribs = ['Name', 'Sex', 'Embarked']

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('to_matrix', DataFrameToMatrix()),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        ])
cat_pipeline = Pipeline([
        ('preprep', Preprepp()),
        ('selector', DataFrameSelector(cat_attribs)),
        ('imputer', DataFrameImputer()),        
        ('dummies', CreateDummies()),
        ('scaling', StandardScaler()),
        ])
full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
        ])

X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.fit_transform(X_test)
#dat_testing = full_pipeline.transform(datTest)

# 1st Model using Logistic Regression
log_regr = LogisticRegression()
log_regr.fit(X_train, y_train)
log_predict = log_regr.predict(X_test)
lr_scores = log_regr.score(X_test, y_test)
# Confusion Matrix to identify False Positive Rate and True Positive Rate
confusion_matrix_log = confusion_matrix(y_test, log_predict)
print(confusion_matrix_log)

# Beginning of the ROC plot using roc_auc_score and roc_curve. Much easier than R.
logit_roc_auc = roc_auc_score(y_test, log_predict)
# Getting the False Posistive Rate, True Posistive Rate, and Thresholds
fpr, tpr, thresholds = roc_curve(y_test, log_regr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
# Limit the x axis with 0 to 1
plt.xlim([0.0, 1.0])
# Limit the y axis with 0 to 1.05. 1.05 just in case the model is perfect.
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# 2nd Model using RandomForestRegressor
forest_reg = RandomForestClassifier(random_state=42, n_estimators=100, max_features=6, oob_score=True)
forest_reg.fit(X_train, y_train)
forest_predict = forest_reg.predict(X_test)
forest_scores = forest_reg.score(X_test, y_test)

forest_reg.get_params,forest_reg.feature_importances_

# Beginning of the ROC plot using roc_auc_score and roc_curve. Much easier than R.
forest_roc_auc = roc_auc_score(y_test, forest_predict)
# Getting the False Posistive Rate, True Posistive Rate, and Thresholds
fpr, tpr, thresholds = roc_curve(y_test, forest_reg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
# Limit the x axis with 0 to 1
plt.xlim([0.0, 1.0])
# Limit the y axis with 0 to 1.05. 1.05 just in case the model is perfect.
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Forest_ROC')
plt.show()