import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#
# read the red wine data
#

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst495/master/winequality-red.csv", sep=";")

# any NaN values?

df.isnull().sum()

X = df.iloc[:,0:11]
y = df['quality'].values

#
# use PCA and plot the cumulative explained variance ratio
#
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum > 0.95) + 1 
plt.plot(cumsum)
#
# transform X to a new data set using the top three principal components
#
pca = PCA(n_components=3)
X3D = pca.fit_transform(X)
plt.plot(X3D)
#
# try linear regression on the original data
#
# split the data into training and test sets, train
# a linear model, make predictions, and compute RMSE

X_train, X_test, y_train, y_test = train_test_split(X,y)

regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)

#
# try linear regression on the transformed data created
# with the first three principal components.  Compare
# your RMSE to the previous case.
#
# YOUR CODE HERE