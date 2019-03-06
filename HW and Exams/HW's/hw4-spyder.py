import numpy as np
import numpy.linalg as LA
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# return the gradient of function f at input vector x (x must be iterable)
def gradient(f, x, epsilon=0.01):
   # your code here
   derive = []
   for i in range(len(x)):
       temp = x.copy()
       temp[i] = x[i] + epsilon
       derive.append((f(*temp) - f(*x))/ epsilon)
   derive = np.array(derive)
   return (derive)

# return the vector x such that multi-variate function f(x) is minimized 
# n - length of input vector to f
# alpha - learning rate
# max_iterations and min_change are stopping conditions:
#   max_iterations - return if max_iterations performed
#   min_change - return if change to x is less than min_change
def find_min(f, n, alpha=0.01, max_iterations=10000, min_change=0.0001):
    # your code here
    change = alpha * gradient(f, X_aug)
    if (change < min_change):
        return X_aug
    if (max_iterations == 0):
        return X_aug
    X_aug = X_aug + change
    gradient(f,n, max_iterations-1)

# compute MSE for simple linear regression problem
# X is an augmented feature matrix of two columns, y is an array of numeric labels
def mse_cost(b0, b1, X, y):
    # your code here
    mse = 0
    m = y.shape[0]
    for i in range(m):
        y_pred = b0 + b1 * X[i][1]
        mse += (y[i] - y_pred) ** 2
    mse = mse/m
    return mse
    
dat = load_boston()
    
X = dat['data'][:,5:6]   # average number of rooms (this gives a matrix, not an array)
y = dat['target']        # house price (thousands of dollars)
dat['feature_names']

# remove data where the house price is exactly the max value
# of 50.0; this is a result of censoring.
not_max_rows = (y != 50.0)
y = y[not_max_rows]
X = X[not_max_rows]
n,m = X.shape

X0 = np.ones((n,1))
X_aug = np.hstack((X0, X))

plt.scatter(X, y)
plt.xlabel('Number of rooms')
plt.ylabel('House prices (10K $)')
plt.title('House price by number of rooms');

beta_normal = LA.inv(X_aug.T.dot(X_aug)).dot(X_aug.T.dot(y))
beta_normal = beta_normal.reshape(-1, beta_normal.shape[0])  # -> 1x2 matrix

def f_mse(b0,b1):
    return(mse_cost(b0, b1, X_aug, y))

# compute MSE for a couple of choices for b0, b1
print(np.round(f_mse(-30, 8),2), np.round(f_mse(-10, 6),2))
# NEEDS TO EQUAL 35.46 AND 68.89


B0 = np.arange(-100, 100, 10)
B1 = np.arange(-10, 10, 2)
B0, B1 = np.meshgrid(B0, B1)

# see stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
zs = np.array([f_mse(b0,b1) for b0,b1 in zip(np.ravel(B0), np.ravel(B1))])
Z = zs.reshape(B0.shape)
    
cmap = cm.get_cmap('bwr')   # red value is high, dark blue is low
plt.contourf(B0, B1, Z, 30, cmap=cmap);      # filled contour map

beta_gd = find_min(f_mse, 2, alpha=0.01, max_iterations=10000, min_change=0.0001)
beta_gd = beta_gd.reshape(-1, beta_gd.shape[0])  # -> 1x2 matrix






len(y)
