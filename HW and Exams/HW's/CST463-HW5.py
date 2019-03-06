import numpy as np
import numpy.linalg as LA
from sklearn.datasets import load_boston, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def logistic(x):
    return 1/(1 + np.exp(-x))
    
# compute log loss for logistic regression
# theta is an array of coefficients, X is an augmented 
# feature matrix, y is an array of numeric labels
def logistic_gradient(theta, X, y):
    # YOUR CODE HERE
    loss = 0
    for i in range(len(X)):
        loss += ((logistic(theta.dot(X[i])) - y[i]) * X[i]) 
    return loss/len(X)
    
# test logistic_gradient
m = 3
theta = np.array([0.5,1.0])
X = np.array([[0.27], [0.66], [0.87]])
X0 = np.ones((m,1))
X_aug = np.hstack((X0, X))
y = np.array([0,0,1])

# result should be about [0.41, 0.17]
print(logistic_gradient(theta, X_aug, y))

# use gradient descent to find a vector that is the approximate
# minimum of a function whose gradients can be computed with
# function grads
# grads - computes gradients of a function
# n - length of vector expected by grads as input
# alpha - learning rate
# max_iterations and min_change are stopping conditions:
#   max_iterations - return if max_iterations performed
#   min_change - return if change to x is less than min_change
def grad_descent(grads, n, alpha=0.01, max_iterations=10000, min_change=0.0001):
    x = np.zeros(n)        # this is just one way to initialize
    num_iterations = 0
    while num_iterations <= max_iterations:
        x_last = x
        x = x - grads(x)*alpha    # update x
        resid = x - x_last
        change = np.sqrt(resid.dot(resid))
        if change < min_change:
            print("stopped on min change")
            return(x)
        num_iterations += 1
    print("stopped on max iterations")
    return(x)
    
dat = load_boston()
  
X = dat['data'][:,5:6]   # avg. number of rooms
y = dat['target']        # house price (thousands of dollars)
dat['feature_names']

# remove data where the house price is exactly the max value
# of 50.0; this is a result of censoring.
not_max_rows = (y != 50.0)
y = y[not_max_rows]
X = X[not_max_rows]
n,m = X.shape

# convert target to 0/1, with 1 for house price > 25
y = np.where(y > 25.0, 1.0, 0.0)

X0 = np.ones((n,1))
X_aug = np.hstack((X0, X))

plt.scatter(X, y)
plt.xlabel('Number of rooms')
plt.ylabel('House price > $25K')
plt.title('House price by number of rooms');

# logistic regression loss function
# X is an augmented feature matrix of two columns, y is an array of numeric labels
def log_loss(theta, X, y):
    # YOUR CODE HERE
    loss = 0
    for i in range(len(X)):
        loss += (y[i] * np.log(logistic(theta.dot(X[i]))) + (1 - y[i]) * np.log(1 - logistic(theta.dot(X[i]))))
    return loss/-len(X)

# a version of the loss function where the training data is hidden
def f_loss(theta):
    return log_loss(theta, X_aug, y)

theta0 = np.linspace(-15, 5, 20)
theta1 = np.linspace(-1.5, 2.5, 20)
theta0, theta1 = np.meshgrid(theta0, theta1)

# see stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
zs = np.array([f_loss(np.array([t0, t1])) for t0,t1 in zip(np.ravel(theta0), np.ravel(theta1))])
Z = zs.reshape(theta0.shape)
    
cmap = cm.get_cmap('bwr')   # red value is high, dark blue is low
plt.contourf(theta0, theta1, Z, 30, cmap=cmap);      # filled contour map

# create version of log loss function with single vector input
def f_grads(theta):
    return logistic_gradient(theta, X_aug, y)

gd_coefs = grad_descent(f_grads, 2, alpha=0.01, max_iterations=30000, min_change=0.0001)
print(gd_coefs)
print(f_loss(gd_coefs))

lr_clf = LogisticRegression()
lr_clf.fit(X, y)
lr_clf.score(X, y)
sk_coefs = np.array([lr_clf.intercept_[0], lr_clf.coef_[0,0]])
print(sk_coefs)
print(f_loss(sk_coefs))

# compute accuracy using coefficients
def accuracy(theta):
    y_pred = logistic(X_aug.dot(theta))
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return accuracy_score(y, y_pred)

print(accuracy(gd_coefs))
print(accuracy(sk_coefs))

X_new = np.linspace(3, 9, 490).reshape(-1,1)
plt.figure(1, figsize=(6, 4))
plt.scatter(X, y, color='steelblue', zorder=20) 

plt.plot(X_new, logistic(sk_coefs[1] * X_new + sk_coefs[0]), color="green")
plt.plot(X_new, logistic(gd_coefs[1] * X_new + gd_coefs[0]), color="orange")
plt.title("Housing price by number of rooms")
plt.xlabel("Number of rooms")
plt.ylabel("Housing price > $25k")
plt.xlim(2.5, 9.5)
plt.legend(('sklearn', 'gradient descent'), loc="center left", fontsize='small')
