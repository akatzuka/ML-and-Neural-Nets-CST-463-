# -*- coding: utf-8 -*-
"""

Gradient descent for single LTU

Created on Wed Oct 24 11:49:07 2018

@author: Glenn
"""

import numpy as np

max_iter = 10000
learning_rate = 0.01
w1 = 0.5
w2 = 2.0
x1 = 1
x2 = 3

# overall function computed by the LTU
def z(x1,x2,w1,w2):
    return g(x1*w1 + x2*w2)

# logistic function
def g(y):
    return 1/(1 + np.exp(-y))

# gradient descent
for i in range(max_iter):
    # YOUR CODE HERE
    w1 = # YOUR CODE HERE
    w2 = # YOUR CODE HERE
    print(z(x1,x2,w1,w2))
    

    


