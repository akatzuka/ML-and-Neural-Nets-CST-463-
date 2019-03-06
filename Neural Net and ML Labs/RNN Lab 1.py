# 
# A manually-built RNN
#
   
# graph construction

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])
X2 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

# key to getting this is that we have unrolled the RNN over two time steps
Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)
Y2 = tf.tanh(tf.matmul(Y1, Wy) + tf.matmul(X2, Wx) + b)

init = tf.global_variables_initializer()

# execution 

X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1
X2_batch = np.array([[6, 5, 4], [3, 2, 1], [0, 0, 0], [9, 8, 7]]) # t = 2


with tf.Session() as sess:
   init.run()
   Y0_val, Y1_val, Y2_val = sess.run([Y0, Y1, Y2], feed_dict={X0: X0_batch, X1: X1_batch, X2: X2_batch})
   
print(Y0_val)
print(Y1_val)
print(Y2_val)