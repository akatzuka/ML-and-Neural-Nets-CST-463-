import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# get MNIST data
mnist = input_data.read_data_sets("/tmp/data")

tf.reset_default_graph()

# constants
n_inputs  = 28*28    # number of pixels in an MNIST image
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10     # 10 classes

# create placeholders
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# create network
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits =  tf.layers.dense(hidden2, n_outputs, name="outputs")
    
# cost function (cross entropy)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
# select optimizer
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
# specify how to evaluate model performance
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
# create a node to initialize all variables
init = tf.global_variables_initializer()

n_epochs = 5   # set very low
batch_size = 50

# train the model
with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_test  = accuracy.eval(feed_dict={X: mnist.test.images,
                                             y: mnist.test.labels})
        print(epoch, "Test accuracy:", acc_test)
        
#######################
        
tf.reset_default_graph()

# constants
n_inputs  = 28*28    # number of pixels in an MNIST image
n_hidden1 = 200
n_hidden2 = 100
n_hidden3 = 50
n_outputs = 10     # 10 classes

# create placeholders
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# create network
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.relu)
    logits =  tf.layers.dense(hidden2, n_outputs, name="outputs")
    
# cost function (cross entropy)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
# select optimizer
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
# specify how to evaluate model performance
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
# create a node to initialize all variables
init = tf.global_variables_initializer()

n_epochs = 5   # set very low
batch_size = 50

# train the model
with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_test  = accuracy.eval(feed_dict={X: mnist.test.images,
                                             y: mnist.test.labels})
        print(epoch, "Test accuracy:", acc_test)
        
#########################################
        
tf.reset_default_graph()

# constants
n_inputs  = 28*28    # number of pixels in an MNIST image
n_hidden1 = 200
n_hidden2 = 100
n_hidden3 = 50
n_outputs = 10     # 10 classes

# create placeholders
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# create network
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.relu)
    logits =  tf.layers.dense(hidden2, n_outputs, name="outputs")
    
# cost function (cross entropy)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
# select optimizer
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)
    
# specify how to evaluate model performance
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
# create a node to initialize all variables
init = tf.global_variables_initializer()

n_epochs = 5   # set very low
batch_size = 50

# train the model
with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_test  = accuracy.eval(feed_dict={X: mnist.test.images,
                                             y: mnist.test.labels})
        print(epoch, "Test accuracy:", acc_test)
        
#########################################
        
tf.reset_default_graph()

# constants
n_inputs  = 28*28    # number of pixels in an MNIST image
n_hidden1 = 200
n_hidden2 = 100
n_hidden3 = 50
n_outputs = 10     # 10 classes

# create placeholders
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# create network
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.elu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.elu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.elu)
    logits =  tf.layers.dense(hidden2, n_outputs, name="outputs")
    
# cost function (cross entropy)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
# select optimizer
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)
    
# specify how to evaluate model performance
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
# create a node to initialize all variables
init = tf.global_variables_initializer()

n_epochs = 5   # set very low
batch_size = 50

# train the model
with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_test  = accuracy.eval(feed_dict={X: mnist.test.images,
                                             y: mnist.test.labels})
        print(epoch, "Test accuracy:", acc_test)