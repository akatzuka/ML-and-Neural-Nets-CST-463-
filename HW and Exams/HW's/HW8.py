# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals


# Common imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
#from tensorflow_graph_in_jupyter import show_graph

######################################

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

input_file = "https://raw.githubusercontent.com/akatzuka/CST-463-Project-1/master/default_cc_train.csv"
df = pd.read_csv(input_file)

X_old = df.iloc[:,0:24]
#y = np.asarray(df["default.payment.next.month"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,
y, test_size=0.25)

n_inputs = 24  # Credit
n_hidden1 = 300
n_hidden2 = 200
n_hidden3 = 100
n_hidden4 = 50
n_outputs = 20

#reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.elu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.elu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.elu)
    hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4", activation=tf.nn.elu)
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)
    
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    #base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
    #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #loss = tf.add_n([base_loss] + reg_losses, name="loss")
    
learning_rate = 0.01

with tf.name_scope("train"):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9, use_nesterov=True)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
n_batches = 50
batch_size = n_batches

batch_total = []
valid_total = []

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_test, y: y_test})
        batch_total.append(acc_batch)
        valid_total.append(acc_valid)
        print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    print("Average batch accuracy:", (sum(batch_total) / n_epochs), "Average validation accuracy:", (sum(valid_total) / n_epochs))
    save_path = saver.save(sess, "./my_model_final.ckpt")
    
#show_graph(tf.get_default_graph())