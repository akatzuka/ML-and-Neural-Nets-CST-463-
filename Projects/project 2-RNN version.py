import numpy as np
import tensorflow as tf
import requests

def next_batch(batch_size, inputs, targets):
    rnd_idx = np.arange(0 , len(inputs))
    np.random.shuffle(rnd_idx)
    rnd_idx = rnd_idx[:batch_size]
    inputs_shuffle = one_hot([inputs[i] for i in rnd_idx])
    targets_shuffle = one_hot([targets[i] for i in rnd_idx])
    return np.asarray(inputs_shuffle), np.asarray(targets_shuffle)

def one_hot(X):
    return (np.eye(vocab_size)[X])

tf.reset_default_graph()

# data I/O
# should be simple plain text file
#data = open('D:/CST 463/alice.txt', 'r').read()
data = requests.get("https://raw.githubusercontent.com/bretstine/alice/master/alice.txt")
data = data.text
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 0.001

# create training sequences and corresponding labels
Xi = []
yi = []
for i in range(0, len(data)-seq_length-1, 1):
        Xi.append([char_to_ix[ch] for ch in data[i:i+seq_length]])
        yi.append([char_to_ix[ch] for ch in data[i+1:i+seq_length+1]])
# reshape the data
        
# in X_modified, each row is an encoded sequence of characters
X_modified = np.reshape(Xi, (len(Xi), seq_length))
y_modified = np.reshape(yi, (len(yi), seq_length))

inputs = tf.placeholder(tf.float32, [None, seq_length, vocab_size], name='inputs')
targets = tf.placeholder(tf.float32, [None, seq_length, vocab_size], name='targets')

with tf.name_scope("rnn"):
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=vocab_size, activation=tf.nn.relu)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, inputs, dtype=tf.float32)

with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.cast(targets, tf.int32), logits=outputs)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
init = tf.global_variables_initializer()

n_iterations = 20000
batch_size = vocab_size
smooth_loss = -np.log(1.0/vocab_size) * seq_length
p = 0

with tf.Session() as sess:
     init.run()
     for iteration in range(n_iterations):
         X_batch, y_batch = next_batch(batch_size, X_modified, y_modified)
         sess.run(training_op, feed_dict={inputs: X_batch, targets: y_batch})
         if iteration % 100 == 0:
             tr = ""
             test = X_modified[iteration]
             predictions = outputs.eval(feed_dict={inputs: X_batch})
             ttest = []
             for j in predictions:
                 p = np.exp(j) / np.sum(np.exp(j))
                 ix = np.random.choice(range(vocab_size), p = np.sum(p, axis = 0))
                 x = np.zeros((vocab_size , 1))
                 x[ix] = 1
                 tr = tr + ix_to_char[ix]
                 ttest.append(ix)
             test = np.array(ttest)
             print ('----\n %s \n----' % (tr, ))
             mse = loss.eval(feed_dict={inputs: X_batch, targets: y_batch})
             smooth_loss = smooth_loss * 0.999 + mse * 0.001
             print ('iter %d, \tloss: %f, \tsmooth_loss: %f' % (iteration, mse, smooth_loss))
