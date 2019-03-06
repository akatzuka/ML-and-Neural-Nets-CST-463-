import numpy as np
import tensorflow as tf

# replace the file path here with your own
#data = (open("C:/Users/Glenn/Google Drive/CSUMB/Spring18/DS/homework/week6/alice.txt").read())
data = open('D:/CST 463/alice.txt', 'r')

# create mapping from characters to numbers and back
#chars = list(set(data))
#data_size, vocab_size = len(data), len(chars)
#char_to_ix = { ch:i for i,ch in enumerate(chars) }   # dict comprehension
#ix_to_char = { i:ch for i,ch in enumerate(chars) }

words = []
for word in data.read().split():
    words.append(word)
data_size, vocab_size = len(data), words.size()
char_to_ix = { ch:i for i,ch in enumerate(words) }   # dict comprehension
ix_to_char = { i:ch for i,ch in enumerate(words) }

# create training sequences, and corresponding labels
X = []
y = []
seq_length = 50
for i in range(0, len(data)-seq_length-1, 1):
    X.append([char_to_ix[ch] for ch in data[i:i+seq_length]])
    y.append([char_to_ix[ch] for ch in data[i+1:i+seq_length+1]])

# reshape the data; in X_modified, each row is an encoded 
# sequence of characters
X_modified = np.reshape(X, (len(X), seq_length))
y_modified = np.reshape(y, (len(y), seq_length))

#
# graph construction
#

n_steps = seq_length
n_inputs = vocab_size
n_outputs = vocab_size
n_neurons = 100

learning_rate = 0.001

tf.reset_default_graph()

X = tf.placeholder(tf.int32, [None, n_steps])
X_hot = tf.one_hot(X, n_inputs)
y = tf.placeholder(tf.int32, [None, n_steps])

# alternative: GPU-enabled GRU cell
#        tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=n_neurons),

basic_cell = tf.contrib.rnn.OutputProjectionWrapper(
              tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons),
              output_size=n_outputs)
logits, states = tf.nn.dynamic_rnn(basic_cell, X_hot, dtype=tf.float32)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# Get the last character of the first batch, then convert it
# to a probability vector.
last_char_matrix = tf.slice(logits, [0,n_steps-1,0], [1,1,vocab_size])
last_char_vector = tf.reshape(last_char_matrix, [vocab_size])
probs = tf.nn.softmax(last_char_vector)

init = tf.global_variables_initializer()

#
# graph execution
#

n_epochs = 10
batch_size = 50     # sequences per batch
n_batches = 50     # batches per epoch
sample_size = 200   # length of generated text at end of each epoch

def fetch_batch(epoch, batch_index, batch_size, X_dat, y_dat):
    m = X_dat.shape[0]
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = X_dat[indices] 
    y_batch = y_dat[indices]
    return X_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, iteration, batch_size, X_modified, y_modified)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval(feed_dict={X: X_batch, y: y_batch})
        print(epoch, "loss:", loss_val)
        
        # every so often, generate some sample output
        ixes = X_batch[[0]].tolist()[0]
        for i in range(sample_size):
            sample_batch = np.array(ixes[-n_steps:]).reshape(1, n_steps)
            p = probs.eval(feed_dict={X: sample_batch})
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            ixes.append(ix)
        txt = ''.join(ix_to_char[ix] for ix in ixes[-sample_size:])
        print('----\n %s \n----' % (txt, ))