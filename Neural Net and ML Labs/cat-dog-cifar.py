import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

height = 32
width = 32
channels = 3
n_inputs = height * width * channels

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10

tf.reset_default_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    p3_flip = tf.image.flip_left_right(conv2)
    p3_rot = tf.image.rot90(p3_flip,k=1)
    pool3 = tf.nn.max_pool(p3_flip, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 8 * 8])

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()




    
# load data
cifar_file1 = "D:\CST 463\cifar-10-python.tar\cifar-10-batches-py\data_batch_1"
cifar_file2 = "D:\CST 463\cifar-10-python.tar\cifar-10-batches-py\data_batch_2"
cifar_file3 = "D:\CST 463\cifar-10-python.tar\cifar-10-batches-py\data_batch_3"
cifar_file4 = "D:\CST 463\cifar-10-python.tar\cifar-10-batches-py\data_batch_4"
cifar_file5 = "D:\CST 463\cifar-10-python.tar\cifar-10-batches-py\data_batch_5"

#class 3 = cats, class 5 = dogs

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict     

cf1 = unpickle(cifar_file1)
cf2 = unpickle(cifar_file2)
cf3 = unpickle(cifar_file3)
cf4 = unpickle(cifar_file4)
cf5 = unpickle(cifar_file5)

# get the data and labels
dat = np.vstack((cf1[b'data'], cf2[b'data'], cf3[b'data'], cf4[b'data'], cf5[b'data']))
labels = np.hstack((cf1[b'labels'], cf2[b'labels'], cf3[b'labels'], cf4[b'labels'], cf5[b'labels']))

j = 0
dat_revised =  np.empty([10000, 3072])
labels_revised = np.empty([10000])
for i in range(0, np.shape(dat)[0]):
    if labels[i] == 3 or labels[i] == 5:
        dat_revised[j] = dat[i]
        labels_revised[j] = labels[i]
        j = j + 1
        

# reshape image data and display a grid of images
# from https://stackoverflow.com/questions/35995999
m = 32
n = 32
yi = np.array(labels_revised)

#scaler = StandardScaler()
#dat_scaled = scaler.fit_transform(dat)

X_train, X_test, y_train, y_test = train_test_split(dat_revised, yi, test_size=0.25)

print("conv1: ",conv1.get_shape())
print("conv2: ",conv2.get_shape())

n_epochs = 10
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

        save_path = saver.save(sess, "./my_new_model_final.ckpt")

