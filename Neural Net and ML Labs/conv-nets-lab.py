import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image


# a couple of utility functions by Geron
def plot_image(image):
    plt.figure(figsize=(10,10))  # GRB added
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")

# Load sample images

# the images are an array of height x width x num_channels
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)

# color images, so 3 channels (RGB, I imagine)
batch_size, height, width, channels = dataset.shape

# create 2 filters
#filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
#filters[:, 3, :, 0] = 1  # vertical line
#filters[3, :, :, 1] = 1  # horizontal line

# create 3 filters
filters = np.zeros(shape=(7, 7, channels, 3), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line
filters[3, 3, :, 2] = 1

# Create a graph with input X plus a convolutional layer applying the 2 filters
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

# execute
with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})


# plot the feature map of every image
for image_index in (0, 1):
    for feature_map_index in (0, 1):
#        plot_image(output[image_index, :, :, feature_map_index])
        plot_image(output[0].astype(np.uint8))  # astype is needed
        plt.show()