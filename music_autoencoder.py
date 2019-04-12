""" Auto Encoder Example.
Build a 2 layers auto-encoder with TensorFlow to compress 4-bar scores with
16ths resolution to
lower latent space and then reconstruct them.

Author: Maximos Kaliakatsos Papakostas, based on turial by Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
# __MAX__
# from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# __MAX__
'''
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
'''
# load data
rows = []
columns = []
with open('saved_data' + os.sep + 'data_tower.pickle', 'rb') as handle:
    d = pickle.load(handle)
    serialised_segments = d['serialised_segments']
    rows = d['rows']
    columns = d['columns']

# Training Parameters
learning_rate = 0.01
num_steps = 30000 # what should this be?
batch_size = 256

# __MAX__
# split in batches
batches_train = []
batches_test = []
tmp_counter = 1
batch_idx_start = 0
batch_idx_end = batch_idx_start + batch_size
while batch_idx_end < serialised_segments.shape[0]:
    # decide whether to put it in test or train
    if tmp_counter%10 == 0:
        batch_idx_end = batch_idx_start + 4
        batches_test.append( serialised_segments[ batch_idx_start:batch_idx_end,: ] )
        batch_idx_start += 4
        batch_idx_end = batch_idx_start + batch_size
    else:
        batch_idx_end = batch_idx_start + batch_size
        batches_train.append( serialised_segments[ batch_idx_start:batch_idx_end,: ] )
        batch_idx_start += batch_size
    tmp_counter += 1

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = rows*columns # serialised score

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    # __MAX__
    batch_idx = 0
    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x = batches_train[ batch_idx ]
        batch_idx += 1
        batch_idx = batch_idx%len(batches_train)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((rows * n, columns * n))
    canvas_recon = np.empty((rows * n, columns * n))
    # __MAX__
    batch_idx = 0
    for i in range(n):
        # MNIST test set
        batch_x = batches_test[ batch_idx ][:n,:]
        batch_idx += 1
        batch_idx = batch_idx%len(batches_test)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * rows:(i + 1) * rows, j * columns:(j + 1) * columns] = \
                batch_x[j].reshape([rows, columns])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * rows:(i + 1) * rows, j * columns:(j + 1) * columns] = \
                g[j].reshape([rows, columns])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.savefig('figs/original.png', dpi=300); plt.clf()
    # plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.savefig('figs/reconstructed.png', dpi=300); plt.clf()
    # plt.show()