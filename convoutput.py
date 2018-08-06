# coding=utf-8
import tensorflow as tf
import numpy as np
import nibabel as nib
import math
import os


def autoencoder(nx, ny, nz, n_input):
    x = tf.placeholder(tf.float32, [None, nx * ny * nz * n_input], name='x')
    x_input = tf.reshape(x, [-1, nx, ny, nz, n_input])

    W_conv = tf.Variable(tf.random_uniform([3, 3, 3, n_input, 8], -0.1, 0.1))
    b_conv = tf.Variable(tf.zeros([8]))
    h_conv = tf.nn.relu(tf.add(tf.nn.conv3d(x_input, W_conv, strides=[
                        1, 1, 1, 1, 1], padding='SAME'), b_conv))

    h_pool = tf.nn.max_pool3d(h_conv, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                              padding='SAME')

    W_deconv = tf.Variable(tf.random_uniform([3, 3, 3, n_input, 8], -0.1, 0.1))
    b_deconv = tf.Variable(tf.zeros([n_input]))
    y = tf.nn.relu(tf.add(tf.nn.conv3d_transpose(h_conv, W_deconv, tf.shape(
        x_input), strides=[1, 1, 1, 1, 1], padding='SAME'), b_deconv))

    cost = tf.reduce_sum(tf.nn.l2_loss(y - x_input))

    return {'x': x, 'z': h_pool, 'y': y, 'cost': cost, 'W': W_conv, 'b': b_conv}


def fully_connected(num_pix, num_classes):
    x = tf.placeholder(tf.float32, [None, num_pix], name='fc_input')
    labels = tf.placeholder(tf.int64, shape=[None])
    W_fc1 = tf.Variable(tf.truncated_normal(
        [num_pix, 2000], stddev=0.1), name='fc1_weights')
    b_fc1 = tf.Variable(tf.truncated_normal(
        [2000], stddev=0.1), name='fc1_biases')
    W_fc2 = tf.Variable(tf.truncated_normal(
        [2000, 500], stddev=0.1), name='fc2_weights')
    b_fc2 = tf.Variable(tf.truncated_normal(
        [500], stddev=0.1), name='fc2_biases')
    W_fc3 = tf.Variable(tf.truncated_normal(
        [500, num_classes], stddev=0.1), name='fc3_weights')
    b_fc3 = tf.Variable(tf.truncated_normal(
        [num_classes], stddev=0.1), name='fc3_biases')
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    logits = tf.matmul(h_fc2, W_fc3) + b_fc3
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy'), name="loss")
    regularizer = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(
        W_fc2) + tf.nn.l2_loss(b_fc2) + tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(b_fc3)
    correct = tf.reduce_sum(
        tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.int32))
    return {'x': x, 'logits': logits, 'loss': loss, 'correct': correct, 'reg': regularizer, 'labels': labels}


learning_rate = 0.001
epochs = 200
batch_size = 1
display_step = 100
num_train = 1000
num_test = 100

ae1 = autoencoder(79, 95, 79, 1)
ae2 = autoencoder(40, 48, 40, 8)
ae3 = autoencoder(20, 24, 20, 8)
fc = fully_connected(10 * 12 * 10 * 8, 2)
optimizer1 = tf.train.AdamOptimizer(learning_rate).minimize(ae1['cost'])
optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(ae2['cost'])
optimizer3 = tf.train.AdamOptimizer(learning_rate).minimize(ae3['cost'])
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(fc['loss'])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

data_list = np.load('./datalist/ADNI_test_tri.npy')
num_data = len(data_list)


saver1 = tf.train.Saver({'w1': ae1['W'], 'b1': ae1['b']})
saver1.restore(sess, "./train/model2/CAE1.ckpt")

saver2 = tf.train.Saver({'w2': ae2['W'], 'b2': ae2['b']})
saver2.restore(sess, "./train/model2/CAE2.ckpt")

saver3 = tf.train.Saver({'w3': ae3['W'], 'b3': ae3['b']})
saver3.restore(sess, "./train/model2/CAE3.ckpt")
print ("Model restored")

print("##########################")
print("Saving Conv Outputs...")
index = 0
for filename, label in data_list:#[0:1]:
    img = nib.load(filename)
    image_data = img.get_data()
    image_data = np.nan_to_num(
        (image_data - np.nanmin(image_data)) / (np.nanmax(image_data) - np.nanmin(image_data)))
    data = image_data.reshape([-1, 79 * 95 * 79])
    h2_input_data = sess.run(ae1['z'], feed_dict={ae1['x']: data})
    np.save("conv1.npy", h2_input_data)
    h2_input_data = h2_input_data.reshape([batch_size, -1])
    data = sess.run(ae2['z'], feed_dict={ae2['x']: h2_input_data})
    np.save("conv2.npy", data)
    input_data = data.reshape([batch_size, -1])
    conv_output = sess.run(ae3['z'], feed_dict={ae3['x']: input_data})
    np.save("conv3.npy", conv_output)
    np.save(filename + "_conv.npy", conv_output)
    if (index + 1) % 100 == 0:
        print(index + 1, "of", len(data_list), "files saved.")
    index += 1
print("Conv Outputs saved!")
