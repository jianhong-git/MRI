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
epochs = 1000
batch_size = 1
display_step = 10
num_train = 382
num_test = 0

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

data_list = np.load('ADNI_data_list.npy')
num_data = len(data_list)
data_train = data_list[0:num_train]
data_test = data_list[num_train: (num_train + num_test)]
data_val = data_list[(num_train + num_test):]

hos_data_list = np.load('301_data_list.npy')
num_data = len(data_list)
hos_data_train = hos_data_list[0:100]
hos_data = []
for filename, label in hos_data_train:
    img = nib.load(filename)
    image_data = img.get_data()
    image_data = np.nan_to_num(
        (image_data - np.nanmin(image_data)) / (np.nanmax(image_data) - np.nanmin(image_data)))
    data = image_data.reshape([-1, 79 * 95 * 79])
    hos_data.append(data)

b_size = 10
num_batches = num_train // b_size
if num_train % b_size != 0:
    num_batches += 1

print("##########################")
print("Training First Conv Layer!")
for epoch_i in range(epochs):
    step = 0
    for index in range(num_batches):
        if index < num_batches - 1:
            data_batch = data_train[b_size * step: b_size * step + b_size]
        else:
            data_batch = data_train[b_size * step:]
        for filename, label in data_batch:
            img = nib.load(filename)
            image_data = img.get_data()
            image_data = np.nan_to_num(
                (image_data - np.nanmin(image_data)) / (np.nanmax(image_data) - np.nanmin(image_data)))
            data = image_data.reshape([-1, 79 * 95 * 79])
            sess.run(optimizer1, feed_dict={ae1['x']: data})
            if (step + 1) % display_step == 0:
                print("step", step + 1, "in epoch", epoch_i, "loss:",
                      sess.run(ae1['cost'], feed_dict={ae1['x']: data}))
            step += 1


print("First Conv Layer Optimization Finished!")
saver = tf.train.Saver({'w1': ae1['W'], 'b1': ae1['b']})
save_path = saver.save(sess, "./train/CAE1.ckpt")
print ("Model saved in file: ", save_path)

print("##########################")
print("Training Second Conv Layer!")
for epoch_i in range(epochs):
    step = 0
    for index in range(num_batches):
        if index < num_batches - 1:
            data_batch = data_train[b_size * step: b_size * step + b_size]
        else:
            data_batch = data_train[b_size * step:]
        for filename, label in data_batch:
            img = nib.load(filename)
            image_data = img.get_data()
            image_data = np.nan_to_num(
                (image_data - np.nanmin(image_data)) / (np.nanmax(image_data) - np.nanmin(image_data)))
            data = image_data.reshape([-1, 79 * 95 * 79])
            data = sess.run(ae1['z'], feed_dict={ae1['x']: data})
            input_data = data.reshape([batch_size, -1])
            sess.run(optimizer2, feed_dict={ae2['x']: input_data})
            if (step + 1) % display_step == 0:
                print("step", step + 1, "in epoch", epoch_i, "loss:",
                      sess.run(ae2['cost'], feed_dict={ae2['x']: input_data}))
            step += 1


print("Second Conv Layer Optimization Finished!")
saver = tf.train.Saver({'w2': ae2['W'], 'b2': ae2['b']})
save_path = saver.save(sess, "./train/CAE2.ckpt")
print ("Model saved in file: ", save_path)

# saver1 = tf.train.Saver({'w1':ae1['W'], 'b1': ae1['b']})
# saver1.restore(sess, "./train/newCAE1.ckpt")

# saver2 = tf.train.Saver({'w2':ae2['W'], 'b2':ae2['b']})
# saver2.restore(sess, "./train/newCAE2.ckpt")

print("##########################")
print("Training Third Conv Layer!")
for epoch_i in range(epochs):
    step = 0
    for index in range(num_batches):
        if index < num_batches - 1:
            data_batch = data_train[b_size * step: b_size * step + b_size]
        else:
            data_batch = data_train[b_size * step:]
        for filename, label in data_batch:
            img = nib.load(filename)
            image_data = img.get_data()
            image_data = np.nan_to_num(
                (image_data - np.nanmin(image_data)) / (np.nanmax(image_data) - np.nanmin(image_data)))
            data = image_data.reshape([-1, 79 * 95 * 79])
            h2_input_data = sess.run(ae1['z'], feed_dict={ae1['x']: data})
            h2_input_data = h2_input_data.reshape([batch_size, -1])
            data = sess.run(ae2['z'], feed_dict={ae2['x']: h2_input_data})
            input_data = data.reshape([batch_size, -1])
            sess.run(optimizer3, feed_dict={ae3['x']: input_data})
            if (step + 1) % display_step == 0:
                print("step", step + 1, "in epoch", epoch_i, "loss:",
                      sess.run(ae3['cost'], feed_dict={ae3['x']: input_data}))
            step += 1


print("Third Conv Layer Optimization Finished!")
saver = tf.train.Saver({'w3': ae3['W'], 'b3': ae3['b']})
save_path = saver.save(sess, "./train/CAE3.ckpt")
print ("Model saved in file: ", save_path)

print("##########################")
print("Saving Conv Outputs...")
index = 0
for filename, label in data_list:
    img = nib.load(filename)
    image_data = img.get_data()
    image_data = np.nan_to_num(
        (image_data - np.nanmin(image_data)) / (np.nanmax(image_data) - np.nanmin(image_data)))
    data = image_data.reshape([-1, 79 * 95 * 79])
    h2_input_data = sess.run(ae1['z'], feed_dict={ae1['x']: data})
    h2_input_data = h2_input_data.reshape([batch_size, -1])
    data = sess.run(ae2['z'], feed_dict={ae2['x']: h2_input_data})
    input_data = data.reshape([batch_size, -1])
    conv_output = sess.run(ae3['z'], feed_dict={ae3['x']: input_data})
    np.save(filename + "_conv.npy", conv_output)
    if (index + 1) % 100 == 0:
        print(index + 1, "of", len(data_list), "files saved.")
    index += 1
print("ADNI Conv Outputs saved!")

for filename, label in hos_data_list:
    img = nib.load(filename)
    image_data = img.get_data()
    image_data = np.nan_to_num(
        (image_data - np.nanmin(image_data)) / (np.nanmax(image_data) - np.nanmin(image_data)))
    data = image_data.reshape([-1, 79 * 95 * 79])
    h2_input_data = sess.run(ae1['z'], feed_dict={ae1['x']: data})
    h2_input_data = h2_input_data.reshape([batch_size, -1])
    data = sess.run(ae2['z'], feed_dict={ae2['x']: h2_input_data})
    input_data = data.reshape([batch_size, -1])
    conv_output = sess.run(ae3['z'], feed_dict={ae3['x']: input_data})
    np.save(filename + "_conv.npy", conv_output)
    if (index + 1) % 100 == 0:
        print(index + 1, "of", len(data_list), "files saved.")
    index += 1

print("All Conv Outputs saved!")


# print("##########################")
# print("Traing Fully Connected Layers!")
# b_size = 100
# num_batches = num_train//b_size
# if num_train%b_size != 0:
#     num_batches += 1
# # for epoch_i in range(epochs):
# for epoch_i in range(10000):
#     step = 0
#     for index in range(num_batches):
#         if index < num_batches-1:
#             data_batch = data_train[num_batches*step: num_batches*step+b_size]
#         else:
#             data_batch = data_train[num_batches*step:]
#         size = len(data_batch)
#         batch_data = []
#         batch_labels = []
#         for filename, label in data_batch:
#             data = np.load(filename+"_conv.npy")
#             batch_data.append(data.reshape([1, -1]))
#             batch_labels.append(int(label))
#         input_data=np.array(batch_data).reshape([size, -1])
#         labels = np.array(batch_labels).reshape([size])
#         _, loss = sess.run([optimizer, fc['loss']], feed_dict={fc['x']: input_data, fc['labels']: labels})
#         if (step+1) % 10 == 0:
#             print("step",step+1, "in epoch", epoch_i, "loss:", loss)
#         step += 1
#     if (epoch_i+1)%100 == 0:
#         val_data = []
#         val_labels = []
#         for filename, label in data_val:
#             data = np.load(filename+"_conv.npy")
#             val_data.append(data.reshape([1, -1]))
#             val_labels.append(int(label))
#         input_data=np.array(val_data).reshape([len(data_val), -1])
#         labels = np.array(val_labels).reshape([len(data_val)])
#         accuracy=sess.run(fc['correct'], feed_dict={fc['x']: input_data, fc['labels']: labels})
#         print("Validation accuracy:", accuracy, "of", len(data_val))


# test_data = []
# test_labels = []
# for filename, label in data_test:
#     data = np.load(filename+"_conv.npy")
#     test_data.append(data.reshape([1, -1]))
#     test_labels.append(int(label))
# input_data=np.array(test_data).reshape([len(data_test), -1])
# labels = np.array(test_labels).reshape([len(data_test)])
# accuracy=sess.run(fc['correct'], feed_dict={fc['x']: input_data, fc['labels']: labels})
# print("Test accuracy:", accuracy, "of", len(data_test))
