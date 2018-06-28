# coding=utf-8
import tensorflow as tf
import numpy as np
import nibabel as nib
import math
import os
import time
time1 = time.time()


def fully_connected(num_pix, num_classes):
    x = tf.placeholder(tf.float32, [None, num_pix], name='fc_input')
    # labels = tf.placeholder(tf.int64, shape=[None])
    W_fc1 = tf.Variable(tf.truncated_normal(
        [num_pix, 600], stddev=0.1), name='fc1_weights')
    b_fc1 = tf.Variable(tf.truncated_normal(
        [600], stddev=0.1), name='fc1_biases')
    W_fc2 = tf.Variable(tf.truncated_normal(
        [600, 100], stddev=0.1), name='fc2_weights')
    b_fc2 = tf.Variable(tf.truncated_normal(
        [100], stddev=0.1), name='fc2_biases')
    W_fc3 = tf.Variable(tf.truncated_normal(
        [100, num_classes], stddev=0.1), name='fc3_weights')
    b_fc3 = tf.Variable(tf.truncated_normal(
        [num_classes], stddev=0.1), name='fc3_biases')
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    logits = tf.matmul(h_fc2, W_fc3) + b_fc3
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, name='xentropy'), name="loss")
    regularizer = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(
        W_fc2) + tf.nn.l2_loss(b_fc2) + tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(b_fc3)
    # correct = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.int32))
    return {'x': x, 'logits': logits, 'reg': regularizer,
            'W_fc1': W_fc1, 'W_fc2': W_fc2, 'W_fc3': W_fc3, 'b_fc1': b_fc1, 'b_fc2': b_fc2, 'b_fc3': b_fc3}


learning_rate = 0.0001
epochs = 200
batch_size = 1
display_step = 100
num_train = 1034
num_test = 0
loss_all = np.zeros(3001)
accuracy_all = np.zeros(32)


fc = fully_connected(10 * 12 * 10 * 8, 2)
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(fc['loss'])

saver4 = tf.train.Saver({'W_fc1': fc['W_fc1'], 'W_fc2': fc['W_fc2'], 'W_fc3': fc['W_fc3'],
                         'b_fc1': fc['b_fc1'], 'b_fc2': fc['b_fc2'], 'b_fc3': fc['b_fc3']})

sess = tf.Session()
sess.run(tf.global_variables_initializer())

data_list = np.load('data_list.npy')
data_list = data_list[(num_train + num_test):]
num_data = len(data_list)
print(num_data)

saver4.restore(sess, "./train/fc.ckpt")
print ("Model FC restored")


test_data = []
test_labels = []
for filename, label in data_list:
    data = np.load(filename + "_conv.npy")
    test_data.append(data.reshape([1, -1]))
    test_labels.append(int(label))
input_data = np.array(test_data).reshape([-1, 9600])
#labels = np.array(test_labels).reshape([18])
pred = sess.run(fc['logits'], feed_dict={fc['x']: input_data})


np.savetxt("./train/pred.txt", pred, delimiter=",")
# np.savetxt("./train/pred.txt", pred, delimiter=",")

np.savetxt("./train/label.txt", test_labels, delimiter=",")


saver = tf.train.Saver({'W_fc1': fc['W_fc1'], 'W_fc2': fc['W_fc2'], 'W_fc3': fc['W_fc3'],
                        'b_fc1': fc['b_fc1'], 'b_fc2': fc['b_fc2'], 'b_fc3': fc['b_fc3']})
# save_path = saver.save(sess, "./train/fc.ckpt")

time2 = time.time()
print(time2 - time1)
