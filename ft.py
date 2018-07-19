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
    labels = tf.placeholder(tf.int64, shape=[None])
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
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy'), name="loss")
    regularizer = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(
        W_fc2) + tf.nn.l2_loss(b_fc2) + tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(b_fc3)
    correct = tf.reduce_sum(
        tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.int32))
    return {'x': x, 'logits': logits, 'loss': loss, 'correct': correct, 'reg': regularizer, 'labels': labels,
            'W_fc1': W_fc1, 'W_fc2': W_fc2, 'W_fc3': W_fc3, 'b_fc1': b_fc1, 'b_fc2': b_fc2, 'b_fc3': b_fc3}


learning_rate = 1e-5
epochs = 200
batch_size = 1
display_step = 100
num_train = 100
num_test = 0
loss_all = np.zeros(3001)
accuracy_all = np.zeros(32)


fc = fully_connected(10 * 12 * 10 * 8, 2)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(fc['loss'])

saver4 = tf.train.Saver({'W_fc1': fc['W_fc1'], 'W_fc2': fc['W_fc2'], 'W_fc3': fc['W_fc3'],
                         'b_fc1': fc['b_fc1'], 'b_fc2': fc['b_fc2'], 'b_fc3': fc['b_fc3']})

sess = tf.Session()
sess.run(tf.global_variables_initializer())

data_list = np.load('./datalist/301_data_list.npy')
num_data = len(data_list)
data_train = data_list[0:num_train]
data_test = data_list[num_train: (num_train + num_test)]
data_val = data_list[(num_train + num_test):]
accuracy_max = 0


saver4.restore(sess, "./train/fc.ckpt")
print ("Model FC restored")


print("##########################")
print("Training Fully Connected Layers!")
b_size = 10
num_batches = num_train // b_size
if num_train % b_size != 0:
    num_batches += 1
# for epoch_i in range(epochs):
for epoch_i in range(3000):
    step = 0
    for index in range(num_batches):
        if index < num_batches - 1:
            data_batch = data_train[b_size * index: b_size * index + b_size]
        else:
            data_batch = data_train[b_size * index:]
        size = len(data_batch)
        batch_data = []
        batch_labels = []
        for filename, label in data_batch:
            data = np.load(filename + "_conv.npy")
            batch_data.append(data.reshape([1, -1]))
            batch_labels.append(int(label))
        input_data = np.array(batch_data).reshape([size, -1])
        labels = np.array(batch_labels).reshape([size])
        _, loss = sess.run([optimizer, fc['loss']], feed_dict={
                           fc['x']: input_data, fc['labels']: labels})
        if (step + 1) % 10 == 0:
            size = len(data_train)
            batch_data = []
            batch_labels = []
            for filename, label in data_train:
                data = np.load(filename + "_conv.npy")
                batch_data.append(data.reshape([1, -1]))
                batch_labels.append(int(label))
            input_data = np.array(batch_data).reshape([size, -1])
            labels = np.array(batch_labels).reshape([size])
            _, loss, correct = sess.run([optimizer, fc['loss'], fc['correct']], feed_dict={
                                        fc['x']: input_data, fc['labels']: labels})
            print("step", step + 1, "in epoch", epoch_i,
                  "loss:", loss, 'correct', correct)
            loss_all[epoch_i] = loss

        step += 1
    if (epoch_i + 1) % 100 == 0:
        val_data = []
        val_labels = []
        for filename, label in data_val:
            data = np.load(filename + "_conv.npy")
            val_data.append(data.reshape([1, -1]))
            val_labels.append(int(label))
        input_data = np.array(val_data).reshape([len(data_val), -1])
        labels = np.array(val_labels).reshape([len(data_val)])
        accuracy, logits = sess.run([fc['correct'], fc['logits']], feed_dict={
                                    fc['x']: input_data, fc['labels']: labels})
        print("Validation accuracy:", accuracy, "of", len(data_val))
        accuracy_all[(epoch_i + 1) // 100 - 1] = accuracy
        # if accuracy > accuracy_max:
        #     accuracy_max = accuracy
        pred = np.argmax(logits, axis=1)


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
print(accuracy_all)
np.savetxt("./result/ftloss.txt", loss_all, fmt='%10.5f', delimiter=",")
np.savetxt("./result/ftaccuracy.txt", accuracy_all, fmt='%.u', delimiter=",")
np.savetxt("./result/ftpred.txt", pred, fmt='%.u', delimiter=",")
np.savetxt("./result/ftlabel.txt", labels, fmt='%.u', delimiter=",")
saver = tf.train.Saver({'W_fc1': fc['W_fc1'], 'W_fc2': fc['W_fc2'], 'W_fc3': fc['W_fc3'],
                        'b_fc1': fc['b_fc1'], 'b_fc2': fc['b_fc2'], 'b_fc3': fc['b_fc3']})
save_path = saver.save(sess, "./train/fcft.ckpt")

time2 = time.time()
print(time2 - time1)
