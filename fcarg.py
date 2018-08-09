# coding=utf-8
import tensorflow as tf
import numpy as np
import nibabel as nib
import math
import os
import time
import argparse
time1 = time.time()
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MRI project')
    parser.add_argument('--batchsize', type=int, default=64, metavar='N',  # len(train_set),  #
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='M',
                        help='Adam weight_decay (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--fc1', type=int, default=500, metavar='S',
                        help='length for the first fully connected layer (default: 500)')
    parser.add_argument('--fc2', type=int, default=100, metavar='S',
                        help='length for the second fully connected layer (default: 100)')
    args = parser.parse_args()
    return args


def fully_connected(num_pix, num_classes, fc1, fc2):
    x = tf.placeholder(tf.float32, [None, num_pix], name='fc_input')
    labels = tf.placeholder(tf.int64, shape=[None])
    W_fc1 = tf.Variable(tf.truncated_normal(
        [num_pix, fc1], stddev=0.1), name='fc1_weights')
    b_fc1 = tf.Variable(tf.truncated_normal(
        [fc1], stddev=0.1), name='fc1_biases')
    W_fc2 = tf.Variable(tf.truncated_normal(
        [fc1, fc2], stddev=0.1), name='fc2_weights')
    b_fc2 = tf.Variable(tf.truncated_normal(
        [fc2], stddev=0.1), name='fc2_biases')
    W_fc3 = tf.Variable(tf.truncated_normal(
        [fc2, num_classes], stddev=0.1), name='fc3_weights')
    b_fc3 = tf.Variable(tf.truncated_normal(
        [num_classes], stddev=0.1), name='fc3_biases')
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    logits = tf.matmul(h_fc2, W_fc3) + b_fc3
    regularizer = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(
        W_fc2) + tf.nn.l2_loss(b_fc2) + tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(b_fc3)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                          (labels=labels, logits=logits, name='xentropy'), name="loss")  # +0*regularizer
    correct = tf.reduce_sum(
        tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.int32))
    return {'x': x, 'logits': logits, 'loss': loss, 'correct': correct, 'reg': regularizer, 'labels': labels,
            'W_fc1': W_fc1, 'W_fc2': W_fc2, 'W_fc3': W_fc3, 'b_fc1': b_fc1, 'b_fc2': b_fc2, 'b_fc3': b_fc3}


# global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(
#     1e-3, global_step, 200, 0.5, staircase=True)
def main():
    learning_rate = args.lr
    epochs = args.epochs
    b_size = args.batchsize
    display_step = 100
    # num_train = 1800  # an  # tri
    # num_test = 0

    fc = fully_connected(10 * 12 * 10 * 8, 2, args.fc1, args.fc2)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        fc['loss'])  # , global_step=global_step

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    data_list = np.load('./datalist/ADNI_train_an.npy')
    num_data = len(data_list)
    print(num_data)
    data_train = data_list  # [0:num_train]
    # data_test = np.load('./datalist/ADNI_test_an.npy')
    data_val = np.load('./datalist/ADNI_test_an.npy')
    loss_all = np.zeros(epochs)
    accuracy_all = np.zeros(epochs // 10 + 1)

    print("##########################")
    print("Traing Fully Connected Layers!")

    # print(len(data_list))
    num_batches = num_data // b_size
    if num_data % b_size != 0:
        num_batches += 1
    # for epoch_i in range(epochs):
    for epoch_i in range(epochs):
        step = 0
        for index in range(num_batches):
            if index < num_batches - 1:
                data_batch = data_train[b_size * step: b_size * step + b_size]
            else:
                data_batch = data_train[b_size * step:]
            size = len(data_batch)
            batch_data = []
            batch_labels = []
            for filename, label in data_batch:
                data = np.load(filename + "_conv.npy")
                batch_data.append(data.reshape([1, -1]))
                batch_labels.append(int(label))
            # print(epoch_i, index, data.reshape([1, -1]), batch_data)
            input_data = np.array(batch_data).reshape([size, -1])
            labels = np.array(batch_labels).reshape([size])
            _, loss = sess.run([optimizer, fc['loss']], feed_dict={
                               fc['x']: input_data, fc['labels']: labels})
            # if (step + 1) % 10 == 0:
            #     print("step", step + 1, "in epoch", epoch_i, "loss:", loss)
            loss_all[epoch_i] = loss
            step += 1
        if (epoch_i + 1) % 10 == 0:
            val_data = []
            val_labels = []
            for filename, label in data_val:
                data = np.load(filename + "_conv.npy")
                val_data.append(data.reshape([1, -1]))
                val_labels.append(int(label))
            input_data = np.array(val_data).reshape([len(data_val), -1])
            labels = np.array(val_labels).reshape([len(data_val)])
            accuracy = sess.run(fc['correct'], feed_dict={
                                fc['x']: input_data, fc['labels']: labels})
            # loss = sess.run(fc['loss'], feed_dict={
            #     fc['x']: input_data, fc['labels']: labels})
            print('epoch = %d' % (epoch_i + 1) + '\n' +
                  # "Training loss:", loss, '\n' +
                  "Validation accuracy:", accuracy, "of", len(data_val))
            # loss_all[(epoch_i + 1) // 10 - 1] = loss
            accuracy_all[(epoch_i + 1) // 10 - 1] = accuracy

    with open('./result/model2/fcarg/accuracyfc%dfc%dbatch%depoch%d.txt' % (args.fc1, args.fc2, args.batchsize, args.epochs), "w+") as text_file1:
        np.savetxt('./result/model2/fcarg/accuracyfc%dfc%dbatch%depoch%d.txt' % (args.fc1,
                                                                                 args.fc2, args.batchsize, args.epochs), accuracy_all, fmt='%.u', delimiter=",")
    with open('./result/model2/fcarg/lossfc%dfc%dbatch%depoch%d.txt' % (args.fc1, args.fc2, args.batchsize, args.epochs), "w+") as text_file2:
        np.savetxt('./result/model2/fcarg/lossfc%dfc%dbatch%depoch%d.txt' % (args.fc1,
                                                                             args.fc2, args.batchsize, args.epochs), loss_all, fmt='%10.5f', delimiter=",")

    saver = tf.train.Saver({'W_fc1': fc['W_fc1'], 'W_fc2': fc['W_fc2'], 'W_fc3': fc['W_fc3'],
                            'b_fc1': fc['b_fc1'], 'b_fc2': fc['b_fc2'], 'b_fc3': fc['b_fc3']})
    save_path = saver.save(sess, "./train/model2/fc%dfc%dbatch%depoch%d/fc.ckpt" %
                           (args.fc1, args.fc2, args.batchsize, args.epochs))
    time2 = time.time()
    print(time2 - time1)


if __name__ == '__main__':
    args = parse_args()
    main()
