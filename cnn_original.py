from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var.name.split(":")[0]):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    return


class DNNNet(object):

    def __init__(self):
        self._create_net()
        self._prepare_loss()
        return

    def _create_net(self):
        with tf.name_scope("DNNNet"):
            self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="X")

            # initializer = tf.truncated_normal_initializer()
            initializer = tf.glorot_uniform_initializer()
            W_fc1 = tf.get_variable("W_fc1", shape=[784, 64], dtype=tf.float32, initializer=initializer)
            b_fc1 = tf.get_variable("b_fc1", shape=[64], dtype=tf.float32, initializer=initializer)

            W_fc2 = tf.get_variable("W_fc2", shape=[64, 10], dtype=tf.float32, initializer=initializer)
            b_fc2 = tf.get_variable("b_fc2", shape=[10], dtype=tf.float32, initializer=initializer)

            X_ = tf.reshape(self.X, [-1, 784])
            h_fc1 = tf.nn.relu(tf.matmul(X_, W_fc1) + b_fc1)
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
            self.logits = h_fc2
            self.y_pred = tf.nn.softmax(self.logits)
            self.label_pred = tf.argmax(self.y_pred, axis=1)

        with tf.name_scope("dnn_summaries_var"):
            variable_summaries(W_fc1)
            variable_summaries(b_fc1)
            variable_summaries(W_fc2)
            variable_summaries(b_fc2)
        return

    def _prepare_loss(self):
        self.y_true = tf.placeholder(tf.float32, [None, 10])
        # format 1:
        self.loss = - tf.reduce_sum(self.y_true * tf.log(self.y_pred))
        # or format 2:
        # self.loss = tf.losses.softmax_cross_entropy(self.label, self.logits)
        return


class CNNNet(object):

    def __init__(self, training=True):
        self._training = training
        self._create_net()
        self._prepare_loss()
        return

    def _create_net(self):
        with tf.name_scope("CNNNet"):
            self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="X")

            h_conv1 = tf.layers.conv2d(
                inputs=self.X,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name="h_conv1")
            h_pool1 = tf.layers.max_pooling2d(inputs=h_conv1, pool_size=[2, 2], strides=2, name="h_pool1")

            h_conv2 = tf.layers.conv2d(
                inputs=h_pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name="h_conv2")
            h_pool2 = tf.layers.max_pooling2d(inputs=h_conv2, pool_size=[2, 2], strides=2, name="h_pool2")

            # fully-connected Layer
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.layers.dense(inputs=h_pool2_flat, units=1024, activation=tf.nn.relu, name="h_fc1")

            # you can use a placeholder to dynamically adjust the dropout rate during training or test
            h_dropout = tf.layers.dropout(inputs=h_fc1, rate=0.5, training=self._training)

            # Logits Layer
            self.logits = tf.layers.dense(inputs=h_dropout, units=10, name="h_fc2")
            self.y_pred = tf.nn.softmax(self.logits)
            self.label_pred = tf.argmax(self.y_pred, axis=1)

        with tf.name_scope("cnn_summaries_var"):
            tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return

    def _prepare_loss(self):
        self.y_true = tf.placeholder(tf.float32, [None, 10])
        # format 1:
        self.loss = - tf.reduce_sum(self.y_true * tf.log(self.y_pred))
        # or format 2:
        # self.loss = tf.losses.softmax_cross_entropy(self.label, self.logits)
        return


def main(args):
    # refer to: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
    mnist = input_data.read_data_sets("data/", one_hot=True)
    net = DNNNet()
    # net = CNNNet()

    saver = tf.train.Saver()
    # learning rate can be dynamically adjusted by a placeholder
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    # minimize() actually has two steps: 1) compute_gradients, and 2) apply_gradients
    apply_gradients = optimizer.minimize(net.loss)

    with tf.Session() as sess:
        # init_op needs to run after initilizing adamoptimizers
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # you can call saver.restore(sess) to recover the previous training

        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./tmp/log", sess.graph)

        train_size = mnist.train.num_examples
        batch_size = 32
        n_batches = (train_size + 1) // batch_size
        n_batches = 10

        # training steps
        for k in range(n_batches):
            batch_X, batch_label = mnist.train.next_batch(batch_size)
            batch_X = np.reshape(batch_X, [-1, 28, 28, 1])
            feed_dict = {
                net.X: batch_X,
                net.y_true: batch_label,
            }
            summary, loss, _ = sess.run([merged_summary, net.loss, apply_gradients], feed_dict=feed_dict)
            train_writer.add_summary(summary)
            print("batch:{} / loss:{}".format(k, loss))
            # break

        # you can use early stopping to save the best model parameters
        save_path = saver.save(sess, "./tmp/model/model.ckpt")
        print("Model saved in path: %s" % save_path)
    return


if __name__ == "__main__":
    tf.app.run()
