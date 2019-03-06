import tensorflow as tf
from archs.P_Net import batch_norm, bias_variable, weight_variable, conv2d
def max_pool_3x3(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

class R_Net():

    def __init__(self):

        self.W_conv1 = weight_variable("W_conv1", [3, 3, 3, 28])
        self.b_conv1 = bias_variable('b_conv1', [28])

        self.W_conv2 = weight_variable("W_conv2", [3, 3, 28, 48])
        self.b_conv2 = bias_variable('b_conv2', [48])

        self.W_conv3 = weight_variable("W_conv3", [2, 2, 48, 64])
        self.b_conv3 = bias_variable('b_conv3', [64])

        self.W_1 = weight_variable("W_1", [64, 10])
        self.b_1 = bias_variable('b_1', [10])

    def pr(self, inputs):
        h_conv1 = conv2d(inputs, self.W_conv1) + self.b_conv1  # output size
        h_conv1_bn = tf.nn.relu(batch_norm(h_conv1, 28))

        h_pool1 = max_pool_3x3(h_conv1_bn)

        h_conv2 = conv2d(h_pool1, self.W_conv2) + self.b_conv2  # output size
        h_conv2_bn = tf.nn.relu(batch_norm(h_conv2, 48))

        h_pool2 = max_pool_3x3(h_conv2_bn)

        h_conv3 = conv2d(h_pool2, self.W_conv3) + self.b_conv3  # output size
        h_conv3_bn = tf.nn.relu(batch_norm(h_conv3, 64))

        flatten = tf.reshape(h_conv3_bn, [-1, 64])
        h_fc1 = tf.nn.relu(tf.matmul(flatten, self.W_1) + self.b_1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=1)

        return h_fc1_drop
