import tensorflow as tf
import numpy as np

weight_decay = tf.constant(0.0005, dtype=tf.float32)


def weight_variable(name, shape, init=None):
    if init is not None:
        initial = tf.get_variable(name, shape,
                                  initializer=tf.constant_initializer(value=init),
                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    else:
        initial = tf.get_variable(name=name, shape=shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
    return tf.Variable(initial)


def bias_variable(name, shape, init=None):
    if init is not None:
        initial = tf.get_variable(name, shape=shape,
                                  initializer=tf.constant_initializer(value=init))
    else:
        initial = initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def batch_norm(x, n_out):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        """
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        """

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = mean_var_with_update()
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed



class P_Net():

    def __init__(self):

        self.W_conv1 = weight_variable("P_W_conv1", [3, 3, 3, 10])
        self.b_conv1 = bias_variable('P_b_conv1', [10])

        self.W_conv2 = weight_variable("P_W_conv2", [3, 3, 10, 16])
        self.b_conv2 = bias_variable('P_b_conv2', [16])


        self.W_conv3 = weight_variable("P_W_conv3", [3, 3, 16, 32])
        self.b_conv3 = bias_variable('P_b_conv3', [32])

        self.W_conv4 = weight_variable("P_W_conv4", [1, 1, 32, 10])
        self.b_conv4 = bias_variable('P_b_conv4', [10])

    def pr(self, inputs):
        h_conv1 = conv2d(inputs, self.W_conv1) + self.b_conv1  # output size
        h_conv1_bn = tf.nn.relu(batch_norm(h_conv1, 10))

        h_pool1 = max_pool_3x3(h_conv1_bn)

        h_conv2 = conv2d(h_pool1, self.W_conv2) + self.b_conv2  # output size
        h_conv2_bn = tf.nn.relu(batch_norm(h_conv2, 16))

        h_conv3 = conv2d(h_conv2_bn, self.W_conv3) + self.b_conv3  # output size
        h_conv3_bn = tf.nn.relu(batch_norm(h_conv3, 32))

        h_conv4 = conv2d(h_conv3_bn, self.W_conv4) + self.b_conv4  # output size
        h_conv4_bn = tf.nn.relu(batch_norm(h_conv4, 10))

        return h_conv4_bn
