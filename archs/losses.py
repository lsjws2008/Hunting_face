import tensorflow as tf


def Euclidean_loss(output, target):
    return tf.abs(target - output)


def cross_entropy_loss(output, target):
    output = tf.squeeze(output)
    target = tf.squeeze(target)
    return tf.losses.softmax_cross_entropy(target, output)
