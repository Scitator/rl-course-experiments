import tensorflow as tf


def linear_network(
        states, is_training=False, scope=None, reuse=False,
        layers=None, activation_fn=tf.nn.elu, use_bn=False, dropout=-1):
    layers = layers or [16, 16]
    x = states
    with tf.variable_scope(scope or "linear_network", reuse=reuse):
        for n_out in layers:
            x = tf.layers.dense(x, n_out, activation=None)
            if use_bn:
                x = tf.layers.batch_normalization(x, training=is_training)
            x = activation_fn(x)
            if dropout > 0.0:
                x = tf.layers.dropout(x, rate=dropout, training=is_training)
        return x


def conv_network(
        states, is_training=False, scope=None, reuse=False,
        n_filters=None, kernels=None, strides=None,
        activation_fn=tf.nn.elu, use_bn=False, dropout=-1):
    n_filters = n_filters or [32, 64, 64]
    kernels = kernels or [8, 4, 4]
    strides = strides or [4, 2, 1]
    x = states
    with tf.variable_scope(scope or "linear_network", reuse=reuse):
        for n_filter, kernel, stride in zip(n_filters, kernels, strides):
            x = tf.layers.conv2d(x, n_filter, kernel, stride, activation=None)
            if use_bn:
                x = tf.layers.batch_normalization(x, training=is_training)
            x = activation_fn(x)
            if dropout > 0.0:
                x = tf.layers.dropout(x, rate=dropout, training=is_training)
        return x


def network_wrapper(network, params):
    def wrapper(states, is_training=False, scope=None, reuse=False,):
        return network(states, is_training, scope, reuse, **params)
    return wrapper
