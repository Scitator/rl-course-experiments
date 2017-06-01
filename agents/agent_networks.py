import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class FeatureNet(object):
    def __init__(self, state_shape, network, special=None):
        self.special = special or {}
        self.state_shape = state_shape

        self.states = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32, name="states")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.loss = None
        self.optimizer = None
        self.train_op = None

        self.relative_scope = self.special.get("scope", "feature_network")
        self.scope = tf.get_variable_scope().name + "/" + self.relative_scope

        self.feature_state = network(
            self.states,
            scope=self.relative_scope + "/feature",
            reuse=self.special.get("reuse_feature", False),
            is_training=self.is_training)


class PolicyNet(object):
    def __init__(self, hidden_state, n_actions, special=None):
        self.special = special or {}
        self.n_actions = n_actions

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.cumulative_rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="rewards")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optimizer = None
        self.train_op = None

        self.relative_scope = self.special.get("scope", "policy_network")
        self.scope = tf.get_variable_scope().name + "/" + self.relative_scope

        self.predicted_probs = self._probs(
            hidden_state,
            scope=self.relative_scope + "/probs",
            reuse=self.special.get("reuse_probs", False)) + 1e-8

        batch_size = tf.shape(self.actions)[0]
        predicted_ids = tf.range(batch_size) * tf.shape(self.predicted_probs)[1] + self.actions

        self.predicted_probs_for_actions = tf.gather(
            tf.reshape(self.predicted_probs, [-1]), predicted_ids)

        J = -tf.reduce_mean(tf.log(self.predicted_probs_for_actions) * self.cumulative_rewards)
        self.loss = J

        # a bit of regularization
        if self.special.get("entropy_loss", True):
            entropy = tf.reduce_mean(
                tf.reduce_sum(
                    self.predicted_probs * tf.log(self.predicted_probs),
                    axis=-1))
            entropy *= self.special.get("entropy_factor", 0.01)
            self.loss += entropy

    def _probs(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            probs = tf.layers.dense(
                hidden_state,
                units=self.n_actions,
                activation=tf.nn.softmax)
            return probs


class ValueNet(object):
    def __init__(self, hidden_state, special=None):
        self.special = special or {}

        self.td_target = tf.placeholder(shape=[None], dtype=tf.float32, name="td_target")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optimizer = None
        self.train_op = None

        self.relative_scope = self.special.get("scope", "value_network")
        self.scope = tf.get_variable_scope().name + "/" + self.relative_scope

        self.predicted_values = self._state_value(
            hidden_state,
            scope=self.relative_scope + "/state_value",
            reuse=self.special.get("reuse_state_value", False))

        self.predicted_values_for_actions = tf.squeeze(self.predicted_values, axis=1)

        self.loss = tf.losses.mean_squared_error(
            labels=self.td_target,
            predictions=self.predicted_values_for_actions)

    def _state_value(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            state_values = tf.layers.dense(
                hidden_state,
                units=1,
                activation=None)
            return state_values


class QvalueNet(object):
    def __init__(self, hidden_state, n_actions, special=None):
        self.special = special or {}
        self.n_actions = n_actions

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.td_target = tf.placeholder(shape=[None], dtype=tf.float32, name="td_target")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optimizer = None
        self.train_op = None

        self.relative_scope = self.special.get("scope", "qvalue_network")
        self.scope = tf.get_variable_scope().name + "/" + self.relative_scope

        self.predicted_qvalues = self._qvalues(
            hidden_state,
            scope=self.relative_scope + "/qvalue",
            reuse=self.special.get("reuse_state_value", False))

        batch_size = tf.shape(self.actions)[0]
        predicted_ids = tf.range(batch_size) * tf.shape(self.predicted_qvalues)[1] + self.actions

        self.predicted_qvalues_for_actions = tf.gather(
            tf.reshape(self.predicted_qvalues, [-1]), predicted_ids)

        self.loss = tf.losses.mean_squared_error(
            labels=self.td_target,
            predictions=self.predicted_qvalues_for_actions)

    def _qvalues(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            qvalues = tf.layers.dense(
                hidden_state,
                units=self.n_actions,
                activation=None)
            if self.special.get("advantage", False):
                qvalues -= tf.reduce_mean(qvalues, axis=-1, keep_dims=True)
            return qvalues


def copy_scope_parameters(sess, net1_scope, net2_scope):
    net1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=net1_scope)
    net2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=net2_scope)
    net1_params = sorted(net1_params, key=lambda v: v.name)
    net2_params = sorted(net2_params, key=lambda v: v.name)

    update_ops = []
    for net1_v, net2_v in zip(net1_params, net2_params):
        op = net2_v.assign(net1_v)
        update_ops.append(op)

    sess.run(update_ops)


def copy_model_parameters(sess, net1, net2):
    """
    Copies the model parameters of one net to another.

    Args:
      sess: Tensorflow session instance
      net1: net to copy the parameters from
      net2: net to copy the parameters to
    """

    copy_scope_parameters(sess, net1.scope, net2.scope)
