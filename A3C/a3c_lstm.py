#!/usr/bin/python3

import tensorflow as tf
from tensorflow.contrib import rnn

from rstools.tf.optimization import build_model_optimization, build_scope_optimization

from agent_networks import FeatureNet, PolicyNet, ValueNet, \
    get_state_variables, get_state_update_op


class A3CLstmAgent(object):
    def __init__(self, state_shape, n_actions, network, cell, special=None):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.cell_size = cell.state_size

        self.is_end = tf.placeholder(shape=[None], dtype=tf.bool, name="is_end")

        self.special = special
        self.scope = special.get("scope", "a3c_lstm")

        with tf.variable_scope(self.scope):
            self._build_graph(network, cell)

    def _build_graph(self, network, cell):
        self.feature_net = FeatureNet(
            self.state_shape, network,
            self.special.get("feature_net", None))

        n_games = tf.unstack(tf.shape(self.is_end))
        with tf.variable_scope("belief_state"):
            self.belief_state = get_state_variables(n_games, cell)
            # very bad dark magic, need to refactor all of this
            # supports only ine layer cell
            self.belief_out = tf.placeholder(
                tf.float32, [2, n_games, cell.output_size])
            l = tf.unstack(self.belief_out, axis=0)
            rnn_tuple_state = rnn.LSTMStateTuple(l[0], l[1])
            self.belief_assign = get_state_update_op([self.belief_state], [rnn_tuple_state])

        with tf.variable_scope("hidden_state"):
            logits, rnn_states = tf.nn.dynamic_rnn(
                cell, tf.expand_dims(self.feature_net.feature_state, 1),
                sequence_length=[1] * n_games, initial_state=self.belief_state)

        self.logits = tf.squeeze(logits, 1)

        # @TODO: very hacky 2
        self.belief_update = get_state_update_op([self.belief_state], [rnn_states], self.is_end)

        self.policy_net = PolicyNet(
            self.logits, self.n_actions,
            self.special.get("policy_net", {}))
        self.state_net = ValueNet(
            self.logits,
            self.special.get("value_net", {}))

        build_model_optimization(
            self.policy_net,
            self.special.get("policy_net_optimization", None))
        build_model_optimization(
            self.state_net,
            self.special.get("value_net_optimization", None))
        build_model_optimization(
            self.feature_net, self.special.get("feature_net_optimization", None),
            loss=0.5 * (self.state_net.loss + self.policy_net.loss))

        self.hidden_optimizer, self.hidden_train_op = build_scope_optimization(
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=self.scope + "/hidden_state"),
            optimization_params=self.special.get("feature_net_optimization", None),
            loss=self.feature_net.loss)

    def predict_value(self, sess, state_batch):
        return sess.run(
            self.state_net.predicted_values,
            feed_dict={
                self.feature_net.states: state_batch,
                self.feature_net.is_training: False
            })

    def predict_action(self, sess, state_batch):
        return sess.run(
            self.policy_net.predicted_probs,
            feed_dict={
                self.feature_net.states: state_batch,
                self.feature_net.is_training: False
            })

    def update_belief_state(self, sess, state_batch, done_batch):
        _ = sess.run(
            self.belief_update,
            feed_dict={
                self.feature_net.states: state_batch,
                self.is_end: done_batch,
                self.feature_net.is_training: False
            })

    def assign_belief_state(self, sess, new_belief):
        _ = sess.run(
            self.belief_assign,
            feed_dict={
                self.belief_out: new_belief
            })

    def get_belief_state(self, sess):
        return sess.run(self.belief_state)
