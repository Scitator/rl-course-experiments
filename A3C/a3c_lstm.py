#!/usr/bin/python3

import tensorflow as tf
from rstools.tf.optimization import build_model_optimization, build_scope_optimization
from tensorflow.contrib import rnn

from agents.agent_networks import FeatureNet, PolicyNet, ValueNet
from agents.agent_states import RecurrentHiddenState, get_state_variables, get_state_update_op


class A3CLstmAgent(object):
    def __init__(self, state_shape, n_actions, network, special=None):
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.special = special
        self.scope = special.get("scope", "a3c_lstm")

        with tf.variable_scope(self.scope):
            self._build_graph(network)

    def _build_graph(self, network):
        self.feature_net = FeatureNet(
            self.state_shape, network,
            self.special.get("feature_net", None))

        self.hidden_state = RecurrentHiddenState(
            self.feature_net.feature_state,
            self.special.get("hidden_size", 512),
            self.special.get("hidden_activation", tf.tanh),
            self.special.get("batch_size", 1))

        self.policy_net = PolicyNet(
            self.hidden_state.state, self.n_actions,
            self.special.get("policy_net", {}))
        self.value_net = ValueNet(
            self.hidden_state.state,
            self.special.get("value_net", {}))

        build_model_optimization(
            self.policy_net,
            self.special.get("policy_net_optimization", None))
        build_model_optimization(
            self.value_net,
            self.special.get("value_net_optimization", None))
        build_model_optimization(
            self.hidden_state,
            self.special.get("hidden_state_optimization", None),
            loss=0.5 * (self.value_net.loss + self.policy_net.loss))
        build_model_optimization(
            self.feature_net, self.special.get("feature_net_optimization", None),
            loss=0.5 * (self.value_net.loss + self.policy_net.loss))

    def predict_values(self, sess, state_batch):
        return sess.run(
            self.value_net.predicted_values_for_actions,
            feed_dict={
                self.feature_net.states: state_batch,
                self.feature_net.is_training: False
            })

    def predict_probs(self, sess, state_batch):
        return sess.run(
            self.policy_net.predicted_probs,
            feed_dict={
                self.feature_net.states: state_batch,
                self.feature_net.is_training: False
            })

    def update_belief_state(self, sess, state_batch, done_batch):
        _ = sess.run(
            self.hidden_state.belief_update,
            feed_dict={
                self.feature_net.states: state_batch,
                self.hidden_state.is_end: done_batch,
                self.feature_net.is_training: False
            })

    def assign_belief_state(self, sess, new_belief):
        _ = sess.run(
            self.hidden_state.belief_assign,
            feed_dict={
                self.hidden_state.belief_out: new_belief
            })

    def get_belief_state(self, sess):
        return sess.run(self.hidden_state.belief_state)
