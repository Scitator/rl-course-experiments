#!/usr/bin/python3

import tensorflow as tf
from rstools.tf.optimization import build_model_optimization
from agent_networks import FeatureNet, PolicyNet, StateNet


class A3CFFAgent(object):
    def __init__(self, state_shape, n_actions, network, special=None):
        self.special = special or {}
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.special = special

        self._buiild_graph(network)

    def _buiild_graph(self, network):
        self.feature_net = FeatureNet(
            self.state_shape, network,
            self.special.get("feature_get", None))
        self.hidden_state = tf.layers.dense(
            self.feature_net.hidden_state,
            self.special.get("hidden_size", 512),
            activation=self.special.get("hidden_activation", tf.nn.elu))

        self.policy_net = PolicyNet(
            self.hidden_state, self.n_actions,
            self.special.get("policy_net", None))
        self.state_net = StateNet(
            self.hidden_state,
            self.special.get("state_net", None))

        build_model_optimization(
            self.policy_net,
            self.special.get("policy_net_optimization", None))
        build_model_optimization(
            self.state_net,
            self.special.get("state_net_optimization", None))
        build_model_optimization(
            self.feature_net,
            self.special.get("feature_net_optimization", None),
            loss=0.5 * (self.policy_net.loss + self.state_net.loss))

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
