#!/usr/bin/python3

import tensorflow as tf
from agents.agent_states import LinearHiddenState
from rstools.tf.optimization import build_model_optimization

from agents.agent_networks import FeatureNet, PolicyNet


class ReinforceAgent(object):
    def __init__(self, state_shape, n_actions, network, special=None):
        self.special = special or {}
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.special = special

        self.scope = tf.get_variable_scope().name + "/" + special.get("scope", "dqn") \
            if tf.get_variable_scope().name else special.get("scope", "dqn")

        with tf.variable_scope(self.scope):
            self._build_graph(network)

    def _build_graph(self, network):
        self.feature_net = FeatureNet(
            self.state_shape, network,
            self.special.get("feature_net", {}))

        self.hidden_state = LinearHiddenState(
            self.feature_net.feature_state,
            self.special.get("hidden_size", 512),
            self.special.get("hidden_activation", tf.nn.elu))

        self.policy_net = PolicyNet(
            self.hidden_state.state, self.n_actions,
            self.special.get("policy_net", {}))

        build_model_optimization(
            self.policy_net,
            self.special.get("policy_net_optimization", None))
        build_model_optimization(
            self.hidden_state,
            self.special.get("hidden_state_optimization", None),
            loss=self.policy_net.loss)
        build_model_optimization(
            self.feature_net,
            self.special.get("feature_net_optimization", None),
            loss=self.policy_net.loss)

    def predict_probs(self, sess, state_batch, is_training=False):
        return sess.run(
            self.policy_net.predicted_probs,
            feed_dict={
                self.feature_net.states: state_batch,
                self.feature_net.is_training: is_training})
