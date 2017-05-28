#!/usr/bin/python3

import tensorflow as tf

from rstools.tf.optimization import build_model_optimization

from agents.agent_networks import FeatureNet, QvalueNet, ValueNet
from agents.agent_states import RecurrentHiddenState


class DrqnAgent(object):
    def __init__(self, state_shape, n_actions, network, special=None):
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.is_end = tf.placeholder(shape=[None], dtype=tf.bool, name="is_end")

        self.special = special
        self.scope = special.get("scope", "dqrn")

        with tf.variable_scope(self.scope):
            self._build_graph(network)

    def _build_graph(self, network):
        self.feature_net = FeatureNet(
            self.state_shape, network,
            self.special.get("feature_net", {}))

        self.hidden_state = RecurrentHiddenState(
            self.feature_net.feature_state,
            self.special.get("hidden_size", 512),
            self.special.get("hidden_activation", tf.tanh),
            self.special.get("batch_size", 1))

        if self.special.get("dueling_network", False):
            self.qvalue_net = QvalueNet(
                self.hidden_state.state, self.n_actions,
                dict(**self.special.get("qvalue_net", {}), **{"advantage": True}))
            self.value_net = ValueNet(
                self.hidden_state.state,
                self.special.get("value_net", {}))

            # a bit hacky way
            self.agent_loss = tf.losses.mean_squared_error(
                labels=self.qvalue_net.td_target,
                predictions=self.value_net.predicted_values_for_actions +
                            self.qvalue_net.predicted_qvalues_for_actions)

            build_model_optimization(
                self.value_net,
                self.special.get("value_net_optimization", None),
                loss=self.agent_loss)
        else:
            self.qvalue_net = QvalueNet(
                self.hidden_state.state, self.n_actions,
                self.special.get("qvalue_net", {}))
            self.agent_loss = self.qvalue_net.loss

        build_model_optimization(
            self.qvalue_net,
            self.special.get("qvalue_net_optimization", None))

        build_model_optimization(
            self.hidden_state,
            self.special.get("hidden_state_optimization", None),
            loss=self.agent_loss)
        build_model_optimization(
            self.feature_net,
            self.special.get("feature_net_optimization", None),
            loss=self.agent_loss)

    def predict_qvalues(self, sess, state_batch):
        if self.special.get("dueling_network", False):
            return sess.run(
                self.value_net.predicted_values + self.qvalue_net.predicted_qvalues,
                feed_dict={
                    self.feature_net.states: state_batch,
                    self.feature_net.is_training: False})
        else:
            return sess.run(
                self.qvalue_net.predicted_qvalues,
                feed_dict={
                    self.feature_net.states: state_batch,
                    self.feature_net.is_training: False})

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
