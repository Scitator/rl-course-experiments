#!/usr/bin/python3

import tensorflow as tf

from rstools.tf.optimization import build_model_optimization

from agents.agent_networks import FeatureNet, QvalueNet, ValueNet
from agents.agent_states import RecurrentHiddenState


class DqrnAgent(object):
    def __init__(self, state_shape, n_actions, network, cell, special=None):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.cell_size = cell.state_size

        self.is_end = tf.placeholder(shape=[None], dtype=tf.bool, name="is_end")

        self.special = special
        self.scope = special.get("scope", "dqrn")

        with tf.variable_scope(self.scope):
            self._build_graph(network, cell)

    def _build_graph(self, network, cell):
        self.feature_net = FeatureNet(
            self.state_shape, network,
            self.special.get("feature_net", {}))

        self.hidden_state = RecurrentHiddenState(self.feature_net.feature_state, cell)

        if self.special("double_network", False):
            self.qvalue_net = QvalueNet(
                self.hidden_state.state, self.n_actions,
                dict(**self.special.get("qvalue_net", {}), **{"advantage": True}))
            self.value_net = ValueNet(
                self.hidden_state.state,
                self.special.get("value_net", {}))

            build_model_optimization(
                self.value_net,
                self.special.get("value_net_optimization", None))
            model_loss = 0.5 * (self.qvalue_net.loss + self.value_net.loss)
        else:
            self.qvalue_net = QvalueNet(
                self.hidden_state.state, self.n_actions,
                self.special.get("qvalue_net", {}))
            model_loss = self.qvalue_net.loss

        build_model_optimization(
            self.qvalue_net,
            self.special.get("qvalue_net_optimization", None))

        build_model_optimization(
            self.hidden_state,
            self.special.get("hidden_state_optimization", None),
            loss=model_loss)
        build_model_optimization(
            self.feature_net,
            self.special.get("feature_net_optimization", None),
            loss=model_loss)

    def predict_qvalues(self, sess, state_batch):
        return sess.run(
            self.qvalue_net.predicted_qvalues,
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
