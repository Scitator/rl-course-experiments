#!/usr/bin/python3

import tensorflow as tf
from tensorflow.contrib import rnn

from rstools.tf.optimization import build_model_optimization

from agent_networks import FeatureNet, PolicyNet, StateNet


def get_state_variables(batch_size, cell):
    zero_states = cell.zero_state(1, tf.float32)
    if isinstance(zero_states, list):
        state_variables = []
        for i, (state_c, state_h) in enumerate(zero_states):
            init_state_c = tf.get_variable(
                name="initial_state_vector_c:{}".format(i),
                dtype=tf.float32,
                initializer=state_c,
                trainable=False)
            init_state_h = tf.get_variable(
                name="initial_state_vector_h:{}".format(i),
                dtype=tf.float32,
                initializer=state_h,
                trainable=False)
            init_state_c = tf.tile(init_state_c, [batch_size, 1])
            init_state_h = tf.tile(init_state_h, [batch_size, 1])
            state_variables.append(
                rnn.LSTMStateTuple(
                    init_state_c,
                    init_state_h))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)
    elif isinstance(zero_states, tuple):
        state_c, state_h = zero_states
        init_state_c = tf.get_variable(
            name="initial_state_vector_c",
            dtype=tf.float32,
            initializer=state_c,
            trainable=False)
        init_state_h = tf.get_variable(
            name="initial_state_vector_h",
            dtype=tf.float32,
            initializer=state_h,
            trainable=False)
        init_state_c = tf.tile(init_state_c, [batch_size,  1])
        init_state_h = tf.tile(init_state_h, [batch_size,  1])
        return rnn.LSTMStateTuple(init_state_c, init_state_h)


def get_state_update_op(state_variables, new_states, mask=None):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        if mask is None:
            update_ops.extend([
                state_variable[0].assign(new_state[0]),
                state_variable[1].assign(new_state[1])])
        else:
            update_ops.extend([
                state_variable[0].assign(
                    tf.where(mask, tf.zeros_like(new_state[0]), new_state[0])),
                state_variable[1].assign(
                    tf.where(mask, tf.zeros_like(new_state[1]), new_state[1]))])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)


class A3CLstmAgent(object):
    def __init__(self, state_shape, n_actions, network, cell, special=None):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.cell_size = cell.state_size

        self.is_end = tf.placeholder(shape=[None], dtype=tf.bool, name="is_end")

        self.special = special
        self.scope = special.get("scope", "a3c_lstm")

        # with tf.variable_scope(self.scope):
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

        logits, rnn_states = tf.nn.dynamic_rnn(
            cell, tf.expand_dims(self.feature_net.hidden_state, 1),
            sequence_length=[1] * n_games, initial_state=self.belief_state)

        self.logits = tf.squeeze(logits, 1)

        # @TODO: very hacky 2
        self.belief_update = get_state_update_op([self.belief_state], [rnn_states], self.is_end)

        self.policy_net = PolicyNet(
            self.logits, self.n_actions,
            self.special.get("policy_net", None))
        self.state_net = StateNet(
            self.logits,
            self.special.get("state_value_net", None))

        build_model_optimization(
            self.policy_net,
            self.special.get("policy_net_optimization", None))
        build_model_optimization(
            self.state_net,
            self.special.get("state_value_net_optimization", None))
        build_model_optimization(
            self.feature_net, self.special.get("feature_net_optimization", None),
            loss=0.5 * (self.state_net.loss + self.policy_net.loss))

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
