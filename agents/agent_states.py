import tensorflow as tf
from tensorflow.contrib import rnn

#
# def get_state_variables(batch_size, cell):
#     zero_states = cell.zero_state(1, tf.float32)
#     if isinstance(zero_states, list):
#         state_variables = []
#         for i, (state_c, state_h) in enumerate(zero_states):
#             init_state_c = tf.get_variable(
#                 name="initial_state_vector_c:{}".format(i),
#                 dtype=tf.float32,
#                 initializer=state_c,
#                 trainable=False)
#             init_state_h = tf.get_variable(
#                 name="initial_state_vector_h:{}".format(i),
#                 dtype=tf.float32,
#                 initializer=state_h,
#                 trainable=False)
#             init_state_c = tf.tile(init_state_c, [batch_size, 1])
#             init_state_h = tf.tile(init_state_h, [batch_size, 1])
#             state_variables.append(
#                 rnn.LSTMStateTuple(
#                     init_state_c,
#                     init_state_h))
#         # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
#         return tuple(state_variables)
#     elif isinstance(zero_states, tuple):
#         state_c, state_h = zero_states
#         init_state_c = tf.get_variable(
#             name="initial_state_vector_c",
#             dtype=tf.float32,
#             initializer=state_c,
#             trainable=False)
#         init_state_h = tf.get_variable(
#             name="initial_state_vector_h",
#             dtype=tf.float32,
#             initializer=state_h,
#             trainable=False)
#         import pdb; pdb.set_trace()
#         init_state_c = tf.tile(init_state_c, [batch_size,  1])
#         init_state_h = tf.tile(init_state_h, [batch_size,  1])
#         return rnn.LSTMStateTuple(init_state_c, init_state_h)
#
#
# def get_state_update_op(state_variables, new_states, mask=None):
#     # Add an operation to update the train states with the last state tensors
#     update_ops = []
#     for state_variable, new_state in zip(state_variables, new_states):
#         # Assign the new state to the state variables on this layer
#         if mask is None:
#             # @TODO: error here, tiled Tensor has no assign
#             update_ops.extend([
#                 state_variable[0].assign(new_state[0]),
#                 state_variable[1].assign(new_state[1])])
#         else:
#             update_ops.extend([
#                 state_variable[0].assign(
#                     tf.where(mask, tf.zeros_like(new_state[0]), new_state[0])),
#                 state_variable[1].assign(
#                     tf.where(mask, tf.zeros_like(new_state[1]), new_state[1]))])
#     # Return a tuple in order to combine all update_ops into a single operation.
#     # The tuple's actual value should not be used.
#     return tf.tuple(update_ops)
#
#
# @TODO: rewrite for any cell, and refactor it
# @TODO: Not working without known batch_size, so static!
# class RecurrentHiddenState(object):
#     def __init__(self, feature_state, size=512, activation=tf.tanh):
#         self.is_end = tf.placeholder(shape=[None], dtype=tf.bool, name="is_end")
#
#         self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
#         self.global_step = tf.Variable(0, name='global_step', trainable=False)
#
#         self.loss = None
#         self.optimizer = None
#         self.train_op = None
#
#         self.relative_scope = "hidden_state"
#         self.scope = tf.get_variable_scope().name + "/" + self.relative_scope
#         batch_size = tf.unstack(tf.shape(feature_state))[0]
#
#         with tf.variable_scope(self.relative_scope):
#             self.cell = rnn.LSTMCell(size, activation=activation)
#             self.belief_state = get_state_variables(batch_size, self.cell)
#             # very bad dark magic, need to refactor all of this
#             # supports only ine layer cell
#             self.belief_out = tf.placeholder(
#                 tf.float32, [2, None, self.cell.output_size])
#             l = tf.unstack(self.belief_out, axis=0)
#             rnn_tuple_state = rnn.LSTMStateTuple(l[0], l[1])
#             import pdb; pdb.set_trace()
#             self.belief_assign = get_state_update_op([self.belief_state], [rnn_tuple_state])
#
#             logits, rnn_states = tf.nn.dynamic_rnn(
#                 self.cell, tf.expand_dims(feature_state, 1),
#                 sequence_length=[1] * batch_size, initial_state=self.belief_state)
#
#             self.state = tf.squeeze(logits, 1)
#
#             # @TODO: very hacky 2
#             self.belief_update = get_state_update_op(
#                 [self.belief_state], [rnn_states], self.is_end)


def get_state_variables(batch_size, cell):
    zero_states = cell.zero_state(batch_size, tf.float32)
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


class LinearHiddenState(object):
    def __init__(self, feature_state, size=512, activation=None):

        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.loss = None
        self.optimizer = None
        self.train_op = None

        self.relative_scope = "hidden_state"
        self.scope = tf.get_variable_scope().name + "/" + self.relative_scope

        with tf.variable_scope(self.relative_scope):
            self.state = tf.layers.dense(
                feature_state,
                size,
                activation=activation)


class RecurrentHiddenState(object):
    def __init__(self, feature_state, size=512, activation=tf.tanh, batch_size=1):
        self.is_end = tf.placeholder(shape=[None], dtype=tf.bool, name="is_end")

        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.loss = None
        self.optimizer = None
        self.train_op = None

        self.relative_scope = "hidden_state"
        self.scope = tf.get_variable_scope().name + "/" + self.relative_scope

        with tf.variable_scope(self.relative_scope):
            self.cell = rnn.LSTMCell(size, activation=activation)
            self.belief_state = get_state_variables(batch_size, self.cell)
            # very bad dark magic, need to refactor all of this
            # supports only ine layer cell
            self.belief_out = tf.placeholder(
                tf.float32, [2, batch_size, self.cell.output_size])
            l = tf.unstack(self.belief_out, axis=0)
            rnn_tuple_state = rnn.LSTMStateTuple(l[0], l[1])
            self.belief_assign = get_state_update_op([self.belief_state], [rnn_tuple_state])

            logits, rnn_states = tf.nn.dynamic_rnn(
                self.cell, tf.expand_dims(feature_state, 1),
                sequence_length=[1] * batch_size, initial_state=self.belief_state)

            self.state = tf.squeeze(logits, 1)

            # @TODO: very hacky 2
            self.belief_update = get_state_update_op([self.belief_state], [rnn_states], self.is_end)