#!/usr/bin/python3

import gym
import os
import string
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib import rnn
from tqdm import trange, tqdm
import collections

from matplotlib import pyplot as plt

plt.style.use("ggplot")

from wrappers import PreprocessImage, EnvPool


# def copy_model_parameters(sess, source_scope, target_scope):
#     source_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, source_scope)
#     source_params = sorted(source_params, key=lambda v: v.name)
#     target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_scope)
#     target_params = sorted(target_params, key=lambda v: v.name)
#
#     update_ops = []
#     for source_var, target_var in zip(source_params, target_params):
#         update_ops.append(target_var.assign(source_var))
#
#     sess.run(update_ops)


def create_if_need(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_unimetric(history, metric, save_dir):
    plt.figure()
    plt.plot(history[metric])
    plt.title('model {}'.format(metric))
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.savefig("{}/{}.png".format(save_dir, metric),
                format='png', dpi=300)


def save_stats(stats, save_dir="./"):
    for key in stats:
        plot_unimetric(stats, key, save_dir)


activations = {
    "sigmoid": tf.sigmoid,
    "tanh": tf.tanh,
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "elu": tf.nn.elu,
    "softplus": tf.nn.softplus
}


def update_varlist(loss, optimizer, var_list, grad_clip=5.0, global_step=None):
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    capped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs]
    update_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    return update_step


def build_optimization(model, optimization_params=None):
    optimization_params = optimization_params or {}

    initial_lr = optimization_params.get("initial_lr", 1e-4)
    decay_steps = int(optimization_params.get("decay_steps", 100000))
    lr_decay = optimization_params.get("lr_decay", 0.999)

    lr = tf.train.exponential_decay(
        initial_lr,
        model.global_step,
        decay_steps,
        lr_decay,
        staircase=True)

    model.optimizer = tf.train.AdamOptimizer(lr)

    model.train_op = update_varlist(
        model.loss, model.optimizer,
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=model.scope),
        grad_clip=optimization_params.get("grad_clip", 10.0),
        global_step=model.global_step)

    return model


class FeatureNet(object):
    def __init__(self, state_shape, network, special=None):
        self.special = special or {}
        self.state_shape = state_shape

        self.states = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32, name="states")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optimizer = None
        self.train_op = None

        self.scope = self.special.get("scope", "feature_network")

        self.hidden_state = network(
            self.states,
            scope=self.scope + "/hidden",
            reuse=self.special.get("reuse_hidden", False),
            is_training=self.is_training)

        self.losses = []

    def add_loss(self, loss):
        self.losses.append(loss)

    def compute_loss(self):
        self.loss = tf.add_n(self.losses) / float(len(self.losses))


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

        self.scope = self.special.get("scope", "policy_network")

        self.predicted_probs = self._probs(
            hidden_state,
            scope=self.scope + "/probs",
            reuse=self.special.get("reuse_probs", False))

        one_hot_actions = tf.one_hot(self.actions, n_actions)
        predicted_probs_for_actions = tf.reduce_sum(
            tf.multiply(self.predicted_probs, one_hot_actions),
            axis=-1)

        J = tf.reduce_mean(tf.log(predicted_probs_for_actions) * self.cumulative_rewards)
        self.loss = -J

        # a bit of regularization
        if self.special.get("entropy_loss", True):
            H = tf.reduce_mean(
                tf.reduce_sum(
                    self.predicted_probs * tf.log(self.predicted_probs),
                    axis=-1))
            self.loss += H * self.special.get("entropy_koef", 0.01)

    def _probs(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            probs = tflayers.fully_connected(
                hidden_state,
                num_outputs=self.n_actions,
                activation_fn=tf.nn.softmax)
            return probs


class StateValueNet(object):
    def __init__(self, hidden_state, special=None):
        self.special = special or {}

        self.td_target = tf.placeholder(shape=[None], dtype=tf.float32, name="td_target")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optimizer = None
        self.train_op = None

        self.scope = self.special.get("scope", "state_network")

        self.predicted_values = tf.squeeze(
            self._state_value(
                hidden_state,
                scope=self.scope + "/state_value",
                reuse=self.special.get("reuse_state_value", False)),
            axis=1)

        self.loss = tf.losses.mean_squared_error(
            labels=self.td_target,
            predictions=self.predicted_values)

    def _state_value(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            qvalues = tflayers.fully_connected(
                hidden_state,
                num_outputs=1,
                activation_fn=None)
            return qvalues


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

        self.scope = self.special.get("scope", "qvalue_network")

        self.predicted_qvalues = self._qvalues(
            hidden_state,
            scope=self.scope + "/qvalue",
            reuse=self.special.get("reuse_state_value", False))

        one_hot_actions = tf.one_hot(self.actions, n_actions)
        self.predicted_qvalues_for_actions = tf.reduce_sum(
            tf.multiply(self.predicted_qvalues, one_hot_actions),
            axis=-1)

        self.loss = tf.losses.mean_squared_error(
            labels=self.td_target,
            predictions=self.predicted_qvalues_for_actions)

    def _qvalues(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            qvalues = tflayers.fully_connected(
                hidden_state,
                num_outputs=self.n_actions,
                activation_fn=None)
            if self.special.get("advantage", False):
                qvalues -= tf.reduce_mean(qvalues, axis=-1, keep_dims=True)
            return qvalues


def get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    zero_states = cell.zero_state(batch_size, tf.float32)
    if isinstance(zero_states, list):
        state_variables = []
        for state_c, state_h in zero_states:
            state_variables.append(
                rnn.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False),
                    tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)
    elif isinstance(zero_states, tuple):
        state_c, state_h = zero_states
        return rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False))


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
        feature_net = FeatureNet(self.state_shape, network, self.special.get("feature_net", None))

        n_games = self.special["n_games"]
        with tf.variable_scope("belief_state"):
            self.belief_state = get_state_variables(n_games, cell)

        logits, rnn_states = tf.nn.dynamic_rnn(
            cell, tf.expand_dims(feature_net.hidden_state, 1),
            sequence_length=[1] * n_games, initial_state=self.belief_state)

        self.logits = tf.squeeze(logits, 1)

        # @TODO: very hacky 2
        self.belief_update = get_state_update_op([self.belief_state], [rnn_states], self.is_end)

        state_value_net = StateValueNet(self.logits, self.special.get("state_value_net", None))
        self.state_value_net = build_optimization(
            state_value_net,
            self.special.get("state_value_net_optimization", None))
        feature_net.add_loss(self.state_value_net.loss)

        policy_net = PolicyNet(self.logits, self.n_actions, self.special.get("policy_net", None))
        self.policy_net = build_optimization(
            policy_net,
            self.special.get("policy_net_optimization", None))
        feature_net.add_loss(self.policy_net.loss)

        feature_net.compute_loss()
        self.feature_net = build_optimization(
            feature_net, self.special.get("feature_net_optimization", None))

    def predict_value(self, sess, state_batch):
        return sess.run(
            self.state_value_net.predicted_values,
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


def epsilon_greedy_policy(agent, sess, observations):
    A = agent.predict_action(sess, observations)
    actions = [np.random.choice(len(row), p=row) for row in A]
    return actions


def update_on_batch(
        sess, dqrn_agent,
        state_batch, action_batch, reward_batch, next_state_batch, done_batch,
        discount_factor=0.99, reward_norm=1.0):
    values = dqrn_agent.predict_value(sess, state_batch)
    values_next = dqrn_agent.predict_value(sess, next_state_batch)

    td_target = reward_batch * reward_norm + \
                np.invert(done_batch).astype(np.float32) * \
                discount_factor * values_next
    td_error = td_target - values

    loss_policy, loss_state, _, _, _ = sess.run(
        [dqrn_agent.policy_net.loss,
         dqrn_agent.state_value_net.loss,
         dqrn_agent.policy_net.train_op,
         dqrn_agent.state_value_net.train_op,
         dqrn_agent.feature_net.train_op],
        feed_dict={
            dqrn_agent.feature_net.states: state_batch,
            dqrn_agent.feature_net.is_training: True,
            dqrn_agent.policy_net.actions: action_batch,
            dqrn_agent.policy_net.cumulative_rewards: td_error,
            dqrn_agent.policy_net.is_training: True,
            dqrn_agent.state_value_net.td_target: td_target,
            dqrn_agent.state_value_net.is_training: True,
        })

    return loss_policy, loss_state


def update_wraper(discount_factor=0.99, reward_norm=1.0):
    def wrapper(sess, q_net, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        return update_on_batch(
            sess, q_net,
            state_batch, action_batch, reward_batch, next_state_batch, done_batch,
            discount_factor=discount_factor, reward_norm=reward_norm)

    return wrapper


def generate_sessions(sess, dqrn_agent, env_pool, t_max=1000, update_fn=None):
    total_reward = 0.0
    total_policy_loss, total_state_loss = 0, 0
    total_games = 0.0

    states = env_pool.pool_states()
    for t in range(t_max):
        actions = epsilon_greedy_policy(dqrn_agent, sess, states)
        new_states, rewards, dones, _ = env_pool.step(actions)

        if update_fn is not None:
            curr_policy_loss, curr_state_loss = update_fn(
                sess, dqrn_agent,
                states, actions, rewards, new_states, dones)
            total_policy_loss += curr_policy_loss
            total_state_loss += curr_state_loss

        dqrn_agent.update_belief_state(sess, states, dones)
        states = new_states

        total_reward += rewards.mean()
        total_games += dones.sum()

    return total_reward, total_policy_loss, total_state_loss, total_games


def generate_session(
        sess, dqrn_agent, env, t_max=1000, update_fn=None):
    total_reward = 0
    total_policy_loss, total_state_loss = 0, 0
    s = env.reset()

    for t in range(t_max):
        a = epsilon_greedy_policy(dqrn_agent, sess, np.array([s], dtype=np.float32))[0]

        new_s, r, done, _ = env.step(a)

        if update_fn is not None:
            curr_policy_loss, curr_state_loss = update_fn(
                sess, dqrn_agent,
                [s], [a], [r], [new_s], [done])
            total_policy_loss += curr_policy_loss
            total_state_loss += curr_state_loss

        total_reward += r

        dqrn_agent.update_belief_state(sess, [s], [done])
        s = new_s

        if done:
            break

    return total_reward, total_policy_loss,  total_state_loss, t


def a3c_lstm_learning(
        sess, q_net, env, update_fn,
        n_epochs=1000, n_sessions=100, t_max=1000):
    tr = trange(
        n_epochs,
        desc="",
        leave=True)

    moi = ["policy_loss", "state_loss", "reward", "steps"]
    history = {metric: np.zeros(n_epochs) for metric in moi}

    for i in tr:
        sessions = [
            generate_sessions(sess, q_net, env, t_max, update_fn=update_fn)
            for _ in range(n_sessions)]
        session_rewards, session_policy_loss, session_state_loss, session_steps = \
            map(np.array, zip(*sessions))

        history["reward"][i] = np.mean(session_rewards)
        history["policy_loss"][i] = np.mean(session_policy_loss)
        history["state_loss"][i] = np.mean(session_state_loss)
        history["steps"][i] = np.mean(session_steps)

        desc = "\t".join(
            ["{} = {:.3f}".format(key, value[i]) for key, value in history.items()])
        tr.set_description(desc)

    return history


def conv_network(
        states,
        scope, reuse=False,
        is_training=True, activation_fn=tf.nn.elu):
    with tf.variable_scope(scope or "network", reuse=reuse):
        conv = tflayers.conv2d(
            states,
            num_outputs=32,
            kernel_size=8,
            stride=4,
            activation_fn=activation_fn)
        conv = tflayers.conv2d(
            conv,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=activation_fn)
        conv = tflayers.conv2d(
            conv,
            num_outputs=64,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn)

        flat = tflayers.flatten(conv)
        return flat


def network_wrapper(activation_fn=tf.nn.elu):
    def wrapper(states, scope=None, reuse=False, is_training=True):
        return conv_network(states, scope, reuse, is_training, activation_fn=activation_fn)

    return wrapper


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument(
        '--env',
        type=str,
        default='KungFuMasterDeterministic-v0',  # BreakoutDeterministic-v0
        help='The environment to use')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--n_sessions',
        type=int,
        default=10)
    parser.add_argument(
        '--t_max',
        type=int,
        default=1000)
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Gamma discount factor')
    parser.add_argument(
        '--plot_stats',
        action='store_true',
        default=False)
    parser.add_argument(
        '--api_key',
        type=str,
        default=None)
    parser.add_argument(
        '--activation',
        type=str,
        default="elu")
    parser.add_argument(
        '--load',
        action='store_true',
        default=False)
    parser.add_argument(
        '--gpu_option',
        type=float,
        default=0.45)

    parser.add_argument(
        '--initial_lr',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--lr_decay_steps',
        type=float,
        default=1e5)
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.999)
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=10.0)

    parser.add_argument(
        '--n_games',
        type=int,
        default=10)
    parser.add_argument(
        '--lstm_activation',
        type=str,
        default="tanh")

    parser.add_argument(
        '--reward_norm',
        type=float,
        default=1.0)

    args, _ = parser.parse_known_args()
    return args


def run(env, q_learning_args, update_args, agent_args,
        n_games, lstm_activation="tanh",
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4):
    env_name = env
    env = EnvPool(
        PreprocessImage(
            gym.make(env_name).env,
            width=64, height=64, grayscale=True,
            crop=lambda img: img[60:-30, 7:]), n_games)

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    network = agent_args.get("network", None) or conv_network
    cell = rnn.LSTMCell(512, activation=activations[lstm_activation])
    q_net = A3CLstmAgent(
        state_shape, n_actions, network, cell=cell,
        special=agent_args)

    model_dir = "./logs_" + env_name.replace(string.punctuation, "_")
    model_dir += "_{}".format(lstm_activation)
    create_if_need(model_dir)

    # @TODO: very very hintly, need to find best solution
    # vars_of_interest = [
    #     v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #     if not v.name.startswith("belief_state")]
    vars_of_interest = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)
    saver = tf.train.Saver(var_list=vars_of_interest)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if not load:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "{}/model.ckpt".format(model_dir))

        stats = a3c_lstm_learning(
            sess, q_net, env,
            update_fn=update_wraper(**update_args),
            **q_learning_args)

        saver.save(
            sess, "{}/model.ckpt".format(model_dir),
            meta_graph_suffix='meta', write_meta_graph=True)
        # tf.train.write_graph(sess.graph_def, model_dir, "graph.pb", False)

    if plot_stats:
        stats_dir = os.path.join(model_dir, "stats")
        create_if_need(stats_dir)
        save_stats(stats, save_dir=stats_dir)

    if api_key is not None:
        tf.reset_default_graph()

        agent_args["n_games"] = 1
        q_net = A3CLstmAgent(
            state_shape, n_actions, network, cell=cell,
            special=agent_args)
        vars_of_interest = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(var_list=vars_of_interest)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.variables_initializer(
                [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                 if v.name.startswith("belief_state")]))
            # sess.run(tf.global_variables_initializer())
            saver.restore(sess, "{}/model.ckpt".format(model_dir))

            env_name = env_name.replace("Deterministic", "")
            env = PreprocessImage(
                gym.make(env_name),
                width=84, height=84, grayscale=True,
                crop=lambda img: img[60:-30, 7:])
            env = gym.wrappers.Monitor(env, "{}/monitor".format(model_dir), force=True)
            sessions = [generate_session(sess, q_net, env, int(1e10), update_fn=None)
                        for _ in range(300)]
            env.close()
            gym.upload("{}/monitor".format(model_dir), api_key=api_key)


def main():
    args = _parse_args()
    network = network_wrapper(activations[args.activation])
    q_learning_args = {
        "n_epochs": args.n_epochs,
        "n_sessions": args.n_sessions,
        "t_max": args.t_max
    }
    update_args = {
        "discount_factor": args.gamma,
        "reward_norm": args.reward_norm,
    }
    optimization_params = {
        "initial_lr": args.initial_lr,
        "decay_steps": args.lr_decay_steps,
        "lr_decay": args.lr_decay,
        "grad_clip": args.grad_clip
    }
    agent_args = {
        "n_games": args.n_games,
        "network": network,
        "feature_net_optimization": optimization_params,
        "state_value_net_optimiaztion": optimization_params,
        "policy_net_optimiaztion": optimization_params
    }
    run(args.env, q_learning_args, update_args, agent_args,
        args.n_games, args.lstm_activation,
        args.plot_stats, args.api_key,
        args.load, args.gpu_option)


if __name__ == '__main__':
    main()
