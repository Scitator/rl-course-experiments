#!/usr/bin/python3

import gym
from gym.wrappers import SkipWrapper
import os
import string
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
from tqdm import trange
import collections

from matplotlib import pyplot as plt

plt.style.use("ggplot")

from wrappers import PreprocessImage, FrameBuffer


def copy_model_parameters(sess, net1, net2):
    """
    Copies the model parameters of one net to another.

    Args:
      sess: Tensorflow session instance
      net1: net to copy the parameters from
      net2: net to copy the parameters to
    """
    net1_params = [t for t in tf.trainable_variables() if t.name.startswith(net1.scope)]
    net1_params = sorted(net1_params, key=lambda v: v.name)
    net2_params = [t for t in tf.trainable_variables() if t.name.startswith(net2.scope)]
    net2_params = sorted(net2_params, key=lambda v: v.name)

    update_ops = []
    for net1_v, net2_v in zip(net1_params, net2_params):
        op = net2_v.assign(net1_v)
        update_ops.append(op)

    sess.run(update_ops)


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


class DqnAgent(object):
    def __init__(self, state_shape, n_actions, network,special=None):
        self.special = special or {}
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.buffer = collections.deque(maxlen=self.special.get("buffer_len", 10000))

        self.states = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.td_targets = tf.placeholder(shape=[None], dtype=tf.float32)

        self.is_training = tf.placeholder(dtype=tf.bool)

        self.scope = self.special.get("scope", "network")

        self.predicted_qvalues = self.qnetwork(
            network,
            self.states,
            scope=self.scope,
            is_training=self.is_training)

        one_hot_actions = tf.one_hot(self.actions, n_actions)
        predicted_qvalues_for_actions = tf.reduce_sum(
            tf.multiply(self.predicted_qvalues, one_hot_actions),
            axis=-1)

        self.loss = tf.losses.mean_squared_error(
            labels=self.td_targets,
            predictions=predicted_qvalues_for_actions)

        self.update_step = tf.train.AdamOptimizer(self.special.get("lr", 1e-4)).minimize(
            self.loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope))

    def qnetwork(self, network, state, scope, reuse=False, is_training=True):
        hidden_state = network(state, scope=scope + "_hidden", reuse=reuse, is_training=is_training)
        qvalues = self._qvalues(hidden_state, scope=scope + "_qvalues", reuse=reuse)

        if self.special.get("dueling_network", False):
            state_value = self._state_value(
                hidden_state, scope=scope + "_state_values", reuse=reuse)
            qvalues -= tf.reduce_mean(qvalues, axis=-1, keep_dims=True)
            qvalues += state_value

        return qvalues

    def _qvalues(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()
            qvalues = tflayers.fully_connected(
                hidden_state,
                num_outputs=self.n_actions,
                activation_fn=None)
            return qvalues

    def _state_value(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()
            qvalues = tflayers.fully_connected(
                hidden_state,
                num_outputs=1,
                activation_fn=None)
            return qvalues

    def predict(self, sess, state_batch, is_training=False):
        return sess.run(
            self.predicted_qvalues,
            feed_dict={
                self.states: state_batch,
                self.is_training: is_training})

    def update(self, sess, state_batch, action_batch, td_target_batch, is_training=True):
        q_loss, _ = sess.run(
            [self.loss, self.update_step],
            feed_dict={
                self.states: state_batch,
                self.actions: action_batch,
                self.td_targets: td_target_batch,
                self.is_training: is_training})
        return q_loss

    def observe(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done))


def e_greedy_action(agent, sess, state, epsilon=0.0):
    if np.random.rand() < epsilon:
        action = np.random.choice(agent.n_actions)
    else:
        qvalues = agent.predict(sess, np.array([state], dtype=np.float32))
        action = np.argmax(qvalues)
    return action


def update(sess, q_net, target_net, discount_factor=0.99, batch_size=32):
    q_loss = 0.0

    if len(q_net.buffer) > batch_size:
        batch_ids = np.random.choice(len(q_net.buffer), batch_size)
        batch = np.array([q_net.buffer[i] for i in batch_ids])

        state_batch = np.vstack(batch[:, 0]).reshape((-1,) + q_net.state_shape)
        action_batch = np.vstack(batch[:, 1]).reshape(-1)
        reward_batch = np.vstack(batch[:, 2]).reshape(-1)
        next_state_batch = np.vstack(batch[:, 3]).reshape((-1,) + q_net.state_shape)
        done_batch = np.vstack(batch[:, 4]).reshape(-1)

        qvalues = q_net.predict(sess, state_batch)
        best_actions = qvalues.argmax(axis=1)
        qvalues_next = target_net.predict(sess, next_state_batch)
        td_target_batch = reward_batch + \
            np.invert(done_batch).astype(np.float32) * \
            discount_factor * qvalues_next[np.arange(batch_size), best_actions]

        q_loss = q_net.update(sess, state_batch, action_batch, td_target_batch)

    return q_loss


def update_wraper(discount_factor=0.99, batch_size=32):
    def wrapper(sess, q_net, target_net):
        return update(sess, q_net, target_net, discount_factor, batch_size)
    return wrapper


def generate_session(sess, q_net, target_net, env, epsilon=0.5, t_max=1000, update_fn=None):
    """play env with approximate q-learning agent and train it at the same time"""

    total_reward = 0
    s = env.reset()
    total_loss = 0

    for t in range(t_max):
        a = e_greedy_action(q_net, sess, s, epsilon)

        new_s, r, done, info = env.step(a)

        if update_fn is not None:
            q_net.observe(s, a, r, new_s, done)
            curr_loss = update_fn(sess, q_net, target_net)
            total_loss += curr_loss

        total_reward += r

        s = new_s
        if done:
            break

    return total_reward, total_loss / float(t + 1), t


def q_learning(
        sess, q_net, target_net, env, update_fn,
        n_epochs=1000, n_epochs_skip=10, n_sessions=100, t_max=1000,
        initial_epsilon=0.25, final_epsilon=0.01):
    tr = trange(
        n_epochs,
        desc="mean reward = {:.3f}\tepsilon = {:.3f}\tloss = {:.3f}\tsteps = {:.3f}".format(
            0.0, 0.0, 0.0, 0.0),
        leave=True)

    epsilon = initial_epsilon
    n_epochs_decay = n_epochs * 0.8

    history = {
        "reward": np.zeros(n_epochs),
        "epsilon": np.zeros(n_epochs),
        "loss": np.zeros(n_epochs),
        "steps": np.zeros(n_epochs),
    }

    for i in tr:
        sessions = [
            generate_session(sess, q_net, target_net, env, epsilon, t_max, update_fn=update_fn)
            for _ in range(n_sessions)]
        session_rewards, session_loss, session_steps = map(np.array, zip(*sessions))

        if i < n_epochs_decay:
            epsilon -= (initial_epsilon - final_epsilon) / float(n_epochs_decay)

        if (i + 1) % n_epochs_skip == 0:
            copy_model_parameters(sess, q_net, target_net)

        history["reward"][i] = np.mean(session_rewards)
        history["epsilon"][i] = epsilon
        history["loss"][i] = np.mean(session_loss)
        history["steps"][i] = np.mean(session_steps)

        tr.set_description(
            "mean reward = {:.3f}\tepsilon = {:.3f}\tloss = {:.3f}\tsteps = {:.3f}".format(
                history["reward"][i], history["epsilon"][i], history["loss"][i],
                history["steps"][i]))

    return history


def conv_network(states, scope, reuse=False, is_training=True, activation_fn=tf.nn.elu):
    with tf.variable_scope(scope or "network") as scope:
        if reuse:
            scope.reuse_variables()

        conv1 = tflayers.conv2d(
            states,
            32, [5, 5], stride=3, padding='SAME',
            normalizer_fn=tflayers.batch_norm,
            normalizer_params={"is_training": is_training},
            activation_fn=activation_fn)
        conv1 = tflayers.conv2d(
            conv1,
            32, [5, 5], stride=3, padding='VALID',
            normalizer_fn=tflayers.batch_norm,
            normalizer_params={"is_training": is_training},
            activation_fn=activation_fn)
        pool1 = tflayers.max_pool2d(conv1, [3, 3], padding='VALID')
        # pool1 = tflayers.dropout(pool1, keep_prob=keep_prob, is_training=is_training)

        # conv2 = tflayers.conv2d(
        #     pool1,
        #     32, [5, 5], stride=2, padding='SAME',
        #     normalizer_fn=tflayers.batch_norm,
        #     normalizer_params={"is_training": is_training},
        #     activation_fn=activation_fn)
        # conv2 = tflayers.conv2d(
        #     conv2,
        #     32, [5, 5], stride=1, padding='VALID',
        #     normalizer_fn=tflayers.batch_norm,
        #     normalizer_params={"is_training": is_training},
        #     activation_fn=activation_fn)
        # pool2 = tflayers.max_pool2d(conv2, [3, 3], padding='VALID')
        # pool2 = tflayers.dropout(pool2, keep_prob=keep_prob, is_training=is_training)

        flat = tflayers.flatten(pool1)
        logits = tflayers.fully_connected(
            flat,
            256,
            normalizer_fn=tflayers.batch_norm,
            normalizer_params={"is_training": is_training},
            activation_fn=activation_fn)
        return logits


def network_wrapper(activation_fn=tf.nn.elu):
    def wrapper(states, scope=None, reuse=False, is_training=True):
        return conv_network(states, scope, reuse, is_training, activation_fn=activation_fn)
    return wrapper


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument('--env',
                        type=str,
                        default='KungFuMaster-v0',  # BreakoutDeterministic-v0
                        help='The environment to use')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=1000)
    parser.add_argument('--n_sessions',
                        type=int,
                        default=100)
    parser.add_argument('--t_max',
                        type=int,
                        default=1000)
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='Gamma discount factor')
    parser.add_argument('--plot_stats',
                        action='store_true',
                        default=False)
    parser.add_argument('--api_key',
                        type=str,
                        default=None)
    parser.add_argument('--activation',
                        type=str,
                        default="elu")
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--buffer_len',
                        type=int,
                        default=100000)
    parser.add_argument('--initial_epsilon',
                        type=float,
                        default=0.25,
                        help='Gamma discount factor')
    parser.add_argument('--load',
                        action='store_true',
                        default=False)
    parser.add_argument('--gpu_option',
                        type=float,
                        default=0.45)
    parser.add_argument('--initial_lr',
                        type=float,
                        default=1e-4)
    parser.add_argument('--n_epochs_skip',
                        type=int,
                        default=1)
    parser.add_argument('--dueling_network',
                        action='store_true',
                        default=False)

    args, _ = parser.parse_known_args()
    return args


def run(env, q_learning_args, update_args,
        initial_lr=1e-4, dueling_network=False,
        network=None, buffer_len=100000,
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4):
    env_name = env
    height, width = 84, 84
    n_frames = 4
    make_env = lambda: FrameBuffer(
        PreprocessImage(
            SkipWrapper(4)(gym.make(env_name)),
            width=width, height=height, grayscale=True,
            crop=lambda img: img[20:-10, 8:-8]),
        n_frames=n_frames,
        reshape_fn=lambda x: np.transpose(x, [1, 2, 3, 0]).reshape(height, width, n_frames))
    env = make_env()

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    network = network or conv_network

    q_net = DqnAgent(
        state_shape, n_actions, network,
        special={
            "buffer_len": buffer_len,
            "lr": initial_lr,
            "scope": "q_net",
            "dueling_network": dueling_network})

    target_net = DqnAgent(
        state_shape, n_actions, network,
        special={
            "buffer_len": 0,
            "lr": initial_lr,
            "scope": "frozen_q_net",
            "dueling_network": dueling_network})

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.Saver()
        model_dir = "./logs_" + env_name.replace(string.punctuation, "_")
        if dueling_network:
            model_dir += "_dueling_network"
        if not load:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "{}/model.ckpt".format(model_dir))

        stats = q_learning(
            sess, q_net, target_net, env,
            update_fn=update_wraper(**update_args),
            **q_learning_args)
        create_if_need(model_dir)
        saver.save(sess, "{}/model.ckpt".format(model_dir))

        if plot_stats:
            stats_dir = os.path.join(model_dir, "stats")
            create_if_need(stats_dir)
            save_stats(stats, save_dir=stats_dir)

        if api_key is not None:
            env = gym.wrappers.Monitor(env, "{}/monitor".format(model_dir), force=True)
            sessions = [generate_session(sess, q_net, target_net, env, 0.0, int(1e10),
                                         update_fn=None)
                        for _ in range(300)]
            env.close()
            gym.upload("{}/monitor".format(model_dir), api_key=api_key)


def main():
    args = _parse_args()
    network = network_wrapper(activations[args.activation])
    q_learning_args = {
        "n_epochs": args.n_epochs,
        "n_epochs_skip": args.n_epochs_skip,
        "n_sessions": args.n_sessions,
        "t_max": args.t_max,
        "initial_epsilon": args.initial_epsilon,
    }
    update_args = {
        "discount_factor": args.gamma,
        "batch_size": args.batch_size,
    }
    run(args.env, q_learning_args, update_args,
        args.initial_lr, args.dueling_network,
        network, args.buffer_len,
        args.plot_stats, args.api_key,
        args.load, args.gpu_option)


if __name__ == '__main__':
    main()
