#!/usr/bin/python3

import gym
import ppaquette_gym_doom
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
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
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      net1: Estimator to copy the paramters from
      net2: Estimator to copy the parameters to
    """
    net1_params = [t for t in tf.trainable_variables() if t.name.startswith(net1.scope)]
    net1_params = sorted(net1_params, key=lambda v: v.name)
    net2_params = [t for t in tf.trainable_variables() if t.name.startswith(net2.scope)]
    net2_params = sorted(net2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(net1_params, net2_params):
        op = e2_v.assign(e1_v)
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
    def __init__(self, state_shape, n_actions, network, update_fn=None, special=None):
        self.special = special or {}
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.buffer = collections.deque(maxlen=self.special.get("buffer_len", 10000))
        self.batch_size = self.special.get("batch_size", 32)
        self.update_fn = update_fn

        self.states = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.td_targets = tf.placeholder(shape=[None], dtype=tf.float32)

        self.scope = self.special.get("scope", "network")

        self.predicted_qvalues = self.qnetwork(
            network,
            self.states,
            scope=self.scope)

        one_hot_actions = tf.one_hot(self.actions, n_actions)
        predicted_qvalues_for_actions = tf.reduce_sum(
            tf.multiply(self.predicted_qvalues, one_hot_actions),
            axis=-1)

        self.loss = tf.reduce_mean(
            tf.square(self.td_targets - predicted_qvalues_for_actions))

        self.update_step = tf.train.AdamOptimizer(self.special.get("lr", 1e-4)).minimize(
            self.loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope))

    def qnetwork(self, network, state, scope, reuse=False):
        hidden_state = network(state, scope=scope + "_hidden", reuse=reuse)
        qvalues = self._to_qvalues(hidden_state, scope=scope + "_qvalues", reuse=reuse)
        return qvalues

    def _to_qvalues(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()
            qvalues = tflayers.fully_connected(
                hidden_state,
                num_outputs=self.n_actions,
                activation_fn=None)
            return qvalues

    def predict(self, sess, state_batch):
        return sess.run(
            self.predicted_qvalues,
            feed_dict={
                self.states: state_batch})

    def observe(self, sess, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done))

        if len(self.buffer) > self.batch_size:
            return self.update_fn(
                sess=sess,
                q_net=self,
                buffer=self.buffer,
                batch_size=self.batch_size)

        return 0.0


def e_greedy_action(agent, sess, state, epsilon=0.0):
    if np.random.rand() < epsilon:
        action = np.random.choice(agent.n_actions)
    else:
        qvalues = agent.predict(sess, np.array([state], dtype=np.float32))
        action = np.argmax(qvalues)
    return action


def update(sess, q_net, target_net, buffer, discount_factor=0.99, batch_size=32):
    batch_ids = np.random.choice(len(buffer), batch_size)
    batch = np.array([buffer[i] for i in batch_ids])

    # state_batch = np.vstack(batch[:, 0]).reshape((-1,) + q_net.state_shape)
    # state_next_batch = np.vstack(batch[:, 1]).reshape((-1,) + q_net.state_shape)
    # reward_batch = np.vstack(batch[:, 2]).reshape(-1)
    # done_batch = np.vstack(batch[:, 3]).reshape(-1)

    state_batch = np.vstack(batch[:, 0]).reshape((-1,) + q_net.state_shape)
    # action_batch = np.vstack(batch[:, 1]).reshape(-1)
    reward_batch = np.vstack(batch[:, 2]).reshape(-1)
    next_state_batch = np.vstack(batch[:, 3]).reshape((-1,) + q_net.state_shape)
    done_batch = np.vstack(batch[:, 4]).reshape(-1)

    qvalues = q_net.predict(sess, state_batch)
    best_actions = qvalues.argmax(axis=1)
    qvalues_next = target_net.predict(sess, next_state_batch)
    td_target_batch = reward_batch + \
                      np.invert(done_batch).astype(np.float32) * \
                      discount_factor * qvalues_next[np.arange(batch_size), best_actions]

    q_loss, _ = sess.run(
        [q_net.loss, q_net.update_step],
        feed_dict={
            q_net.states: state_batch,
            q_net.actions: best_actions,
            q_net.td_targets: td_target_batch})

    return q_loss


def observe2update(params):
    def wrapper(sess, q_net, buffer, batch_size):
        return update(sess=sess, q_net=q_net, buffer=buffer, batch_size=batch_size, **params)

    return wrapper


def generate_session(sess, agent, env, epsilon=0.5, t_max=1000):
    """play env with approximate q-learning agent and train it at the same time"""

    total_reward = 0
    s = env.reset()
    total_loss = 0

    for t in range(t_max):
        a = e_greedy_action(agent, sess, s, epsilon)

        new_s, r, done, info = env.step(a)

        curr_loss = agent.observe(sess, s, a, r, new_s, done)

        total_reward += r
        total_loss += curr_loss

        s = new_s
        if done:
            break

    return total_reward, total_loss / float(t + 1), t


def q_learning(
        sess, agent, frozen_agent, env, n_epochs, n_epochs_skip=10,
        n_sessions=100, t_max=1000,
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
        sessions = [generate_session(sess, agent, env, epsilon, t_max) for _ in range(n_sessions)]
        session_rewards, session_loss, session_steps = map(np.array, zip(*sessions))

        if i < n_epochs_decay:
            epsilon -= (initial_epsilon - final_epsilon) / float(n_epochs_decay)

        if (i + 1) % n_epochs_skip:
            copy_model_parameters(sess, agent, frozen_agent)

        history["reward"][i] = np.mean(session_rewards)
        history["epsilon"][i] = epsilon
        history["loss"][i] = np.mean(session_loss)
        history["steps"][i] = np.mean(session_steps)

        tr.set_description(
            "mean reward = {:.3f}\tepsilon = {:.3f}\tloss = {:.3f}\tsteps = {:.3f}".format(
                history["reward"][i], history["epsilon"][i], history["loss"][i],
                history["steps"][i]))

    return history


def conv_network(states, scope, reuse=False, activation_fn=tf.nn.elu):
    with tf.variable_scope(scope or "network") as scope:
        if reuse:
            scope.reuse_variables()

        conv1 = tflayers.conv2d(
            states, 32, [5, 5], stride=1, padding='SAME', activation_fn=activation_fn)
        conv1 = tflayers.conv2d(
            conv1, 32, [5, 5], stride=2, padding='VALID', activation_fn=activation_fn)
        pool1 = tflayers.max_pool2d(conv1, [3, 3], padding='VALID')
        # pool1 = tflayers.dropout(pool1, keep_prob=keep_prob, is_training=is_training)

        conv2 = tflayers.conv2d(
            pool1, 32, [5, 5], stride=1, padding='SAME', activation_fn=activation_fn)
        conv2 = tflayers.conv2d(
            conv2, 32, [5, 5], stride=2, padding='VALID', activation_fn=activation_fn)
        pool2 = tflayers.max_pool2d(conv2, [3, 3], padding='VALID')
        # pool2 = tflayers.dropout(pool2, keep_prob=keep_prob, is_training=is_training)

        flat = tflayers.flatten(pool2)

        logits = tflayers.fully_connected(
            flat,
            int(flat.get_shape().as_list()[-1]/2),
            activation_fn=activation_fn)
        return logits


def network_wrapper(activation_fn=tf.tanh):
    def wrapper(states, scope=None, reuse=False):
        return conv_network(states, scope, reuse, activation_fn=activation_fn)

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
                        default=0.99,
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

    args, _ = parser.parse_known_args()
    return args


def run(env, n_epochs, n_epochs_skip, discount_factor,
        plot_stats=False, api_key=None,
        network=None, batch_size=64, buffer_len=100000, initial_epsilon=0.25,
        load=False, gpu_option=0.4, initial_lr=1e-4):
    env_name = env
    height, width = 64, 64
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
    special = {
        "batch_size": batch_size,
        "buffer_len": buffer_len,
        "lr": initial_lr,
        "scope": "q_net"
    }

    network = network or conv_network
    frozen_agent = DqnAgent(
        state_shape, n_actions, network,
        update_fn=None,
        special={"scope": "frozen_q_net"})
    agent = DqnAgent(
        state_shape, n_actions, network,
        update_fn=observe2update({
            "target_net": frozen_agent,
            "discount_factor": discount_factor}),
        special=special)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.Saver()
        model_dir = "./logs_" + env_name.replace(string.punctuation, "_")
        if not load:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "{}/model.ckpt".format(model_dir))

        stats = q_learning(
            sess, agent, frozen_agent, env, n_epochs,
            n_epochs_skip=n_epochs_skip,
            initial_epsilon=initial_epsilon)
        create_if_need(model_dir)
        saver.save(sess, "{}/model.ckpt".format(model_dir))

        if plot_stats:
            save_stats(stats)

        if api_key is not None:
            env = gym.wrappers.Monitor(env, "{}/monitor".format(model_dir), force=True)
            sessions = [generate_session(sess, agent, env, 0.0, int(1e10)) for _ in range(300)]
            env.close()
            gym.upload("{}/monitor".format(model_dir), api_key=api_key)


def main():
    args = _parse_args()
    network = network_wrapper(activations[args.activation])
    run(args.env, args.n_epochs, args.n_epochs_skip, args.gamma,
        args.plot_stats, args.api_key,
        network, args.batch_size, args.buffer_len, args.initial_epsilon,
        args.load, args.gpu_option, args.initial_lr)


if __name__ == '__main__':
    main()
