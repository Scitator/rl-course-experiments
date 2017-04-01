#!/usr/bin/python3

import gym
from gym.wrappers import SkipWrapper
import os
import string
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
from tqdm import trange, tqdm
import collections

from matplotlib import pyplot as plt

plt.style.use("ggplot")



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


class ReinforceAgent(object):
    def __init__(self, state_shape, n_actions, network, special=None):
        self.special = special or {}
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.buffer = collections.deque(maxlen=self.special.get("buffer_len", 10000))

        self.states = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.cumulative_rewards = tf.placeholder(shape=[None], dtype=tf.float32)

        self.is_training = tf.placeholder(dtype=tf.bool)

        self.scope = self.special.get("scope", "network")

        self.predicted_probs = self.probs_network(
            network,
            self.states,
            scope=self.scope,
            is_training=self.is_training)

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
            self.loss += H * 0.001

        self.update_step = tf.train.AdamOptimizer(self.special.get("lr", 1e-3)).minimize(
            self.loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope))

    def probs_network(self, network, state, scope, reuse=False, is_training=True):
        hidden_state = network(state, scope=scope + "_hidden", reuse=reuse, is_training=is_training)
        qvalues = self._probs(hidden_state, scope=scope + "_probs", reuse=reuse)
        return qvalues

    def _probs(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()
            probs = tflayers.fully_connected(
                hidden_state,
                num_outputs=self.n_actions,
                activation_fn=tf.nn.softmax)
            return probs

    def predict(self, sess, state_batch, is_training=False):
        return sess.run(
            self.predicted_probs,
            feed_dict={
                self.states: state_batch,
                self.is_training: is_training})

    def update(self, sess, state_batch, action_batch, rewards_batch, is_training=True):
        q_loss, _ = sess.run(
            [self.loss, self.update_step],
            feed_dict={
                self.states: state_batch,
                self.actions: action_batch,
                self.cumulative_rewards: rewards_batch,
                self.is_training: is_training})
        return q_loss

    def observe(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done))


def action(agent, sess, state):
    probs = agent.predict(sess, state)
    actions = [np.random.choice(len(row), p=row) for row in probs]
    return actions


def get_cumulative_rewards(rewards, gamma=0.99):
    """
    take a list of immediate rewards r(s,a) for the whole session
    compute cumulative rewards R(s,a) (a.k.a. G(s,a) in Sutton '16)
    R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    The simple way to compute cumulative rewards is to iterate from last to first time tick
    and compute R_t = r_t + gamma*R_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    R = [0]
    for i in range(len(rewards)):
        R.append(rewards[-i - 1] + gamma * R[-1])
    return R[::-1][:-1]


def update(sess, q_net, states, actions, rewards, discount_factor=0.99):
    cumulative_rewards = get_cumulative_rewards(rewards, discount_factor)
    loss = q_net.update(sess, states, actions, cumulative_rewards)

    return loss


def update_wraper(discount_factor=0.99):
    def wrapper(sess, q_net, states, actions, rewards):
        return update(sess, q_net, states, actions, rewards, discount_factor)

    return wrapper


def generate_session(sess, agent, env, t_max=1000, update_fn=None):
    """play env with REINFORCE agent and train at the session end"""

    # arrays to record session
    states, actions, rewards = [], [], []
    total_loss = 0.0

    s = env.reset()

    for t in range(t_max):

        # action probabilities array aka pi(a|s)
        a = action(agent, sess, [s])[0]

        new_s, r, done, info = env.step(a)

        # record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done:
            break

    if update_fn is not None:
        total_loss += update_fn(sess, agent, states, actions, rewards)

    return sum(rewards), total_loss, t


def reinforce_learning(
        sess, q_net, env, update_fn,
        n_epochs=1000, n_sessions=100, t_max=1000):
    tr = trange(
        n_epochs,
        desc="mean reward = {:.3f}\tloss = {:.3f}\tsteps = {:.3f}".format(
            0.0, 0.0, 0.0, 0.0),
        leave=True)

    history = {
        "reward": np.zeros(n_epochs),
        "loss": np.zeros(n_epochs),
        "steps": np.zeros(n_epochs),
    }

    for i in tr:
        sessions = [
            generate_session(sess, q_net, env, t_max, update_fn=update_fn)
            for _ in range(n_sessions)]
        session_rewards, session_loss, session_steps = map(np.array, zip(*sessions))

        history["reward"][i] = np.mean(session_rewards)
        history["loss"][i] = np.mean(session_loss)
        history["steps"][i] = np.mean(session_steps)

        tr.set_description(
            "mean reward = {:.3f}\tloss = {:.3f}\tsteps = {:.3f}".format(
                history["reward"][i], history["loss"][i], history["steps"][i]))
    return history


def linear_network(states, scope=None, reuse=False, is_training=False, layers=None,
                   activation_fn=tf.nn.elu):
    layers = layers or [16, 16]
    with tf.variable_scope(scope or "network") as scope:
        if reuse:
            scope.reuse_variables()

        hidden_state = tflayers.stack(
            states,
            tflayers.fully_connected,
            layers,
            activation_fn=activation_fn)

        return hidden_state


def network_wrapper(layers=None, activation_fn=tf.nn.elu):
    def wrapper(states, scope=None, reuse=False, is_training=False):
        return linear_network(states, scope, reuse, is_training, layers, activation_fn)

    return wrapper


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument('--env',
                        type=str,
                        default='CartPole-v0',  # CartPole-v0, MountainCar-v0
                        help='The environment to use')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=100)
    parser.add_argument('--n_sessions',
                        type=int,
                        default=128)
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
    parser.add_argument('--final_epsilon',
                        type=float,
                        default=0.01)
    parser.add_argument('--load',
                        action='store_true',
                        default=False)
    parser.add_argument('--gpu_option',
                        type=float,
                        default=0.45)
    parser.add_argument('--initial_lr',
                        type=float,
                        default=1e-3)
    parser.add_argument('--layers',
                        type=str,
                        default=None)

    args, _ = parser.parse_known_args()
    return args


def run(env, learning_args, update_args, agent_agrs,
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4):
    env_name = env
    env = gym.make(env).env

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    network = agent_agrs["network"] or linear_network

    q_net = ReinforceAgent(state_shape, n_actions, network)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.Saver()
        model_dir = "./logs_" + env_name.replace(string.punctuation, "_")

        if not load:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "{}/model.ckpt".format(model_dir))

        stats = reinforce_learning(
            sess, q_net, env,
            update_fn=update_wraper(**update_args),
            **learning_args)
        create_if_need(model_dir)
        saver.save(sess, "{}/model.ckpt".format(model_dir))

        if plot_stats:
            stats_dir = os.path.join(model_dir, "stats")
            create_if_need(stats_dir)
            save_stats(stats, save_dir=stats_dir)

        if api_key is not None:
            env = env.env
            env = gym.wrappers.Monitor(env, "{}/monitor".format(model_dir), force=True)
            sessions = [generate_session(sess, q_net, env, int(1e10),
                                         update_fn=None)
                        for _ in range(300)]
            env.close()
            gym.upload("{}/monitor".format(model_dir), api_key=api_key)


def main():
    args = _parse_args()
    try:
        layers = tuple(map(int, args.layers.split("-")))
    except:
        layers = None
    network = network_wrapper(layers, activations[args.activation])
    q_learning_args = {
        "n_epochs": args.n_epochs,
        "n_sessions": args.n_sessions,
        "t_max": args.t_max,
    }
    update_args = {
        "discount_factor": args.gamma
    }
    agent_args = {
        "network": network,
        "initial_lr": args.initial_lr
    }
    run(args.env, q_learning_args, update_args, agent_args,
        args.plot_stats, args.api_key,
        args.load, args.gpu_option)


if __name__ == '__main__':
    main()
