#!/usr/bin/python3

import gym
from gym import wrappers
import sys
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
from tqdm import trange

from matplotlib import pyplot as plt
plt.style.use("ggplot")


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
    "softplus": tf.nn.softplus
}


class DqnAgent(object):
    def __init__(self, state_shape, n_actions, network, gamma=0.99, special=None):
        self.special = special or {}
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.current_states = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.next_states = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32)
        self.is_end = tf.placeholder(shape=[None], dtype=tf.bool)

        scope = self.special.get("scope", "network")
        
        self.predicted_qvalues = self.qnetwork(network, self.current_states, scope=scope)

        one_hot_actions = tf.one_hot(self.actions, n_actions)
        predicted_qvalues_for_actions = tf.reduce_sum(
            tf.multiply(self.predicted_qvalues, one_hot_actions),
            axis=-1)
        predicted_next_qvalues =self.qnetwork(network, self.next_states, scope=scope, reuse=True)

        target_qvalues_for_actions = self.rewards + \
            gamma * tf.reduce_max(predicted_next_qvalues, axis=-1)
        target_qvalues_for_actions = tf.where(
            self.is_end,
            tf.zeros_like(target_qvalues_for_actions),
            target_qvalues_for_actions)

        self.loss = tf.reduce_mean(tf.square(target_qvalues_for_actions - predicted_qvalues_for_actions))

        self.update_step = tf.train.AdamOptimizer(1e-4).minimize(
            self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))

    def qnetwork(self, network, state, scope, reuse=False):
        hidden_state = network(state, scope=scope, reuse=reuse)
        qvalues = self._to_qvalues(hidden_state, scope=scope, reuse=reuse)
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


def generate_session(sess, agent, env, epsilon=0.5, t_max=1000):
    """play env with approximate q-learning agent and train it at the same time"""
    
    total_reward = 0
    s = env.reset()
    total_loss = 0
    
    for t in range(t_max):
        
        #get action q-values from the network
        q_values = sess.run(
            agent.predicted_qvalues, 
            feed_dict={
                agent.current_states:np.array([s])})[0]
        
        if np.random.rand() < epsilon:
            a = np.random.choice(agent.n_actions)
        else:
            a = np.argmax(q_values)
        
        new_s,r,done,info = env.step(a)
        
        curr_loss, _ = sess.run(
            [agent.loss, agent.update_step], 
            feed_dict={
                agent.current_states: np.array([s], dtype=np.float32), 
                agent.actions: np.array([a], dtype=np.int32), 
                agent.rewards: np.array([r], dtype=np.float32), 
                agent.next_states: np.array([new_s], dtype=np.float32), 
                agent.is_end: np.array([done], dtype=np.bool)})

        total_reward += r
        total_loss += curr_loss
        
        s = new_s
        if done: 
            break
            
    return total_reward, total_loss/float(t), t


def q_learning(agent, env, n_epochs, n_sessions=100, t_max=1000, inial_epsilon=0.5, final_epsilon=0.01):
    tr = trange(
        n_epochs,
        desc="mean reward = {:.3f}\tepsilon = {:.3f}\tloss = {:.3f}\tsteps = {:.3f}".format(0.0, 0.0, 0.0, 0.0),
        leave=True)

    epsilon = 0.5
    n_epochs_decay = n_epochs * 0.8
    
    history = {
        "reward": np.zeros(n_epochs),
        "epsilon": np.zeros(n_epochs),
        "loss": np.zeros(n_epochs),
        "steps": np.zeros(n_epochs),
    }

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in tr:
            sessions = [generate_session(sess, agent, env, epsilon, t_max) for _ in range(n_sessions)]
            session_rewards, session_loss, session_steps = map(np.array, zip(*sessions))
            
            if i < n_epochs_decay:
                epsilon -= (inial_epsilon - final_epsilon) / float(n_epochs_decay)
            
            history["reward"][i] = np.mean(session_rewards)
            history["epsilon"][i] = epsilon
            history["loss"][i] = np.mean(session_loss)
            history["steps"][i] = np.mean(session_steps)
            
            tr.set_description("mean reward = {:.3f}\tepsilon = {:.3f}\tloss = {:.3f}\tsteps = {:.3f}".format(
                 history["reward"][i], history["epsilon"][i], history["loss"][i], history["steps"][i]))
    
    return history


def linear_network(states, scope=None, reuse=False, layers=None, activation_fn=tf.tanh):
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


def linear_network_wrapper(layers=None, activation_fn=tf.tanh):
    def wrapper(states, scope=None, reuse=False):
        return linear_network(states, scope, reuse, layers, activation_fn)
    return wrapper


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument('--env',
                        type=str,
                        default='CartPole-v0',  # CartPole-v0, MountainCar-v0
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
    parser.add_argument('--layers',
                        type=str,
                        default=None)
    parser.add_argument('--activation',
                        type=str,
                        default="tanh")

    args, _ = parser.parse_known_args()
    return args


def run(env, n_epochs, discount_factor, plot_stats=False, api_key=None, network=None):
    env_name = env
    env = gym.make(env)

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    network = network or linear_network
    agent = DqnAgent(
        state_shape, n_actions, network,
        gamma=discount_factor)

    stats = q_learning(agent, env, n_epochs)
    if plot_stats:
        save_stats(stats)

    if api_key is not None:
        env = gym.wrappers.Monitor(env, "/tmp/" + env_name, force=True)
        sessions = [generate_session(sess, agent, env, 0.01, int(1e10)) for _ in range(300)]
        env.close()
        gym.upload("/tmp/" + env_name, api_key=api_key)


def main():
    args = _parse_args()
    try:
        layers = tuple(map(int, args.layers.split("-")))
    except:
        layers = None
    network = linear_network_wrapper(layers, activations[args.activation])
    run(args.env, args.n_epochs, args.gamma, args.plot_stats, args.api_key, network)


if __name__ == '__main__':
    main()
