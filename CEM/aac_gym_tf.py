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

from wrappers import EnvPool


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


class ObservationsBuffer(object):
    def __init__(self, buffer_len=10000):
        self.buffer = collections.deque(maxlen=buffer_len)

    def observe(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done))

    def get_batch(self, batch_size):
        batch_ids = np.random.choice(len(self.buffer), batch_size)
        batch = np.array([self.buffer[i] for i in batch_ids])
        return batch


class PolicyAgent(object):
    def __init__(self, state_shape, n_actions, network, special=None):
        self.special = special or {}
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.states = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.cumulative_rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)

        self.scope = self.special.get("scope", "network")

        hidden_state = network(
            self.states,
            scope=self.scope + "_hidden",
            reuse=self.special.get("reuse_hidden", False),
            is_training=self.is_training)

        self.predicted_probs = self._probs(
            hidden_state,
            scope=self.scope + "_probs",
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
            self.loss += H * 0.001

        optim = tf.train.AdamOptimizer(self.special.get("policy_lr", 1e-4))

        with tf.variable_scope(self.scope + "_hidden") as scope:
            if self.special.get("reuse_hidden", False):
                scope.reuse_variables()
            self.hidden_state_update = optim.minimize(
                self.loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=self.scope + "_hidden"))

        with tf.variable_scope(self.scope + "_probs") as scope:
            if self.special.get("reuse_probs", False):
                scope.reuse_variables()
            self.probs_update = optim.minimize(
                self.loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=self.scope + "_probs"))

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
        loss, _, _ = sess.run(
            [self.loss, self.hidden_state_update, self.probs_update],
            feed_dict={
                self.states: state_batch,
                self.actions: action_batch,
                self.cumulative_rewards: rewards_batch,
                self.is_training: is_training})
        return loss


class ValueAgent(object):
    def __init__(self, state_shape, n_actions, network, special=None):
        self.special = special or {}
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.states = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32)
        self.td_targets = tf.placeholder(shape=[None], dtype=tf.float32)

        self.is_training = tf.placeholder(dtype=tf.bool)

        self.scope = self.special.get("scope", "network")

        hidden_state = network(
            self.states,
            scope=self.scope + "_hidden",
            reuse=self.special.get("reuse_hidden", False),
            is_training=self.is_training)

        self.predicted_values = tf.squeeze(
            self._state_value(
                hidden_state,
                scope=self.scope + "_state_value",
                reuse=self.special.get("reuse_state_value", False)))

        self.loss = tf.losses.mean_squared_error(
            labels=self.td_targets,
            predictions=self.predicted_values)

        optim = tf.train.AdamOptimizer(self.special.get("value_lr", 1e-4))

        with tf.variable_scope(self.scope + "_hidden") as scope:
            if self.special.get("reuse_hidden", False):
                scope.reuse_variables()
            self.hidden_state_update = optim.minimize(
                self.loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=self.scope + "_hidden"))

        with tf.variable_scope(self.scope + "_state_value") as scope:
            if self.special.get("reuse_state_value", False):
                scope.reuse_variables()
            self.state_value_update = optim.minimize(
                self.loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=self.scope + "_state_value"))

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
            self.predicted_values,
            feed_dict={
                self.states: state_batch,
                self.is_training: is_training})

    def update(self, sess, state_batch, td_target_batch, is_training=True):
        loss, _, _ = sess.run(
            [self.loss, self.hidden_state_update, self.state_value_update],
            feed_dict={
                self.states: state_batch,
                self.td_targets: td_target_batch,
                self.is_training: is_training})
        return loss


class AACAgent(object):
    def __init__(self, state_shape, n_actions, network, special=None):
        self.special = special or {}
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.policy_net = PolicyAgent(state_shape, n_actions, network, special)
        special["reuse_hidden"] = True
        self.value_net = ValueAgent(state_shape, n_actions, network, special)


def action(agent, sess, state):
    probs = agent.policy_net.predict(sess, state)
    actions = [np.random.choice(len(row), p=row) for row in probs]
    return actions


def update(sess, aac_agent, memory, discount_factor=0.99, batch_size=32):
    policy_loss = 0.0
    value_loss = 0.0

    if len(memory.buffer) >= batch_size:
        batch_ids = np.random.choice(len(memory.buffer), batch_size)
        batch = np.array([memory.buffer[i] for i in batch_ids])

        state_batch = np.vstack(batch[:, 0]).reshape((-1,) + aac_agent.state_shape)
        action_batch = np.vstack(batch[:, 1]).reshape(-1)
        reward_batch = np.vstack(batch[:, 2]).reshape(-1)
        next_state_batch = np.vstack(batch[:, 3]).reshape((-1,) + aac_agent.state_shape)
        done_batch = np.vstack(batch[:, 4]).reshape(-1)

        values = aac_agent.value_net.predict(sess, state_batch)
        values_next = aac_agent.value_net.predict(sess, next_state_batch)
        td_target = reward_batch + \
            np.invert(done_batch).astype(np.float32) * \
            discount_factor * values_next
        td_error = td_target - values

        value_loss = aac_agent.value_net.update(sess, state_batch, td_target)

        policy_loss = aac_agent.policy_net.update(sess, state_batch, action_batch, td_error)

    return policy_loss, value_loss


def update_wraper(discount_factor=0.99, batch_size=32):
    def wrapper(sess, q_net, memory):
        return update(sess, q_net, memory, discount_factor, batch_size)

    return wrapper


def generate_session(
        sess, aac_agent, memory, env, t_max=1000, update_fn=None):
    total_reward = 0
    total_policy_loss, total_state_loss = 0, 0
    s = env.reset()

    for t in range(t_max):
        a = action(aac_agent, sess, np.array([s], dtype=np.float32))[0]

        new_s, r, done, _ = env.step(a)

        if update_fn is not None:
            memory.observe(s, a, r, new_s, done)
            curr_policy_loss, curr_state_loss = update_fn(sess, aac_agent, memory)
            total_policy_loss += curr_policy_loss
            total_state_loss += curr_state_loss

        total_reward += r

        s = new_s
        if done:
            break

    return total_reward, total_policy_loss / float(t + 1), total_state_loss / float(t + 1),  t


def generate_sessions(sess, aac_agent, memory, env_pool, t_max=1000, update_fn=None):
    total_reward = 0.0
    total_policy_loss, total_value_loss = 0, 0
    total_games = 0.0

    states = env_pool.reset()
    for t in range(t_max):
        actions = action(aac_agent, sess, states)
        new_states, rewards, dones, _ = env_pool.step(actions)

        if update_fn is not None:
            for s, a, r, new_s, done in zip(states, actions, rewards, new_states, dones):
                memory.observe(s, a, r, new_s, done)
            curr_policy_loss, curr_state_loss = update_fn(sess, aac_agent, memory)
            total_policy_loss += curr_policy_loss
            total_value_loss += curr_state_loss

        states = new_states

        total_reward += rewards.mean()
        total_games += dones.sum()

    return total_reward, total_policy_loss, total_value_loss, total_games


def actor_critic_learning(
        sess, q_net, memory, env, update_fn,
        n_epochs=1000, n_sessions=100, t_max=1000):
    tr = trange(
        n_epochs,
        desc="",
        leave=True)

    history = {
        "reward": np.zeros(n_epochs),
        "policy_loss": np.zeros(n_epochs),
        "value_loss": np.zeros(n_epochs),
        "steps": np.zeros(n_epochs),
    }

    for i in tr:
        sessions = [
            generate_sessions(sess, q_net, memory, env, t_max, update_fn=update_fn)
            for _ in range(n_sessions)]
        session_rewards, session_policy_loss, session_value_loss, session_steps = \
            map(np.array, zip(*sessions))

        history["reward"][i] = np.mean(session_rewards)
        history["policy_loss"][i] = np.mean(session_policy_loss)
        history["value_loss"][i] = np.mean(session_value_loss)
        history["steps"][i] = np.mean(session_steps)

        tr.set_description(
            "mean reward = {:.3f}\tpolicy loss = {:.3f}"
            "\tvalue loss = {:.3f}\tsteps = {:.3f}".format(
                history["reward"][i], history["policy_loss"][i],
                history["value_loss"][i], history["steps"][i]))

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
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--buffer_len',
                        type=int,
                        default=10000)
    parser.add_argument('--load',
                        action='store_true',
                        default=False)
    parser.add_argument('--gpu_option',
                        type=float,
                        default=0.45)
    parser.add_argument('--policy_lr',
                        type=float,
                        default=1e-2)
    parser.add_argument('--value_lr',
                        type=float,
                        default=1e-2)
    parser.add_argument('--n_games',
                        type=int,
                        default=64)

    args, _ = parser.parse_known_args()
    return args


def run(env, q_learning_args, update_args, agent_agrs,
        n_games,
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4):
    env_name = env
    env = EnvPool(gym.make(env_name).env, n_games)

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    network = agent_agrs["network"] or linear_network

    memory = ObservationsBuffer(agent_agrs["buffer_len"])

    q_net = AACAgent(
        state_shape, n_actions, network,
        special=agent_agrs)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.Saver()
        model_dir = "./logs_" + env_name.replace(string.punctuation, "_")

        if not load:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "{}/model.ckpt".format(model_dir))

        stats = actor_critic_learning(
            sess, q_net, memory, env,
            update_fn=update_wraper(**update_args),
            **q_learning_args)
        create_if_need(model_dir)
        saver.save(sess, "{}/model.ckpt".format(model_dir))

        if plot_stats:
            stats_dir = os.path.join(model_dir, "stats")
            create_if_need(stats_dir)
            save_stats(stats, save_dir=stats_dir)

        if api_key is not None:
            env = gym.make(env_name)
            env = gym.wrappers.Monitor(env, "{}/monitor".format(model_dir), force=True)
            sessions = [generate_session(sess, q_net, memory, env, int(1e10), update_fn=None)
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
        "discount_factor": args.gamma,
        "batch_size": args.batch_size
    }
    agent_args = {
        "network": network,
        "policy_lr": args.policy_lr,
        "value_lr": args.value_lr,
        "buffer_len": args.buffer_len
    }
    run(args.env, q_learning_args, update_args, agent_args,
        args.n_games,
        args.plot_stats, args.api_key,
        args.load, args.gpu_option)


if __name__ == '__main__':
    main()
