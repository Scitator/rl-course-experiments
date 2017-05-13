#!/usr/bin/python3

import gym
import argparse
import numpy as np
import tensorflow as tf
from tqdm import trange
import string
import os

from rstools.tf.optimization import build_model_optimization
from rstools.utils.batch_utils import iterate_minibatches
from rstools.utils.os_utils import save_history, save_model
from rstools.visualization.plotter import plot_all_metrics
from agent_networks import FeatureNet, PolicyNet, StateNet
from networks import conv_network, linear_network, activations, network_wrapper
from wrappers import EnvPool, Transition


class A3CFF(object):
    def __init__(self, state_shape, n_actions, network, special=None):
        self.special = special or {}
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.feature_net = FeatureNet(state_shape, network, special.get("feature_get", None))
        self.hidden_state = tf.layers.dense(
            self.feature_net.hidden_state,
            self.special.get("hidden_size", 512),
            activation=self.special.get("hidden_actiovation", tf.nn.elu))
        self.policy_net = PolicyNet(self.hidden_state, n_actions, special.get("policy_net", None))
        self.state_net = StateNet(self.hidden_state, special.get("state_net", None))

        build_model_optimization(self.policy_net, special.get("policy_net_optimization", None))
        build_model_optimization(self.state_net, special.get("state_net_optimization", None))
        build_model_optimization(
            self.feature_net,
            special.get("feature_net_optimization", None),
            loss=0.5 * (self.policy_net.loss + self.state_net.loss))

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


def action(sess, agent, state):
    probs = agent.predict_action(sess, state)
    actions = [np.random.choice(np.arange(len(row)), p=row) for row in probs]
    return actions


def update(sess, aac_agent, transitions, discount_factor=0.99, batch_size=32, time_major=True):
    policy_targets = []
    value_targets = []
    state_history = []
    action_history = []

    cumulative_reward = np.zeros_like(transitions[-1].reward) + \
                        np.invert(transitions[-1].done) * \
                        aac_agent.value_net.predict(sess, transitions[-1].next_state)
    for transition in reversed(transitions):
        cumulative_reward = transition.reward + \
                            np.invert(transition.done) * discount_factor * cumulative_reward
        policy_target = cumulative_reward - aac_agent.value_net.predict(sess, transition.state)

        value_targets.append(cumulative_reward)
        policy_targets.append(policy_target)
        state_history.append(transition.state)
        action_history.append(transition.action)

    value_targets = np.array(value_targets)
    policy_targets = np.array(policy_targets)
    state_history = np.array(state_history)  # time-major
    action_history = np.array(action_history)

    if not time_major:
        state_history = state_history.swapaxes(0, 1)
        action_history = action_history.swapaxes(0, 1)
        value_targets = value_targets.swapaxes(0, 1)
        policy_targets = policy_targets.swapaxes(0, 1)

    time_len = state_history.shape[0]

    value_loss, policy_loss = 0.0, 0.0
    for state_axis, action_axis, value_target_axis, policy_target_axis in \
            zip(state_history, action_history, value_targets, policy_targets):
        axis_len = state_axis.shape[0]
        axis_value_loss, axis_policy_loss = 0.0, 0.0

        state_axis = iterate_minibatches(state_axis, batch_size, shuffle=False)
        action_axis = iterate_minibatches(action_axis, batch_size, shuffle=False)
        value_target_axis = iterate_minibatches(value_target_axis, batch_size, shuffle=False)
        policy_target_axis = iterate_minibatches(policy_target_axis, batch_size, shuffle=False)

        for state_batch, action_batch, value_target, policy_target in \
                zip(state_axis, action_axis, value_target_axis, policy_target_axis):
            axis_value_loss += aac_agent.value_net.update(
                sess, state_batch, value_target)
            axis_policy_loss += aac_agent.policy_net.update(
                sess, state_batch, action_batch, policy_target)

        policy_loss += axis_policy_loss / axis_len
        value_loss += axis_value_loss / axis_len

    return policy_loss / time_len, value_loss / time_len


def update_wraper(discount_factor=0.99, batch_size=32, time_major=False):
    def wrapper(sess, q_net, memory):
        return update(sess, q_net, memory, discount_factor, batch_size, time_major)

    return wrapper


def generate_session(
        sess, aac_agent, env, t_max=1000, update_fn=None):
    total_reward = 0
    total_policy_loss, total_state_loss = 0, 0

    transitions = []

    s = env.reset()
    for t in range(t_max):
        a = action(aac_agent, sess, np.array([s], dtype=np.float32))[0]

        next_s, r, done, _ = env.step(a)

        transitions.append(Transition(
            state=s, action=a, reward=r, next_state=next_s, done=done))

        total_reward += r

        s = next_s
        if done:
            break

    if update_fn is not None:
        total_policy_loss, total_value_loss = update_fn(sess, aac_agent, transitions)

    return total_reward, total_policy_loss, total_state_loss, t


def generate_sessions(sess, aac_agent, env_pool, t_max=1000, update_fn=None):
    total_reward = 0.0
    total_policy_loss, total_value_loss = 0, 0
    total_games = 0.0

    transitions = []

    states = env_pool.pool_states()
    for t in range(t_max):
        actions = action(aac_agent, sess, states)
        next_states, rewards, dones, _ = env_pool.step(actions)

        transitions.append(Transition(
            state=states, action=actions, reward=rewards, next_state=next_states, done=dones))
        states = next_states

        total_reward += rewards.mean()
        total_games += dones.sum()

    if update_fn is not None:
        total_policy_loss, total_value_loss = update_fn(sess, aac_agent, transitions)

    return total_reward, total_policy_loss, total_value_loss, total_games


def actor_critic_learning(
        sess, q_net, env, update_fn,
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
            generate_sessions(sess, q_net, env, t_max, update_fn=update_fn)
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


def _parse_args():
    parser = argparse.ArgumentParser(description='FeedForward A3C Agent')
    parser.add_argument(
        '--env',
        type=str,
        default='CartPole-v0',  # CartPole-v0, MountainCar-v0
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
        default=500)
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
        '--batch_size',
        type=int,
        default=10)
    parser.add_argument(
        '--policy_lr',
        type=float,
        default=1e-3)
    parser.add_argument(
        '--value_lr',
        type=float,
        default=1e-3)
    parser.add_argument(
        '--entropy_koef',
        type=float,
        default=1e-2)
    parser.add_argument(
        '--n_games',
        type=int,
        default=10)
    parser.add_argument(
        '--time_major',
        action='store_true',
        default=False)

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

    q_net = A3CFF(
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

        try:
            history = actor_critic_learning(
                sess, q_net, env,
                update_fn=update_wraper(**update_args),
                **q_learning_args)
        except KeyboardInterrupt:
            print("Exiting training procedure")
        save_model(sess, saver, model_dir)

        if plot_stats:
            save_history(history, model_dir)
            plotter_dir = os.path.join(model_dir, "plotter")
            plot_all_metrics(history, save_dir=plotter_dir)

        if api_key is not None:
            env = gym.make(env_name)
            monitor_dir = os.path.join(model_dir, "monitor")
            env = gym.wrappers.Monitor(env, monitor_dir, force=True)
            sessions = [generate_session(sess, q_net, env, int(1e10), update_fn=None)
                        for _ in range(300)]
            env.close()
            gym.upload(monitor_dir, api_key=api_key)


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
        "batch_size": args.batch_size,
        "time_major": args.time_major
    }
    agent_args = {
        "network": network,
        "policy_lr": args.policy_lr,
        "value_lr": args.value_lr,
        "entropy_koef": args.entropy_koef
    }
    run(args.env, q_learning_args, update_args, agent_args,
        args.n_games,
        args.plot_stats, args.api_key,
        args.load, args.gpu_option)


if __name__ == '__main__':
    main()
