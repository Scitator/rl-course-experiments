import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm

plt.style.use("ggplot")
import seaborn as sns  # noqa: E402

sns.set(color_codes=True)

import numpy as np
import argparse
import gym
from gym.core import ObservationWrapper
import os
import pickle
from tqdm import trange

from .qlearning import QLearningAgent
from .sarsa import SarsaAgent
from .evsarsa import EVSarsaAgent


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


class Binarizer(ObservationWrapper):
    def __init__(self, env, bins=None):
        super().__init__(env)
        self.n_bins = (bins or [10] * env.action_space.n)

    def _state_encoder(self, i, s_i):
        return int(self.n_bins[i] * s_i)

    def _observation(self, state):
        state = map(lambda x: self._state_encoder(x[0], x[1]), enumerate(state))

        return tuple(state)


def play_and_train_qlearning(env, agent, t_max=10 ** 3):
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        a = agent.getAction(s)

        next_s, r, done, _ = env.step(a)

        agent.update(s, a, next_s, r)

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


def play_and_train_sarsa(env, agent, t_max=10 ** 3):
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        a = agent.getAction(s)

        next_s, r, done, _ = env.step(a)

        agent.update(s, a, next_s, agent.getAction(next_s), r)

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


def play_and_train_evsarsa(env, agent, t_max=10 ** 3):
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        a = agent.getAction(s)

        next_s, r, done, _ = env.step(a)

        agent.update(s, a, next_s, r)

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


def agent_runner(
        env, agent_fn, agent_play_fn,
        n_epochs=int(2e5), alpha=0.05, discount=0.99,
        initial_epsilon=0.25, final_epsilon=0.01):
    n_actions = env.action_space.n

    agent = agent_fn(
        alpha=alpha, epsilon=initial_epsilon, discount=discount,
        getLegalActions=lambda s: range(n_actions))

    n_epochs_decay = n_epochs * 0.8

    tr = trange(
        n_epochs,
        desc="",
        leave=True)

    rewards = np.zeros(n_epochs)
    eps = np.zeros(n_epochs)
    epoch_rewards = np.zeros(n_epochs // 1000)
    agent.epsilon = initial_epsilon
    for i in tr:
        rewards[i] = agent_play_fn(env, agent)
        eps[i] = agent.epsilon

        if i < n_epochs_decay:
            agent.epsilon -= (initial_epsilon - final_epsilon) / float(n_epochs_decay)

        if i % 1000 == 0:
            epoch_rewards[i // 1000] = np.mean(rewards[i - 1000:i])
            desc = "reward: {}\tepsilon: {}".format(epoch_rewards[i // 1000], agent.epsilon)
            tr.set_description(desc)

    return {
        "reward": rewards,
        "epoch_reward": epoch_rewards,
        "epsilon": eps
    }


AGENTS = {
    "qlearning": QLearningAgent,
    "sarsa": SarsaAgent,
    "evsarsa": EVSarsaAgent,
}

AGENTS_FN = {
    "qlearning": play_and_train_qlearning,
    "sarsa": play_and_train_sarsa,
    "evsarsa": play_and_train_evsarsa,
}


def run(env, agent, bins=None,
        lr=0.05, discount_factor=0.99, n_steps=1, initial_epsilon=0.25,
        n_epochs=1000, t_max=1000,
        plot_stats=False, api_key=None):
    env_name = env
    env = Binarizer(gym.make(env).env, bins=bins)
    agent = AGENTS[agent]
    agent_fn = lambda env, agent: AGENTS_FN[agent](env, agent, t_max=t_max)

    history = agent_runner(
        env, agent, agent_fn,
        n_epochs, lr, discount_factor,
        initial_epsilon)

    if plot_stats:
        save_stats(history)

    if api_key is not None:
        env = gym.wrappers.Monitor(env, "/tmp/" + env_name, force=True)
        for _ in range(200):
            agent_fn(env, agent)
        env.close()
        gym.upload("/tmp/" + env_name, api_key=api_key)


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument(
        '--env',
        type=str,
        default='CartPole-v0',  # CartPole-v0, MountainCar-v0
        help='The environment to use')
    parser.add_argument(
        '--agent',
        type=str,
        default='qlearning',  # qlearning, sarsa, evsarsa
        help='The agent to use')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=1000)
    parser.add_argument(
        '--t_max',
        type=int,
        default=1000)
    parser.add_argument(
        '--lr',
        type=float,
        default=0.05,
        help='Agent learning rate')
    parser.add_argument(
        '--initial_epsilon',
        type=float,
        default=0.99,
        help='Agent start exploration factor')
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
        '--n_steps',
        type=int,
        default=1)
    parser.add_argument(
        '--bins',
        type=str,
        default=None)

    args, _ = parser.parse_known_args()
    return args


def main():
    args = _parse_args()
    try:
        bins = tuple(map(int, args.bins.split("-")))
    except:
        bins = None
    run(args.env, args.agent, bins,
        args.lr, args.gamma, args.n_steps, args.initial_epsilon,
        args.n_epochs, args.t_max,
        args.plot_stats, args.api_key)


if __name__ == '__main__':
    main()
