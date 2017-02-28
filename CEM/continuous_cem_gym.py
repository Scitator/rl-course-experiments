#!/usr/bin/python3

import gym
from gym import wrappers
import pickle
import argparse
import numpy as np
from tqdm import trange
from joblib import Parallel, delayed
import collections
from sklearn.neural_network import MLPRegressor

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


def generate_session(env, agent, t_max=int(1e5), step_penalty=0.01):
    states, actions = [], []
    total_reward = 0

    s = env.reset()

    for t in range(t_max):
        a = agent.predict([s])[0]
        a = np.array(list(map(
            lambda x: min(
                max(x[1], env.action_space.low[x[0]]),
                env.action_space.high[x[0]]),
            enumerate(a))))

        new_s, r, done, info = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break

    total_reward -= t * step_penalty

    return states, actions, total_reward, t


glob_env = None
glob_agent = None


def generate_parallel_session(t_max=int(1e5), step_penalty=0.01):
    states, actions = [], []
    total_reward = 0

    s = glob_env.reset()

    for t in range(t_max):
        a = glob_agent.predict([s])[0]
        a = np.array(list(map(
            lambda x: min(
                max(x[1], glob_env.action_space.low[x[0]]),
                glob_env.action_space.high[x[0]]),
            enumerate(a))))

        new_s, r, done, info = glob_env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break

    total_reward -= t * step_penalty

    return states, actions, total_reward, t


def generate_parallel_sessions(n, t_max, step_penalty, n_jobs=-1):
    return Parallel(n_jobs)(n * [delayed(generate_parallel_session)(t_max, step_penalty)])


def cem(env, agent, num_episodes, max_steps=int(1e6), step_penalty=0.01,
        n_samples=200, percentile=50, n_jobs=-1, verbose=False):
    global glob_env, glob_agent
    init_n_samples = n_samples
    final_n_samples = n_samples // 5
    plays_to_decay = num_episodes // 2

    states_deque = collections.deque(maxlen=int(init_n_samples * 2))
    actions_deque = collections.deque(maxlen=int(init_n_samples * 2))
    rewards_deque = collections.deque(maxlen=int(init_n_samples * 2))

    glob_env = env  # NEVER DO LIKE THIS PLEASE!
    glob_agent = agent

    # Keeps track of useful statistics
    history = {
        "threshold": np.zeros(num_episodes),
        "reward": np.zeros(num_episodes),
        "n_steps": np.zeros(num_episodes),
    }

    tr = trange(
        num_episodes,
        desc="mean reward = {:.3f}\tthreshold = {:.3f}\tmean n_steps = {:.3f}".format(0.0, 0.0,
                                                                                      0.0),
        leave=True)

    for i in tr:
        # generate new sessions
        # sessions = [
        #     generate_session(env, agent, max_steps, step_penalty)
        #     for _ in range(n_samples)]
        sessions = generate_parallel_sessions(n_samples, max_steps, step_penalty, n_jobs)
        if i < plays_to_decay:
            n_samples -= (init_n_samples - final_n_samples) // plays_to_decay

        batch_states, batch_actions, batch_rewards, batch_steps = map(np.array, zip(*sessions))
        # batch_states: a list of lists of states in each session
        # batch_actions: a list of lists of actions in each session
        # batch_rewards: a list of floats - total rewards at each session
        states_deque.extend(batch_states)
        actions_deque.extend(batch_actions)
        rewards_deque.extend(batch_rewards)

        batch_states = np.array(states_deque)
        batch_actions = np.array(actions_deque)
        batch_rewards = np.array(rewards_deque)

        threshold = np.percentile(batch_rewards, percentile)

        history["threshold"][i] = threshold
        history["reward"][i] = np.mean(batch_rewards)
        history["n_steps"][i] = np.mean(batch_steps)

        # look like > better, cause >= refer to reuse of bad examples
        if i < plays_to_decay:
            elite_states = batch_states[batch_rewards > threshold]
            elite_actions = batch_actions[batch_rewards > threshold]
        else:
            elite_states = batch_states[batch_rewards >= threshold]
            elite_actions = batch_actions[batch_rewards >= threshold]

        if len(elite_actions) > 0:
            elite_states, elite_actions = map(np.concatenate, [elite_states, elite_actions])
            # elite_states: a list of states from top games
            # elite_actions: a list of actions from top games
            try:
                agent.fit(elite_states, elite_actions)
            except:
                # just a hack
                addition = np.array([env.reset()] * env.action_space.n)
                elite_states = np.vstack((elite_states, addition))
                elite_actions = np.hstack((elite_actions, list(range(env.action_space.n))))
                agent.fit(elite_states, elite_actions)

        tr.set_description(
            "mean reward = {:.3f}\tthreshold = {:.3f}\tmean n_steps = {:.3f}".format(
                np.mean(batch_rewards) + step_penalty * np.mean(batch_steps),
                threshold, np.mean(batch_steps)))

    return history


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument('--env',
                        type=str,
                        default='MountainCarContinuous-v0',  # MountainCar-v0, LunarLander-v2
                        help='The environment to use')
    parser.add_argument('--num_episodes',
                        type=int,
                        default=200,
                        help='Number of episodes')
    parser.add_argument('--max_steps',
                        type=int,
                        default=int(1e5),
                        help='Number of steps per episode')
    parser.add_argument('--n_samples',
                        type=int,
                        default=1000,
                        help='Games per epoch')
    parser.add_argument('--step_penalty',
                        type=float,
                        default=0.01)
    parser.add_argument('--percentile',
                        type=int,
                        default=80,
                        help='percentile')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False)
    parser.add_argument('--plot_stats',
                        action='store_true',
                        default=False)
    parser.add_argument('--layers',
                        type=str,
                        default=None)
    parser.add_argument('--api_key',
                        type=str,
                        default=None)
    parser.add_argument('--n_jobs',
                        type=int,
                        default=-1)
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument('--resume',
                        action='store_true',
                        default=False)

    args, _ = parser.parse_known_args()
    return args


def save_stats(stats, save_dir="./"):
    for key in stats:
        plot_unimetric(stats, key, save_dir)


def run(env, n_episodes=200, max_steps=int(1e5), n_samples=1000, step_penalty=0.01,
        percentile=80, layers=None,
        verbose=False, plot_stats=False, api_key=None, n_jobs=-1, seed=42, resume=False):
    env_name = env
    if env_name == "MountainCarContinuous-v0":
        env = gym.make(env).env
    else:
        env = gym.make(env)
    layers = layers or (256, 256, 128)

    agent = MLPRegressor(
        hidden_layer_sizes=layers,
        activation='tanh',
        warm_start=True,
        max_iter=1)
    agent.fit(
        np.zeros(env.observation_space.shape).reshape(1, -1),
        np.zeros(env.action_space.shape).reshape(1, -1))

    if resume:
        agent = pickle.load(open("agent.pkl", "rb"))

    env.seed(seed)
    np.random.seed(seed)

    stats = cem(env, agent, n_episodes,
                max_steps=max_steps, step_penalty=step_penalty,
                n_samples=n_samples, percentile=percentile,
                n_jobs=n_jobs, verbose=verbose)
    if plot_stats:
        save_stats(stats)

    pickle.dump(agent, open("agent.pkl", "wb"))

    if api_key is not None:
        env = gym.wrappers.Monitor(env, "/tmp/" + env_name, force=True)
        sessions = [generate_session(env, agent, int(1e10), 0.0) for _ in range(300)]
        env.close()
        gym.upload("/tmp/" + env_name, api_key=api_key)


def main():
    args = _parse_args()
    try:
        layers = tuple(map(int, args.layers.split("-")))
    except:
        layers = None
    run(args.env, args.num_episodes, args.max_steps, args.n_samples, args.step_penalty,
        args.percentile, layers,
        args.verbose, args.plot_stats, args.api_key, args.n_jobs, args.seed, args.resume)


if __name__ == '__main__':
    main()
