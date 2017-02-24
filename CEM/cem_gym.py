#!/usr/bin/python3

import gym
from gym import wrappers
import argparse
import numpy as np
from tqdm import trange
from joblib import Parallel, delayed
import collections
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler

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


class Estimator(object):
    """
    Value Function approximator.
    """

    def __init__(self, env, layers):
        self.n_actions = env.action_space.n
        self._prepare_estimator_for_env(env)
        self.model = MLPClassifier(
            hidden_layer_sizes=layers,
            activation='tanh',
            warm_start=True,
            max_iter=1)
        # We need to call partial_fit once to initialize the model
        # or we get a NotFittedError when trying to make a prediction
        # This is quite hacky.
        self.model.fit(
            [self.featurize_state(env.reset())] * self.n_actions,
            range(self.n_actions))

    def _prepare_estimator_for_env(self, env):
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)])
        observation_examples = self._vectorise_state(observation_examples)

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(observation_examples)
        self.scaler = scaler

        featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        featurizer.fit(scaler.transform(observation_examples))
        self.featurizer = featurizer

    def _vectorise_state(self, states):
        obs_shape = states.shape
        if len(obs_shape) < 2:  # just one observation
            states = np.expand_dims(states, 0)
        elif len(obs_shape) > 2:  # some many states magic
            states = states.reshape((obs_shape[0], -1))
        return states

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        state = self._vectorise_state(state)
        scaled = self.scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        if featurized.shape[0] == 1:
            return featurized[0]
        else:
            return featurized

    def predict_proba(self, s):
        features = self.featurize_state(s)
        return self.model.predict_proba([features])

    def fit(self, s, y):
        features = self.featurize_state(s)
        self.model.partial_fit(features, y)


def generate_session(env, agent, t_max=int(1e5)):
    states, actions = [], []
    total_reward = 0

    s = env.reset()

    for t in range(t_max):

        # predict array of action probabilities
        probs = agent.predict_proba([s])[0]

        a = np.random.choice(env.action_space.n, p=probs)

        new_s, r, done, info = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break

    return states, actions, total_reward, t

glob_env = None
glob_agent = None


def generate_parallel_session(t_max=int(1e5)):
    states, actions = [], []
    total_reward = 0

    s = glob_env.reset()

    for t in range(t_max):

        # predict array of action probabilities
        probs = glob_agent.predict_proba([s])[0]

        a = np.random.choice(glob_env.action_space.n, p=probs)

        new_s, r, done, info = glob_env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break

    return states, actions, total_reward, t


def generate_parallel_sessions(n, t_max, n_jobs=-1):
    return Parallel(n_jobs)(n * [delayed(generate_parallel_session)(t_max)])


def cem(env, agent, num_episodes, max_steps=int(1e6),
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
        desc="mean reward = {:.3f}\tthreshold = {:.3f}".format(0.0, 0.0),
        leave=True)

    for i in tr:
        # generate new sessions
        sessions = generate_parallel_sessions(n_samples, max_steps, n_jobs)
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
            "mean reward = {:.3f}\tthreshold = {:.3f}".format(
                np.mean(batch_rewards), threshold))

    return history


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument('--env',
                        type=str,
                        default='CartPole-v0',  # CartPole-v0, MountainCar-v0, LunarLander-v2
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
    parser.add_argument('--features',
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

    args, _ = parser.parse_known_args()
    return args


def save_stats(stats, save_dir="./"):
    for key in stats:
        plot_unimetric(stats, key, save_dir)


def run(env, n_episodes=200, max_steps=int(1e5), n_samples=1000,
        percentile=80, features=False, layers=None,
        verbose=False, plot_stats=False, api_key=None, n_jobs=-1, seed=42):
    env_name = env
    if env_name == "MountainCar-v0":
        env = gym.make(env).env
        layers = layers or (20, 10, 20)
    else:
        env = gym.make(env)
        layers = layers or (256, 256, 128)

    if features:
        agent = Estimator(env, layers)
    else:
        agent = MLPClassifier(hidden_layer_sizes=layers,
                              activation='tanh',
                              warm_start=True,
                              max_iter=1)
        agent.fit([env.reset()] * env.action_space.n, range(env.action_space.n))

    env.seed(seed)
    np.random.seed(seed)

    stats = cem(env, agent, n_episodes,
                max_steps=max_steps,
                n_samples=n_samples, percentile=percentile,
                n_jobs=n_jobs, verbose=verbose)
    if plot_stats:
        save_stats(stats)

    if api_key is not None:
        env = gym.wrappers.Monitor(env, "/tmp/" + env_name, force=True)
        sessions = [generate_session(env, agent, int(1e10)) for _ in range(200)]
        env.close()
        # unwrap
        gym.upload("/tmp/" + env_name, api_key=api_key)


def main():
    args = _parse_args()
    try:
        layers = tuple(map(int, args.layers.split("-")))
    except:
        layers = None
    run(args.env, args.num_episodes, args.max_steps, args.n_samples,
        args.percentile, args.features, layers,
        args.verbose, args.plot_stats, args.api_key, args.n_jobs, args.seed)


if __name__ == '__main__':
    main()
