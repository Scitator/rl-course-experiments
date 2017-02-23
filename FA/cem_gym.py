#!/usr/bin/python3

import gym
from gym import wrappers
import sys
import argparse
import numpy as np
from tqdm import trange
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

    def __init__(self, env):
        self.n_actions = env.action_space.n
        self._prepare_estimator_for_env(env)
        self.model = MLPClassifier(
            hidden_layer_sizes=(50, 50, 50, 50, 50),
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
        return self.model.predict_proba([features])[0]

    def update(self, s, y):
        features = self.featurize_state(s)
        self.model.partial_fit(features, y)


def generate_session(env, agent, t_max=int(1e4)):
    states, actions = [], []
    total_reward = 0

    s = env.reset()

    for t in range(t_max):

        # predict array of action probabilities
        probs = agent.predict_proba(s)

        a = np.random.choice(env.action_space.n, p=probs)

        new_s, r, done, info = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward


def cem(env, agent, num_episodes, max_steps=int(1e4),
        n_samples=200, percentile=50, verbose=False):

    # Keeps track of useful statistics
    history = {
        "threshold": np.zeros(num_episodes),
        "reward": np.zeros(num_episodes),
    }

    tr = trange(
        num_episodes,
        desc="mean reward = %.5f\tthreshold = %.1f" % (0.0, 0.0),
        leave=True)

    for i_episode in tr:
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode

        sessions = [generate_session(env, agent, max_steps) for _ in range(n_samples)]

        batch_states, batch_actions, batch_rewards = map(np.array, zip(*sessions))

        threshold = np.percentile(batch_rewards, percentile)

        history["threshold"][i_episode] = threshold
        history["reward"][i_episode] = np.mean(batch_rewards)

        elite_states = batch_states[batch_rewards >= threshold]
        elite_actions = batch_actions[batch_rewards >= threshold]

        elite_states, elite_actions = map(np.concatenate, [elite_states, elite_actions])
        # elite_states: a list of states from top games
        # elite_actions: a list of actions from top games

        try:
            agent.update(elite_states, elite_actions)
        except:
            # just a hack, because of some sklearn MLP problems
            addition = np.array([env.reset()] * agent.n_actions)
            elite_states = np.vstack((elite_states, addition))
            elite_actions = np.hstack((elite_actions, list(range(agent.n_actions))))
            agent.update(elite_states, elite_actions)

        tr.set_description(
            "mean reward = %.5f\tthreshold = %.1f"%(np.mean(batch_rewards),threshold))

    return history


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument('--env',
                        type=str,
                        default='MountainCar-v0',  # CartPole-v0, MountainCar-v0
                        help='The environment to use')
    parser.add_argument('--num_episodes',
                        type=int,
                        default=200,
                        help='Number of episodes')
    parser.add_argument('--max_steps',
                        type=int,
                        default=16000,
                        help='Number of steps per episode')
    parser.add_argument('--n_samples',
                        type=int,
                        default=200,
                        help='Games per epoch')
    parser.add_argument('--percentile',
                        type=int,
                        default=50,
                        help='percentile')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False)
    parser.add_argument('--plot_stats',
                        action='store_true',
                        default=False)
    parser.add_argument('--api_key',
                        type=str,
                        default=None)

    args, _ = parser.parse_known_args()
    return args


def save_stats(stats, save_dir="./"):
    for key in stats:
        plot_unimetric(stats, key, save_dir)


def run(env, n_episodes, max_steps=int(1e4), n_samples=200,
        percentile=50, verbose=False, plot_stats=False, api_key=None):
    env_name = env
    env = gym.make(env)

    estimator = Estimator(env)

    if api_key is not None:
        env = gym.wrappers.Monitor(env, "/tmp/" + env_name, force=True)

    stats = cem(env, estimator, n_episodes,
                max_steps=max_steps,
                n_samples=n_samples, percentile=percentile,
                verbose=verbose)
    if plot_stats:
        save_stats(stats)

    if api_key is not None:
        env.close()
        gym.upload("/tmp/" + env_name, api_key=api_key)


def main():
    args = _parse_args()
    run(args.env, args.num_episodes, args.max_steps, args.n_samples,
        args.percentile, args.verbose, args.plot_stats, args.api_key)


if __name__ == '__main__':
    main()
