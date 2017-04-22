#!/usr/bin/python3

import gym
from gym import wrappers
import sys
import argparse
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
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


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


class Estimator(object):
    """
    Value Function approximator.
    """

    def __init__(self, env):
        self._prepare_estimator_for_env(env)
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def _prepare_estimator_for_env(self, env):
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(1000)])
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
        if len(obs_shape) > 2:
            states = states.reshape((obs_shape[0], -1))
        return states

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        state = self._vectorise_state(np.array([state]))
        scaled = self.scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        features = self.featurize_state(s)
        return self.models[a].predict([features])[0] if a \
            else np.array([model.predict([features])[0] for model in self.models])

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0,
               verbose=False):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    for i_episode in range(num_episodes):

        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        if verbose:
            last_reward = episode_rewards[i_episode - 1]
            print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
            sys.stdout.flush()

        state = env.reset()
        n_action = None

        len_counter = 0
        reward_counter = 0
        done = False
        while not done:
            if verbose:
                pass
                # env.render()
            if n_action is None:
                probs = policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
            else:
                action = n_action

            n_state, reward, done, info = env.step(action)
            reward_counter += reward
            len_counter += 1

            q_val_next = estimator.predict(n_state)
            td_target = reward + discount_factor * np.max(q_val_next)

            estimator.update(state, action, td_target)

            state = n_state

        episode_rewards[i_episode] = reward_counter
        episode_lengths[i_episode] = len_counter

    return {"episode_rewards": episode_rewards, "episode_lengths": episode_lengths}


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument(
        '--env',
        type=str,
        default='MountainCar-v0',  # CartPole-v0, MountainCar-v0
        help='The environment to use')
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=1000,
        help='Number of episodes')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Gamma discount factor')
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False)
    parser.add_argument(
        '--plot_stats',
        action='store_true',
        default=False)
    parser.add_argument(
        '--api_key',
        type=str,
        default=None)

    args, _ = parser.parse_known_args()
    return args


def save_stats(stats, save_dir="./"):
    for key in stats:
        plot_unimetric(stats, key, save_dir)


def run(env, n_episodes, discount_factor, verbose=False, plot_stats=False, api_key=None):
    env_name = env
    env = gym.make(env)

    estimator = Estimator(env)

    if api_key is not None:
        env = gym.wrappers.Monitor(env, "/tmp/" + env_name, force=True)

    stats = q_learning(env, estimator, n_episodes,
                       discount_factor=discount_factor, epsilon=0.0,
                       verbose=verbose)
    if plot_stats:
        save_stats(stats)

    if api_key is not None:
        env.close()
        gym.upload("/tmp/" + env_name, api_key=api_key)


def main():
    args = _parse_args()
    run(args.env, args.num_episodes, args.gamma,
        args.verbose, args.plot_stats, args.api_key)


if __name__ == '__main__':
    main()
