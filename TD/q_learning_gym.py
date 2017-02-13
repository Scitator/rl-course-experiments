#!/usr/bin/python

import gym
from gym import wrappers
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
plt.style.use("ggplot")


def plot_unimetric(history, metric, save_dir):
    plt.figure()
    plt.plot(history[metric])
    plt.title('model {}'.format(metric))
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.savefig("{}/{}.png".format(save_dir, metric),
                format='png', dpi=300)


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        import pdb; pdb.set_trace()
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, verbose=False):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if verbose and (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()

        len_counter = 0
        reward_counter = 0
        done = False
        while not done:
            if verbose:
                env.render()
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            n_state, reward, done, info = env.step(action)
            reward_counter += reward
            len_counter += 1
            bn_action = np.argmax(Q[n_state])
            td_target = reward + discount_factor * Q[n_state][bn_action]
            td_delta = td_target - Q[state][action]

            Q[state][action] += alpha * td_delta
            state = n_state

        episode_rewards[i_episode] = reward_counter
        episode_lengths[i_episode] = len_counter

    stats = {"episode_rewards": episode_rewards, "episode_lengths": episode_lengths}
    return Q, stats


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument('--env',
                        type=str,
                        default='MountainCar-v0',
                        help='The environment to use')
    parser.add_argument('--num_episodes',
                        type=int,
                        default=1000,
                        help='Number of episodes')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='Gamma discount factor')
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


def run(env, n_episodes, discount_factor, verbose=False, plot_stats=False, api_key=None):
    env_name = env
    env = gym.make(env)

    if api_key is not None:
        env = gym.wrappers.Monitor(env, "/tmp/" + env_name, force=True)

    stats = q_learning(env, n_episodes,
                       discount_factor=discount_factor,
                       verbose=verbose)
    if plot_stats:
        save_stats(stats)

    if api_key is not None:
        env.close()
        gym.upload("/tmp/" + env_name, api_key=api_key)


def main():
    args = _parse_args()
    run(args.env, args.num_episodes, args.gamma, args.verbose, args.plot_stats, args.api_key)


if __name__ == '__main__':
    assert False, "Does not work yet"
    main()
