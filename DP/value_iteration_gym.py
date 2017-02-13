#!/usr/bin/python

import gym
from gym import wrappers
import argparse
import numpy as np


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
        theta: Stopping threshold.
            If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n`
                containing the expected value of each action.
        """
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.env.env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.observation_space.n)
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.observation_space.n):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function
            V[s] = best_action_value
            # Check if we can stop
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, V


def env_description(env, policy, v):
    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), (env.nrow, env.ncol)))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape((env.nrow, env.ncol)))
    print("")


def env_run(env, n_episodes, policy, versbose=False):
    rewards = []
    for ep in range(n_episodes):
        done = False
        epoch_reward = 0
        s = env.reset()
        while not done:
            if versbose:
                env.render()
            action = np.argmax(policy[s])
            s, reward, done, info = env.step(action)
            epoch_reward += reward
        rewards.append(epoch_reward)
    return rewards


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument('--env',
                        type=str,
                        default='Taxi-v1',
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
    parser.add_argument('--api_key',
                        type=str,
                        default=None)

    args, _ = parser.parse_known_args()
    return args


def run(env, n_episodes, discount_factor, verbose=False, api_key=None):
    env_name = env
    env = gym.make(env)
    if api_key is not None:
        env = gym.wrappers.Monitor(env, "/tmp/" + env_name, force=True)
    policy, v = value_iteration(env, discount_factor=discount_factor)
    if verbose:
        try:
            env_description(env, policy, v)
        except:
            print("Sorry, something go wrong.")
    rewards = env_run(env, n_episodes, policy, verbose)
    print("Avg rewards over {} episodes: {:.4f} +/-{:.4f}".format(
        n_episodes, np.mean(rewards), np.std(rewards)))
    if api_key is not None:
        env.close()
        gym.upload("/tmp/" + env_name, api_key=api_key)


def main():
    args = _parse_args()
    run(args.env, args.num_episodes, args.gamma, args.verbose, args.api_key)


if __name__ == '__main__':
    main()
