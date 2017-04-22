#!/usr/bin/python

import gym
from gym import wrappers
import argparse
import numpy as np
import random
from tqdm import trange


def get_random_policy(env):
    """
    Build a numpy array representing agent policy.
    This array must have one element per each of 16 environment states.
    Element must be an integer from 0 to 3, representing action
    to take from that state.
    """
    return np.random.randint(0, int(env.action_space.n), int(env.observation_space.n))


def sample_reward(env, policy, t_max=100):
    """
    Interact with an environment, return sum of all rewards.
    If game doesn't end on t_max (e.g. agent walks into a wall),
    force end the game and return whatever reward you got so far.
    Tip: see signature of env.step(...) method above.
    """
    s = env.reset()
    total_reward = 0

    for _ in range(t_max):
        action = policy[s]
        s, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def evaluate(sample_func, env, policy, n_times=100):
    """Run several evaluations and average the score the policy gets."""
    rewards = [sample_func(env, policy) for _ in range(n_times)]
    return float(np.mean(rewards))


def crossover(env, policy1, policy2, p=0.5, prioritize_func=None):
    """
    for each state, with probability p take action from policy1, else policy2
    """
    if prioritize_func is not None:
        p = prioritize_func(env, policy1, policy2, p)
    return np.choose(
        (np.random.random_sample(policy1.shape[0]) <= p).astype(int), [policy1, policy2])


def mutation(env, policy, p=0.1):
    """
    for each state, with probability p replace action with random action
    Tip: mutation can be written as crossover with random policy
    """
    return crossover(env, get_random_policy(env), policy, p)


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
    parser.add_argument(
        '--env',
        type=str,
        default='FrozenLake8x8-v0',
        help='The environment to use')
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=200,
        help='Number of episodes')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=200,
        help='Max number per episode')
    parser.add_argument(
        '--pool_size',
        type=int,
        default=200,
        help='Population size')
    parser.add_argument(
        '--n_crossovers',
        type=int,
        default=100,
        help='Number of crossovers per episode')
    parser.add_argument(
        '--n_mutations',
        type=int,
        default=100,
        help='Number of mutations per episode')
    parser.add_argument(
        '--seed',
        type=int,
        default=42)
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False)
    parser.add_argument(
        '--api_key',
        type=str,
        default=None)

    args, _ = parser.parse_known_args()
    return args


def run(env, n_episodes, max_steps,
        pool_size, n_crossovers, n_mutations,
        seed=42, verbose=False, api_key=None):
    random.seed(seed)
    np.random.seed(seed)

    env_name = env
    env = gym.make(env)
    env.reset()

    if api_key is not None:
        env = gym.wrappers.Monitor(env, "/tmp/" + env_name, force=True)

    if verbose:
        print("initializing...")
    pool = [get_random_policy(env) for _ in range(pool_size)]

    rewards = np.zeros(n_episodes)

    tr = trange(
        n_episodes,
        desc="best score: {:.4}".format(0.0),
        leave=True)

    def sample_func(env, policy):
        return sample_reward(
            env, policy, t_max=max_steps if api_key is None else int(1e10))

    def prioritize_func(env, policy1, policy2, p):
        return min(
            p * evaluate(sample_func, env, policy1) / (evaluate(sample_func, env, policy2) + 0.001),
            1.0)

    for i_epoch in tr:
        crossovered = [
            crossover(env, random.choice(pool), random.choice(pool),
                      prioritize_func=prioritize_func)
            for _ in range(n_crossovers)]
        mutated = [mutation(env, random.choice(pool)) for _ in range(n_mutations)]

        assert type(crossovered) == type(mutated) == list

        # add new policies to the pool
        pool = pool + crossovered + mutated
        pool_scores = list(map(lambda x: evaluate(sample_func, env, x), pool))

        # select pool_size best policies
        selected_indices = np.argsort(pool_scores)[-pool_size:]
        pool = [pool[i] for i in selected_indices]
        pool_scores = [pool_scores[i] for i in selected_indices]

        # print the best policy so far (last in ascending score order)
        tr.set_description("best score: {:.4}".format(pool_scores[-1]))
        rewards[i_epoch] = pool_scores[-1]

    print("Avg rewards over {} episodes: {:.4f} +/-{:.4f}".format(
        n_episodes, np.mean(rewards), np.std(rewards)))
    if api_key is not None:
        env.close()
        gym.upload("/tmp/" + env_name, api_key=api_key)


def main():
    args = _parse_args()
    run(args.env, args.num_episodes, args.max_steps,
        args.pool_size, args.n_crossovers, args.n_mutations,
        args.seed, args.verbose, args.api_key)


if __name__ == '__main__':
    main()
