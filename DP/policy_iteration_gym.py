#!/usr/bin/python

import gym
import argparse
import numpy as np


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment
        and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation
            one our value function change is less than theta for all states.
        discount_factor: lambda discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (
                        reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (policy, V).
            policy is the optimal policy,
            a matrix of shape [S, A] where each state s
            contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state...
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (
                        reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V


def env_description(env, policy, v):
    print("Policy Probability Distribution:")
    print(policy)
    print("")

    import pdb; pdb.set_trace()

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
        env.monitor.start("/tmp/" + env_name, force=True)
    policy, v = policy_improvement(env, discount_factor=discount_factor)
    if verbose:
        try:
            env_description(env, policy, v)
        except:
            print("Sorry, something go wrong.")
    rewards = env_run(env, n_episodes, policy, verbose)
    print("Avg rewards over {} episodes: {:.4f} +/-{:.4f}".format(
        n_episodes, np.mean(rewards), np.std(rewards)))
    if api_key is not None:
        env.monitor.close()
        gym.upload("/tmp/" + env_name, api_key=api_key)


def main():
    args = _parse_args()
    run(args.env, args.num_episodes, args.gamma, args.verbose, args.api_key)


if __name__ == '__main__':
    main()
