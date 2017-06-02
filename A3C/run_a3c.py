import argparse

import numpy as np
from rstools.utils.batch_utils import iterate_minibatches, merge_generators
from tqdm import trange

from A3C.a3c_ff import A3CFFAgent
from A3C.a3c_lstm import A3CLstmAgent
from agents.networks import activations
from wrappers.gym_wrappers import Transition
from wrappers.run_wrappers import typical_args, typical_argsparse, run_wrapper, update_wraper, \
    epsilon_greedy_policy, play_session


def update(sess, a3c_agent, transitions, initial_state=None,
           discount_factor=0.99, reward_norm=1.0,
           batch_size=32, time_major=True):
    policy_targets = []
    value_targets = []
    state_history = []
    action_history = []
    done_history = []

    cumulative_reward = np.zeros_like(transitions[-1].reward) + \
                        np.invert(transitions[-1].done) * \
                        a3c_agent.predict_values(sess, transitions[-1].next_state)
    for transition in reversed(transitions):
        cumulative_reward = reward_norm * transition.reward + \
                            np.invert(transition.done) * discount_factor * cumulative_reward
        policy_target = cumulative_reward - a3c_agent.predict_values(sess, transition.state)

        value_targets.append(cumulative_reward)
        policy_targets.append(policy_target)
        state_history.append(transition.state)
        action_history.append(transition.action)
        done_history.append(transition.done)

    value_targets = np.array(value_targets[::-1])  # time-major
    policy_targets = np.array(policy_targets[::-1])
    state_history = np.array(state_history[::-1])
    action_history = np.array(action_history[::-1])
    done_history = np.array(done_history[::-1])

    # if not time_major:
    #     assert not isinstance(a3c_agent, A3CLstmAgent), "Please, use time_major flag for updates"
    #
    #     state_history = state_history.swapaxes(0, 1)
    #     action_history = action_history.swapaxes(0, 1)
    #     done_history = done_history.swapaxes(0, 1)
    #     value_targets = value_targets.swapaxes(0, 1)
    #     policy_targets = policy_targets.swapaxes(0, 1)

    if isinstance(a3c_agent, A3CLstmAgent):
        a3c_agent.assign_belief_state(sess, initial_state)

    time_len = state_history.shape[0]
    value_loss, policy_loss = 0.0, 0.0
    for state_axis, action_axis, value_target_axis, policy_target_axis, done_axis in \
            zip(state_history, action_history, value_targets, policy_targets, done_history):
        axis_len = state_axis.shape[0]
        axis_value_loss, axis_policy_loss = 0.0, 0.0

        state_axis = iterate_minibatches(state_axis, batch_size)
        action_axis = iterate_minibatches(action_axis, batch_size)
        value_target_axis = iterate_minibatches(value_target_axis, batch_size)
        policy_target_axis = iterate_minibatches(policy_target_axis, batch_size)
        done_axis = iterate_minibatches(done_axis, batch_size)

        batch_generator = merge_generators(
            [state_axis, action_axis, value_target_axis, policy_target_axis, done_axis])

        for state_batch, action_batch, value_target, policy_target, done_batch in batch_generator:
            run_params = [
                 a3c_agent.policy_net.loss,
                 a3c_agent.value_net.loss,
                 a3c_agent.policy_net.train_op,
                 a3c_agent.value_net.train_op,
                 a3c_agent.feature_net.train_op]
            feed_params = {
                a3c_agent.feature_net.states: state_batch,
                a3c_agent.feature_net.is_training: True,
                a3c_agent.policy_net.actions: action_batch,
                a3c_agent.policy_net.cumulative_rewards: policy_target,
                a3c_agent.policy_net.is_training: True,
                a3c_agent.value_net.td_target: value_target,
                a3c_agent.value_net.is_training: True
            }

            if isinstance(a3c_agent, A3CLstmAgent):
                run_params += [a3c_agent.hidden_state.belief_update]
                feed_params[a3c_agent.hidden_state.is_end] = done_batch

            run_result = sess.run(
                run_params,
                feed_dict=feed_params)

            batch_loss_policy = run_result[0]
            batch_loss_state = run_result[1]

            axis_value_loss += batch_loss_state
            axis_policy_loss += batch_loss_policy

        policy_loss += axis_policy_loss / axis_len
        value_loss += axis_value_loss / axis_len

    return policy_loss / time_len, value_loss / time_len


def generate_sessions(sess, a3c_agent, env_pool, update_fn, t_max=1000):
    total_reward = 0.0
    total_games = 0.0

    transitions = []
    init_state = None
    if isinstance(a3c_agent, A3CLstmAgent):
        init_state = a3c_agent.get_belief_state(sess)

    states = env_pool.pool_states()
    for t in range(t_max):
        actions = epsilon_greedy_policy(a3c_agent, sess, states)
        next_states, rewards, dones, _ = env_pool.step(actions)

        transitions.append(Transition(
            state=states, action=actions, reward=rewards, next_state=next_states, done=dones))

        if isinstance(a3c_agent, A3CLstmAgent):
            a3c_agent.update_belief_state(sess, states, dones)

        states = next_states

        total_reward += rewards.mean()
        total_games += dones.sum()

    total_policy_loss, total_value_loss = update_fn(sess, a3c_agent, transitions, init_state)

    return total_reward, total_policy_loss, total_value_loss, total_games


def a3c_learning(
        sess, agent, env, update_fn,
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
            generate_sessions(sess, agent, env, update_fn, t_max)
            for _ in range(n_sessions)]
        session_rewards, session_policy_loss, session_value_loss, session_steps = \
            map(np.array, zip(*sessions))

        history["reward"][i] = np.mean(session_rewards)
        history["policy_loss"][i] = np.mean(session_policy_loss)
        history["value_loss"][i] = np.mean(session_value_loss)
        history["steps"][i] = np.mean(session_steps)

        desc = "\t".join(
            ["{} = {:.3f}".format(key, value[i]) for key, value in history.items()])
        tr.set_description(desc)

    return history


def run(env_name, make_env_fn, agent_cls,
        run_args, update_args, agent_agrs,
        log_dir=None, episode_limit=None,
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4,
        n_games=10):
    run_wrapper(
        n_games, a3c_learning, update_wraper(update, **update_args),
        play_session, epsilon_greedy_policy,
        env_name, make_env_fn, agent_cls,
        run_args, agent_agrs,
        log_dir=log_dir, episode_limit=episode_limit,
        plot_stats=plot_stats, api_key=api_key,
        load=load, gpu_option=gpu_option)


def _parse_args():
    parser = argparse.ArgumentParser(description='A3C Agent Learning')
    # typical params
    parser.add_argument(
        '--agent',
        type=str,
        default="feed_forward",
        choices=["feed_forward", "recurrent"],
        help='Which agent to use. (default: %(default)s)')

    parser = typical_args(parser)

    # agent special params & optimization
    parser.add_argument(
        '--policy_lr',
        type=float,
        default=1e-5,
        help='Learning rate for policy network. (default: %(default)s)')
    parser.add_argument(
        '--value_lr',
        type=float,
        default=1e-5,
        help='Learning rate for value network. (default: %(default)s)')

    parser.add_argument(
        '--entropy_factor',
        type=float,
        default=1e-2,
        help='Entropy factor for policy network. (default: %(default)s)')

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    assert args.time_major, "Please, use time_major flag for updates"

    network, run_args, update_args, optimization_params, make_env_fn = typical_argsparse(args)

    policy_optimization_params = {
        **optimization_params,
        **{"initial_lr": args.policy_lr}
    }

    value_optimization_params = {
        **optimization_params,
        **{"initial_lr": args.value_lr}
    }
    policy_net_params = {
        "entropy_factor": args.entropy_factor
    }

    agent_cls = A3CFFAgent if args.agent == "feed_forward" else A3CLstmAgent

    special = {
        "policy_net": policy_net_params,
        "hidden_size": args.hidden_size,
        "hidden_activation": activations[args.hidden_activation],
        "feature_net_optimization": optimization_params,
        "hidden_state_optimization": optimization_params,
        "value_net_optimization": value_optimization_params,
        "policy_net_optimization": policy_optimization_params,
    }

    agent_args = {
        "network": network,
        "special": special
    }

    run(args.env, make_env_fn, agent_cls,
        run_args, update_args, agent_args,
        args.log_dir, args.episode_limit,
        args.plot_history, args.api_key,
        args.load, args.gpu_option,
        args.n_games)


if __name__ == '__main__':
    main()
