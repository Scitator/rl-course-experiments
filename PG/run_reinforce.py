import argparse
import numpy as np
from tqdm import trange

from rstools.utils.batch_utils import iterate_minibatches

from PG.reinforce import ReinforceAgent
from agents.networks import activations
from wrappers.gym_wrappers import Transition
from wrappers.run_wrappers import typical_args, typical_argsparse, run_wrapper, update_wraper, \
    epsilon_greedy_policy, play_session


def update(sess, reinforce_agent, transitions, initial_state=None,
           discount_factor=0.99, reward_norm=1.0, batch_size=32, time_major=True):
    policy_targets = []
    state_history = []
    action_history = []

    cumulative_reward = np.zeros_like(transitions[-1].reward)
    for transition in reversed(transitions):
        cumulative_reward = reward_norm * transition.reward + \
                            np.invert(transition.done) * discount_factor * cumulative_reward

        policy_targets.append(cumulative_reward)
        state_history.append(transition.state)
        action_history.append(transition.action)

    # time-major
    policy_targets = np.array(policy_targets[::-1])
    state_history = np.array(state_history[::-1])
    action_history = np.array(action_history[::-1])

    # if not time_major:
    #     state_history = state_history.swapaxes(0, 1)
    #     action_history = action_history.swapaxes(0, 1)
    #     policy_targets = policy_targets.swapaxes(0, 1)

    time_len = state_history.shape[0]

    policy_loss = 0.0
    for state_axis, action_axis, policy_target_axis in \
            zip(state_history, action_history, policy_targets):
        axis_len = state_axis.shape[0]
        axis_policy_loss = 0.0

        state_axis = iterate_minibatches(state_axis, batch_size)
        action_axis = iterate_minibatches(action_axis, batch_size)
        policy_target_axis = iterate_minibatches(policy_target_axis, batch_size)

        for state_batch, action_batch, policy_target in \
                zip(state_axis, action_axis, policy_target_axis):
            run_params = [
                reinforce_agent.policy_net.loss,
                reinforce_agent.policy_net.train_op,
                reinforce_agent.feature_net.train_op]
            feed_params = {
                reinforce_agent.feature_net.states: state_batch,
                reinforce_agent.feature_net.is_training: True,
                reinforce_agent.policy_net.actions: action_batch,
                reinforce_agent.policy_net.cumulative_rewards: policy_target,
                reinforce_agent.policy_net.is_training: True
            }

            run_result = sess.run(
                run_params,
                feed_dict=feed_params)

            batch_loss_policy = run_result[0]

            axis_policy_loss += batch_loss_policy

        policy_loss += axis_policy_loss / axis_len

    return policy_loss / time_len


def generate_sessions(sess, a3c_agent, env_pool, update_fn, t_max=1000):
    total_reward = 0.0
    total_games = 0.0

    transitions = []

    states = env_pool.pool_states()
    for t in range(t_max):
        actions = epsilon_greedy_policy(a3c_agent, sess, states)
        next_states, rewards, dones, _ = env_pool.step(actions)

        transitions.append(Transition(
            state=states, action=actions, reward=rewards, next_state=next_states, done=dones))
        states = next_states

        total_reward += rewards.mean()
        total_games += dones.sum()

    total_policy_loss = update_fn(sess, a3c_agent, transitions)

    return total_reward, total_policy_loss, total_games


def reinforce_learning(
        sess, agent, env, update_fn,
        n_epochs=1000, n_sessions=100, t_max=1000):
    tr = trange(
        n_epochs,
        desc="",
        leave=True)

    history = {
        "reward": np.zeros(n_epochs),
        "policy_loss": np.zeros(n_epochs),
        "steps": np.zeros(n_epochs),
    }

    for i in tr:
        sessions = [
            generate_sessions(sess, agent, env, update_fn, t_max)
            for _ in range(n_sessions)]
        session_rewards, session_policy_loss, session_steps = \
            map(np.array, zip(*sessions))

        history["reward"][i] = np.mean(session_rewards)
        history["policy_loss"][i] = np.mean(session_policy_loss)
        history["steps"][i] = np.mean(session_steps)

        desc = "\t".join(
            ["{} = {:.3f}".format(key, value[i]) for key, value in history.items()])
        tr.set_description(desc)

    return history


def run(env_name, make_env_fn, agent_cls,
        run_args, update_args, agent_agrs,
        log_dir=None,
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4,
        n_games=10):
    run_wrapper(
        n_games, reinforce_learning, update_wraper(update, **update_args),
        play_session, epsilon_greedy_policy,
        env_name, make_env_fn, agent_cls,
        run_args, agent_agrs,
        log_dir=log_dir,
        plot_stats=plot_stats, api_key=api_key,
        load=load, gpu_option=gpu_option)


def _parse_args():
    parser = argparse.ArgumentParser(description='Reinforce Agent Learning')
    # typical params
    parser = typical_args(parser)

    # agent special params & optimization
    parser.add_argument(
        '--policy_lr',
        type=float,
        default=1e-5,
        help='Learning rate for policy network. (default: %(default)s)')

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

    policy_net_params = {
        "entropy_factor": args.entropy_factor
    }

    agent_cls = ReinforceAgent

    special = {
        "policy_net": policy_net_params,
        "hidden_size": args.hidden_size,
        "hidden_activation": activations[args.hidden_activation],
        "feature_net_optimization": optimization_params,
        "hidden_state_optimization": optimization_params,
        "policy_net_optimization": policy_optimization_params,
    }

    agent_args = {
        "network": network,
        "special": special
    }

    run(args.env, make_env_fn, agent_cls,
        run_args, update_args, agent_args,
        args.log_dir,
        args.plot_history, args.api_key,
        args.load, args.gpu_option,
        args.n_games)


if __name__ == '__main__':
    main()
