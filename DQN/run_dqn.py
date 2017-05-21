import numpy as np
import argparse
from tqdm import trange

from rstools.utils.batch_utils import iterate_minibatches

from agent_networks import copy_model_parameters
from wrappers import Transition
from run_wrapper import typical_args, typical_argsparse, run_wrapper, update_wraper, \
    epsilon_greedy_policy, play_session

from DQN.dqn import DqnAgent
from DQN.dqrn import DqrnAgent


def update(sess, agent, transitions,
           discount_factor=0.99, reward_norm=1.0, batch_size=32):
    loss = 0.0

    transitions_it = zip(
            iterate_minibatches(transitions.state, batch_size),
            iterate_minibatches(transitions.action, batch_size),
            iterate_minibatches(transitions.reward, batch_size),
            iterate_minibatches(transitions.next_state, batch_size),
            iterate_minibatches(transitions.done, batch_size))

    for states, actions, rewards, next_states, dones in transitions_it:
        values_next = agent.predict(sess, next_states)
        td_target = rewards * reward_norm + \
                    np.invert(dones).astype(np.float32) * \
                    discount_factor * values_next.max(axis=1)

        batch_loss, _, _, _ = sess.run(
            [agent.qvalue_net.loss,
             agent.qvalue_net.train_op, agent.hidden_state.train_op, agent.feature_net.train_op],
            feed_dict={
                agent.feature_net.states: states,
                agent.feature_net.is_training: True,
                agent.qvalue_net.actions: actions,
                agent.qvalue_net.td_target: td_target,
                agent.qvalue_net.is_training: True,
            })
        loss += batch_loss
    return loss


# @TODO: rewrite for DqrnAgent too
def generate_sessions(sess, agent, env_pool, t_max=1000, update_fn=None):
    total_reward = 0.0
    total_qvalue_loss = 0, 0
    total_games = 0.0

    # init_state = None
    # if hasattr(agent, "get_belief_state"):
    #     init_state = agent.get_belief_state(sess)

    states = env_pool.pool_states()
    for t in range(t_max):
        actions = epsilon_greedy_policy(agent, sess, states)
        next_states, rewards, dones, _ = env_pool.step(actions)

        if update_fn is not None:
            transition = Transition(
                state=states, action=actions, reward=rewards, next_state=next_states, done=dones)
            total_qvalue_loss += update_fn(sess, agent, transition)

        states = next_states

        total_reward += rewards.mean()
        total_games += dones.sum()

    return total_reward, total_qvalue_loss, total_games


# @TODO: add target network support
def dqn_learning(sess, agent, env, update_fn, n_epochs=1000, n_sessions=100, t_max=1000):
    tr = trange(
        n_epochs,
        desc="",
        leave=True)

    history = {
        "reward": np.zeros(n_epochs),
        "qvalue_loss": np.zeros(n_epochs),
        "steps": np.zeros(n_epochs)}

    for i in tr:
        sessions = [
            generate_sessions(sess, agent, env, t_max, update_fn=update_fn)
            for _ in range(n_sessions)]
        session_rewards, session_qvalue_loss, session_steps = \
            map(np.array, zip(*sessions))

        history["reward"][i] = np.mean(session_rewards)
        history["qvalue_loss"][i] = np.mean(session_qvalue_loss)
        history["steps"][i] = np.mean(session_steps)

        desc = "\t".join(
            ["{} = {:.3f}".format(key, value[i]) for key, value in history.items()])
        tr.set_description(desc)

    return history


def run(env_name, make_env_fn, agent_cls,
        run_args, update_args, agent_agrs,
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4):
    run_wrapper(
        1, dqn_learning, update_wraper(update, **update_args), play_session,
        env_name, make_env_fn, agent_cls,
        run_args, agent_agrs,
        plot_stats, api_key,
        load, gpu_option)


def _parse_args():
    parser = argparse.ArgumentParser(description='DQN Agent Learning')
    # typical params
    parser.add_argument(
        '--agent',
        type=str,
        choices=["dqn", "dqrn"])

    parser = typical_args(parser)

    # special optimization params
    parser.add_argument(
        '--qvalue_lr',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--value_lr',
        type=float,
        default=1e-4)

    # agent special params & optimization
    parser.add_argument(
        '--target_dqn',
        action='store_true',
        default=False)
    parser.add_argument(
        '--double_dqn',
        action='store_true',
        default=False)
    parser.add_argument(
        '--advantage_dqn',
        action='store_true',
        default=False)

    args = parser.parse_known_args()
    return args


def main():
    args = _parse_args()

    network, run_args, update_args, optimization_params, make_env_fn = typical_argsparse(args)

    qvalue_optimization_params = {
        **optimization_params,
        **{"initial_lr": args.policy_lr}
    }

    value_optimization_params = {
        **optimization_params,
        **{"initial_lr": args.policy_lr}
    }

    agent_args = {
        "network": network,
        "feature_net_optimization": optimization_params,
        "hidden_state_optimization": optimization_params,
        "value_net_optimiaztion": value_optimization_params,
        "qvalue_net_optimiaztion": qvalue_optimization_params,
    }

    agent_cls = DqnAgent if args.agent_type == "dqn" else DqrnAgent

    run(args.env, make_env_fn, agent_cls,
        run_args, update_args, agent_args,
        args.plot_history, args.api_key,
        args.load, args.gpu_option)


if __name__ == '__main__':
    main()
