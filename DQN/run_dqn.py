import argparse
import numpy as np
from tqdm import trange
from tensorflow.contrib import rnn

from rstools.utils.batch_utils import iterate_minibatches

from DQN.dqn import DqnAgent
from DQN.dqrn import DqrnAgent
from agents.agent_networks import copy_model_parameters
from agents.networks import activations
from wrappers.gym_wrappers import Transition
from wrappers.run_wrappers import typical_args, typical_argsparse, run_wrapper, update_wraper, \
    epsilon_greedy_actions, play_session


def update(sess, agent, target_agent, transitions, init_state=None,
           discount_factor=0.99, reward_norm=1.0, batch_size=32, time_major=False):
    loss = 0.0
    time_len = transitions.state.shape[0]

    transitions_it = zip(
        iterate_minibatches(transitions.state, batch_size),
        iterate_minibatches(transitions.action, batch_size),
        iterate_minibatches(transitions.reward, batch_size),
        iterate_minibatches(transitions.next_state, batch_size),
        iterate_minibatches(transitions.done, batch_size))

    for states, actions, rewards, next_states, dones in transitions_it:
        qvalues_next = agent.predict_qvalues(sess, next_states)
        best_actions = qvalues_next.argmax(axis=1)
        qvalues_next_target = target_agent.predict_qvalues(sess, next_states)
        qvalues_next_target = qvalues_next_target[np.arange(batch_size), best_actions]

        td_target = rewards * reward_norm + \
                    np.invert(dones).astype(np.float32) * \
                    discount_factor * qvalues_next_target

        run_dict = [
            agent.qvalue_net.loss,
            agent.qvalue_net.train_op, agent.hidden_state.train_op, agent.feature_net.train_op
        ]

        feed_dict = {
            agent.feature_net.states: states,
            agent.feature_net.is_training: True,
            agent.qvalue_net.actions: actions,
            agent.qvalue_net.td_target: td_target,
            agent.qvalue_net.is_training: True,
        }

        if agent.special.get("dueling_network", False):
            run_dict[0] = agent.agent_loss
            run_dict += [agent.value_net.train_op]
            feed_dict[agent.value_net.td_target] = td_target  # @TODO: why need to feed?
            feed_dict[agent.value_net.is_training] = True

        run_results = sess.run(
            run_dict,
            feed_dict=feed_dict)

        batch_loss = run_results[0]
        loss += batch_loss
    return loss / time_len


def generate_sessions(
        sess, agent, target_agent, env_pool,
        t_max=1000, epsilon=0.01,
        update_fn=None):
    total_reward = 0.0
    total_qvalue_loss = 0.0
    total_games = 0.0

    states = env_pool.pool_states()
    for t in range(t_max):
        actions = epsilon_greedy_actions(agent, sess, states, epsilon=epsilon)
        next_states, rewards, dones, _ = env_pool.step(actions)

        if update_fn is not None:
            transition = Transition(
                state=states, action=actions, reward=rewards, next_state=next_states, done=dones)
            total_qvalue_loss += update_fn(sess, agent, target_agent, transition)

        if isinstance(agent, DqrnAgent):
            agent.update_belief_state(sess, states, dones)

        states = next_states

        total_reward += rewards.mean()
        total_games += dones.sum()

    return total_reward, total_qvalue_loss, total_games


def dqn_learning(
        sess, agent, env, update_fn,
        n_epochs=1000, n_sessions=100, t_max=1000,
        initial_epsilon=0.5, final_epsilon=0.01,
        target_dqn=False, copy_n_epoch=5):
    tr = trange(
        n_epochs,
        desc="",
        leave=True)

    if target_dqn:
        agent, target_agent = agent
        # copy_model_parameters(sess, agent, target_agent)
    else:
        target_agent = agent

    history = {
        "reward": np.zeros(n_epochs),
        "qvalue_loss": np.zeros(n_epochs),
        "steps": np.zeros(n_epochs),
        "epsilon": np.zeros(n_epochs)
    }

    epsilon = initial_epsilon
    n_epochs_decay = n_epochs * 0.8

    for i in tr:
        sessions = [
            generate_sessions(
                sess, agent, target_agent, env, t_max, epsilon=epsilon, update_fn=update_fn)
            for _ in range(n_sessions)]
        session_rewards, session_qvalue_loss, session_steps = map(np.array, zip(*sessions))

        history["reward"][i] = np.mean(session_rewards)
        history["qvalue_loss"][i] = np.mean(session_qvalue_loss)
        history["steps"][i] = np.mean(session_steps)
        history["epsilon"][i] = epsilon

        if i < n_epochs_decay:
            epsilon -= (initial_epsilon - final_epsilon) / float(n_epochs_decay)

        if target_dqn and (i + 1) % copy_n_epoch == 0:
            copy_model_parameters(sess, agent, target_agent)

        desc = "\t".join(
            ["{} = {:.3f}".format(key, value[i]) for key, value in history.items()])
        tr.set_description(desc)

    if target_dqn:
        copy_model_parameters(sess, agent, target_agent)

    return history


def run(env_name, make_env_fn, agent_cls,
        run_args, update_args, agent_agrs,
        log_dir=None,
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4,
        n_games=10,
        target_dqn=False):
    run_wrapper(
        n_games, dqn_learning, update_wraper(update, **update_args),
        play_session, epsilon_greedy_actions,
        env_name, make_env_fn, agent_cls,
        run_args, agent_agrs,
        log_dir=log_dir,
        plot_stats=plot_stats, api_key=api_key,
        load=load, gpu_option=gpu_option,
        use_target_network=target_dqn)


def _parse_args():
    parser = argparse.ArgumentParser(description='DQN Agent Learning')
    # typical params
    parser.add_argument(
        '--agent',
        type=str,
        choices=["dqn", "dqrn"])

    # special exploration params
    parser.add_argument(
        '--initial_epsilon',
        type=float,
        default=0.5)
    parser.add_argument(
        '--final_epsilon',
        type=float,
        default=0.01)

    parser.add_argument(
        '--copy_n_epoch',
        type=int,
        default=5)

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
        '--dueling_dqn',
        action='store_true',
        default=False)

    parser = typical_args(parser)

    args = parser.parse_args()
    return args


# @TODO: where is epsilon?!
def main():
    args = _parse_args()

    network, run_args, update_args, optimization_params, make_env_fn = typical_argsparse(args)

    special_run_args = {
        "target_dqn": args.target_dqn,
        "initial_epsilon": args.initial_epsilon,
        "final_epsilon": args.final_epsilon,
        "copy_n_epoch": args.copy_n_epoch
    }

    run_args = {
        **run_args, **special_run_args}

    qvalue_optimization_params = {
        **optimization_params,
        **{"initial_lr": args.qvalue_lr}
    }

    value_optimization_params = {
        **optimization_params,
        **{"initial_lr": args.value_lr}
    }

    agent_cls = DqnAgent if args.agent == "dqn" else DqrnAgent

    special = {
        "dueling_network": args.dueling_dqn,
        "hidden_size": args.hidden_size,
        "hidden_activation": activations[args.hidden_activation],
        "feature_net_optimization": optimization_params,
        "hidden_state_optimization": optimization_params,
        "value_net_optimiaztion": value_optimization_params,
        "qvalue_net_optimiaztion": qvalue_optimization_params,
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
        args.n_games,
        args.target_dqn)


if __name__ == '__main__':
    main()
