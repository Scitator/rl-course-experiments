import numpy as np
import gym
import argparse
import string
import os
import tensorflow as tf
from tqdm import trange

from rstools.utils.batch_utils import iterate_minibatches
from rstools.utils.os_utils import save_history, save_model
from rstools.visualization.plotter import plot_all_metrics

from networks import activations, networks, network_wrapper
from agent_networks import epsilon_greedy_policy
from wrappers import Transition, make_env, make_image_env_wrapper
from A3C.a3c_ff import  A3CFFAgent
from A3C.a3c_lstm import A3CLstmAgent


def update(sess, aac_agent, transitions, initial_state=None,
           discount_factor=0.99, reward_norm=1.0,
           batch_size=32, time_major=True):
    policy_targets = []
    value_targets = []
    state_history = []
    action_history = []
    done_history = []

    cumulative_reward = np.zeros_like(transitions[-1].reward) + \
                        np.invert(transitions[-1].done) * \
                        aac_agent.predict_value(sess, transitions[-1].next_state)
    for transition in reversed(transitions):
        cumulative_reward = reward_norm * transition.reward + \
                            np.invert(transition.done) * discount_factor * cumulative_reward
        policy_target = cumulative_reward - aac_agent.predict_value(sess, transition.state)

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

    if not time_major:
        state_history = state_history.swapaxes(0, 1)
        action_history = action_history.swapaxes(0, 1)
        done_history = done_history.swapaxes(0, 1)
        value_targets = value_targets.swapaxes(0, 1)
        policy_targets = policy_targets.swapaxes(0, 1)

    time_len = state_history.shape[0]
    if initial_state is not None:
        aac_agent.assign_belief_state(sess, initial_state)

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

        for state_batch, action_batch, value_target, policy_target, done_batch in \
                zip(state_axis, action_axis, value_target_axis, policy_target_axis, done_axis):
            batch_loss_policy, batch_loss_state, _, _, _, _ = sess.run(
                [aac_agent.policy_net.loss,
                 aac_agent.state_value_net.loss,
                 aac_agent.policy_net.train_op,
                 aac_agent.state_value_net.train_op,
                 aac_agent.feature_net.train_op,
                 aac_agent.belief_update],
                feed_dict={
                    aac_agent.feature_net.states: state_batch,
                    aac_agent.feature_net.is_training: True,
                    aac_agent.policy_net.actions: action_batch,
                    aac_agent.policy_net.cumulative_rewards: policy_target,
                    aac_agent.policy_net.is_training: True,
                    aac_agent.state_value_net.td_target: value_target,
                    aac_agent.state_value_net.is_training: True,
                    aac_agent.is_end: done_batch
                })

            axis_value_loss += batch_loss_state
            axis_policy_loss += batch_loss_policy

        policy_loss += axis_policy_loss / axis_len
        value_loss += axis_value_loss / axis_len

    return policy_loss / time_len, value_loss / time_len


def update_wraper(discount_factor=0.99, reward_norm=1.0, batch_size=32, time_major=False):
    def wrapper(sess, a3c_agent, transitions, init_state=None):
        return update(
            sess, a3c_agent, transitions, init_state,
            discount_factor=discount_factor, reward_norm=reward_norm,
            batch_size=batch_size, time_major=time_major)

    return wrapper


# def generate_session(
#         sess, aac_agent, env, t_max=1000, update_fn=None):
#     total_reward = 0
#     total_policy_loss, total_state_loss = 0, 0
#
#     transitions = []
#
#     s = env.reset()
#     for t in range(t_max):
#         a = epsilon_greedy_policy(aac_agent, sess, np.array([s], dtype=np.float32))[0]
#
#         next_s, r, done, _ = env.step(a)
#
#         transitions.append(Transition(
#             state=s, action=a, reward=r, next_state=next_s, done=done))
#
#         total_reward += r
#
#         s = next_s
#         if done:
#             break
#
#     if update_fn is not None:
#         total_policy_loss, total_value_loss = update_fn(sess, aac_agent, transitions)
#
#     return total_reward, total_policy_loss, total_state_loss, t

def play_session(
        sess, a3c_agent, env, t_max=1000):
    total_reward = 0
    total_policy_loss, total_state_loss = 0, 0

    transitions = []

    s = env.reset()
    for t in range(t_max):
        a = epsilon_greedy_policy(a3c_agent, sess, np.array([s], dtype=np.float32))[0]

        next_s, r, done, _ = env.step(a)

        transitions.append(Transition(
            state=s, action=a, reward=r, next_state=next_s, done=done))

        total_reward += r
        a3c_agent.update_belief_state(sess, [s], [done])
        s = next_s
        if done:
            break

    return total_reward, total_policy_loss, total_state_loss, t


def generate_sessions(sess, a3c_agent, env_pool, t_max=1000, update_fn=None):
    total_reward = 0.0
    total_policy_loss, total_value_loss = 0, 0
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
        states = next_states

        total_reward += rewards.mean()
        total_games += dones.sum()

    if update_fn is not None:
        total_policy_loss, total_value_loss = update_fn(sess, a3c_agent, transitions, init_state)

    return total_reward, total_policy_loss, total_value_loss, total_games


def a3c_learning(
        sess, a3c_agent, env, update_fn,
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
            generate_sessions(sess, a3c_agent, env, t_max, update_fn=update_fn)
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
        q_learning_args, update_args, agent_agrs,
        n_games,
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4):
    env = make_env_fn(env_name, n_games)

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    network = agent_agrs["network"]

    agent = agent_cls(
        state_shape, n_actions, network,
        special=agent_agrs)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.Saver()
        model_dir = "./logs_" + env_name.replace(string.punctuation, "_")

        if not load:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "{}/model.ckpt".format(model_dir))

        try:
            history = a3c_learning(
                sess, agent, env,
                update_fn=update_wraper(**update_args),
                **q_learning_args)
        except KeyboardInterrupt:
            print("Exiting training procedure")
        save_model(sess, saver, model_dir)

        if plot_stats:
            tf.reset_default_graph()
            save_history(history, model_dir)
            plotter_dir = os.path.join(model_dir, "plotter")
            plot_all_metrics(history, save_dir=plotter_dir)

        if api_key is not None:
            env_name = env_name.replace("Deterministic", "")
            env = make_env_fn(env_name, 1)
            monitor_dir = os.path.join(model_dir, "monitor")
            env = gym.wrappers.Monitor(env, monitor_dir, force=True)
            sessions = [play_session(sess, agent, env, int(1e10))
                        for _ in range(300)]
            env.close()
            gym.upload(monitor_dir, api_key=api_key)


def _parse_args():
    parser = argparse.ArgumentParser(description='A3C Agent Learning')
    parser.add_argument(
        '--agent',
        type=str,
        choise=["feed_forward", "recurrent"])
    parser.add_argument(
        '--env',
        type=str,
        default='KungFuMasterDeterministic-v0',  # BreakoutDeterministic-v0
        help='The environment to use')

    parser.add_argument(
        '--n_epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--n_sessions',
        type=int,
        default=10)
    parser.add_argument(
        '--t_max',
        type=int,
        default=1000)
    parser.add_argument(
        '--plot_history',
        action='store_true',
        default=False)
    parser.add_argument(
        '--api_key',
        type=str,
        default=None)
    parser.add_argument(
        '--load',
        action='store_true',
        default=False)
    parser.add_argument(
        '--gpu_option',
        type=float,
        default=0.45)

    parser.add_argument(
        '--feature_network',
        type=str,
        choise=["linear", "convolution"])
    parser.add_argument(
        '--activation',
        type=str,
        default="elu")
    parser.add_argument(
        '--use_bn',
        action='store_true',
        default=False)
    parser.add_argument(
        '--dropout',
        type=float,
        default=-1)

    parser.add_argument(
        '--policy_lr',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--value_lr',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--feature_lr',
        type=float,
        default=1e-3)
    parser.add_argument(
        '--lr_decay_steps',
        type=float,
        default=1e5)
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.999)
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=10.0)

    parser.add_argument(
        '--entropy_koef',
        type=float,
        default=1e-2)

    parser.add_argument(
        '--n_games',
        type=int,
        default=10)

    # update args
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Gamma discount factor')
    parser.add_argument(
        '--reward_norm',
        type=float,
        default=1.0)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10)
    parser.add_argument(
        '--time_major',
        action='store_true',
        default=False)

    # preprocess args for image envs
    parser.add_argument(
        '--image_width',
        type=float,
        default=64)
    parser.add_argument(
        '--image_height',
        type=float,
        default=64)
    parser.add_argument(
        '--image_grayscale',
        action='store_true',
        default=False)
    parser.add_argument(
        '--image_crop_x1',
        type=int,
        default=None)
    parser.add_argument(
        '--image_crop_x2',
        type=int,
        default=None)
    parser.add_argument(
        '--image_crop_y1',
        type=int,
        default=None)
    parser.add_argument(
        '--image_crop_y2',
        type=int,
        default=None)

    args, _ = parser.parse_known_args()
    return args


def main():
    args = _parse_args()

    network = network_wrapper(
        networks[args.feature_network], {
            "activation_fn": activations[args.activation],
            "use_bn": args.use_bn,
            "dropout": args.dropout
        })

    q_learning_args = {
        "n_epochs": args.n_epochs,
        "n_sessions": args.n_sessions,
        "t_max": args.t_max
    }
    update_args = {
        "discount_factor": args.gamma,
        "reward_norm": args.reward_norm,
        "batch_size": args.batch_size,
        "time_major": args.time_major
    }
    optimization_params = {
        "initial_lr": args.feature_lr,
        "decay_steps": args.lr_decay_steps,
        "lr_decay": args.lr_decay,
        "grad_clip": args.grad_clip
    }
    policy_optimization_params = {
        **optimization_params,
        **{"initial_lr": args.policy_lr}
    }
    value_optimization_params = {
        **optimization_params,
        **{"initial_lr": args.policy_lr}
    }
    policy_net_params = {
        "entropy_koef": args.entropy_koef
    }
    agent_args = {
        "n_games": args.n_games,
        "network": network,
        "policy_net": policy_net_params,
        "feature_net_optimization": optimization_params,
        "state_value_net_optimiaztion": value_optimization_params,
        "policy_net_optimiaztion": policy_optimization_params,
        "cell_activation": args.lstm_activation
    }
    image_preprocessing_params = {
        "width": args.image_width,
        "height": args.image_height,
        "grayscale": args.image_grayscale,
        "crop": lambda img: img[
                            args.image_crop_x1:args.image_crop_x1,
                            args.image_crop_y1:args.image_crop_y1]
    }

    agent_cls = A3CFFAgent if args.agent == "feed_forward" else A3CLstmAgent
    make_env_fn = make_env \
        if args.feature_network == "linear" \
        else make_image_env_wrapper(image_preprocessing_params)

    run(args.env, make_env_fn, agent_cls,
        q_learning_args, update_args, agent_args,
        args.n_games,
        args.plot_history, args.api_key,
        args.load, args.gpu_option)
