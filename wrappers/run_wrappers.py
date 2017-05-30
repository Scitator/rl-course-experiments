import os
import string

import gym
import numpy as np
import tensorflow as tf
from rstools.utils.os_utils import save_history, save_model
from rstools.visualization.plotter import plot_all_metrics

from agents.networks import activations, networks, network_wrapper, str2params
from wrappers.gym_wrappers import make_env, make_image_env, make_env_wrapper

try:
    import ppaquette_gym_doom
except ImportError:
    print("no doom envs")


def epsilon_greedy_policy(agent, sess, observations):
    probs = agent.predict_probs(sess, observations)
    actions = [np.random.choice(len(row), p=row) for row in probs]
    return actions


# @TODO: rewrite more numpy way (no for usage)
def epsilon_greedy_actions(agent, sess, observations, epsilon=0.01):
    qvalues = agent.predict_qvalues(sess, observations)
    probs = np.ones_like(qvalues, dtype=float) * epsilon / agent.qvalue_net.n_actions
    best_action = np.argmax(qvalues, axis=-1)
    for i, action in enumerate(best_action):
        probs[i, action] += (1.0 - epsilon)
    actions = [np.random.choice(len(row), p=row) for row in probs]
    return actions


def play_session(sess, agent, env, t_max=int(1e10), action_fn=None):
    total_reward = 0

    s = env.reset()
    for t in range(t_max):
        a = action_fn(agent, sess, np.array([s], dtype=np.float32))[0]
        next_s, r, done, _ = env.step(a)
        total_reward += r

        if hasattr(agent, "update_belief_state"):
            agent.update_belief_state(sess, [s], [done])

        s = next_s
        if done:
            break

    return total_reward, t


def update_wraper(
        update_fn,
        discount_factor=0.99, reward_norm=1.0, batch_size=32, time_major=False):
    def wrapper(sess, a3c_agent, transitions, init_state=None):
        return update_fn(
            sess, a3c_agent, transitions, init_state,
            discount_factor=discount_factor, reward_norm=reward_norm,
            batch_size=batch_size, time_major=time_major)

    return wrapper


def create_agent(agent_cls, state_shape, n_actions, agent_agrs, use_target_network):
    agent = agent_cls(
        state_shape, n_actions, **agent_agrs)

    if use_target_network:
        targets_special = {**agent_agrs["special"], **{"scope": "target_" + agent.scope}}
        agent_agrs["special"] = targets_special
        target_agent = agent_cls(
            state_shape, n_actions, **agent_agrs)
        agent = (agent, target_agent)

    from pprint import pprint
    pprint([(v.name, v.get_shape().as_list()) for v in tf.trainable_variables()])

    return agent


def run_wrapper(
        n_games, learning_fn, update_fn, play_fn, action_fn,
        env_name, make_env_fn, agent_cls,
        run_args, agent_agrs,
        log_dir=None,
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4,
        use_target_network=False):
    env = make_env_fn(env_name, n_games)

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    # hack, I know
    agent_agrs["special"]["batch_size"] = n_games
    agent = create_agent(agent_cls, state_shape, n_actions, agent_agrs, use_target_network)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.Saver()
        log_dir = log_dir or "./logs_" + env_name.replace(string.punctuation, "_")

        if not load:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "{}/model.ckpt".format(log_dir))

        try:
            history = learning_fn(
                sess, agent, env,
                update_fn=update_fn,
                **run_args)

            if plot_stats:
                save_history(history, log_dir)
                plotter_dir = os.path.join(log_dir, "plotter")
                plot_all_metrics(history, save_dir=plotter_dir)
        except KeyboardInterrupt:
            print("Exiting training procedure")
        save_model(sess, saver, log_dir)

    if api_key is not None:
        tf.reset_default_graph()
        agent_agrs["special"]["batch_size"] = 1
        agent = create_agent(agent_cls, state_shape, n_actions, agent_agrs, use_target_network)

        env_name = env_name.replace("Deterministic", "")
        env = make_env_fn(env_name, 1, limit=True)
        monitor_dir = os.path.join(log_dir, "monitor")
        env = gym.wrappers.Monitor(env, monitor_dir, force=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "{}/model.ckpt".format(log_dir))

            sessions = [play_fn(sess, agent, env, action_fn=action_fn) for _ in range(300)]

        env.close()
        gym.upload(monitor_dir, api_key=api_key)


def typical_args(parser):
    parser.add_argument(
        '--env',
        type=str,
        default='CartPole-v0',  # BreakoutDeterministic-v0
        help='The environment to use. (default: %(default)s)')

    # env pool params
    parser.add_argument(
        '--n_games',
        type=int,
        default=10,
        help='Number of parallel games to play during training. (default: %(default)s)')

    parser.add_argument(
        '--n_epochs',
        type=int,
        default=100,
        help='Number of epochs to train. (default: %(default)s)')
    parser.add_argument(
        '--n_sessions',
        type=int,
        default=10,
        help='Number of game session to play per one epoch. (default: %(default)s)')
    parser.add_argument(
        '--t_max',
        type=int,
        default=1000,
        help='Steps per game session to play. (default: %(default)s)')
    parser.add_argument(
        '--plot_history',
        action='store_true',
        default=False,
        help='Plot graph with main train statistics (reward, loss, steps). (default: %(default)s)')
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='Your API key to submit to gym. (default: %(default)s)')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help='Your API key to submit to gym. (default: %(default)s)')
    parser.add_argument(
        '--load',
        action='store_true',
        default=False,
        help='Flag to load previous model from log_dir. (default: %(default)s)')
    parser.add_argument(
        '--gpu_option',
        type=float,
        default=0.45,
        help='GPU usage. (default: %(default)s)')

    # feature network params
    parser.add_argument(
        '--feature_network',
        type=str,
        choices=["linear", "convolution"],
        default="linear",
        help='Feature network type, need to create vector representation of the state. '
             '(default: %(''default)s)')
    parser.add_argument(
        '--activation',
        type=str,
        default="elu",
        help='Typical activation for feature network. (default: %(default)s)')
    parser.add_argument(
        '--use_bn',
        action='store_true',
        default=False,
        help='Batchnorm usage flag. (default: %(default)s) - no batchnorm')
    parser.add_argument(
        '--dropout',
        type=float,
        default=-1,
        help='Dropout keep prob rate. (default: %(default)s) - no dropout')

    # special args for linear network
    parser.add_argument(
        '--layers',
        type=str,
        default=None,
        help='Linear feature network layers, splitted by \'-\'.')

    # special args for convolution network:
    parser.add_argument(
        '--n_filters',
        type=str,
        default=None,
        help='Convolution feature network filters, splitted by \'-\'.')
    parser.add_argument(
        '--kernels',
        type=str,
        default=None,
        help='Convolution feature network kernels, splitted by \'-\'.')
    parser.add_argument(
        '--strides',
        type=str,
        default=None,
        help='Convolution feature network strides, splitted by \'-\'.')

    # typical hidden state params
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=512,
        help='Hidden state size. (default: %(default)s)')
    parser.add_argument(
        '--hidden_activation',
        type=str,
        default="elu",
        help='Hidden state activation. (default: %(default)s)')

    # typical optimization params
    parser.add_argument(
        '--feature_lr',
        type=float,
        default=1e-3,
        help='Learning rate for feature network. (default: %(default)s)')
    parser.add_argument(
        '--lr_decay_steps',
        type=float,
        default=1e5,
        help='Learning rate decay steps. (default: %(default)s)')
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.999,
        help='Learning rate decay factor. (default: %(default)s)')
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=1.0,
        help='Gradient clip factor. (default: %(default)s)')

    # update args
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Gamma discount factor. (default: %(default)s)')
    parser.add_argument(
        '--reward_norm',
        type=float,
        default=1.0,
        help='Reward norm factor. (default: %(default)s)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Batch size for update, should be more than parallel games count. '
             '(default: %(''default)s)')
    parser.add_argument(
        '--time_major',
        action='store_true',
        default=False,
        help='Time-major flag for update. (default: %(default)s) - batch-major')

    # preprocess args for image envs
    parser.add_argument(
        '--image_width',
        type=float,
        default=64,
        help='Image-based environments preprocessing, cut to current width. '
             '(default: %(default)s)')
    parser.add_argument(
        '--image_height',
        type=float,
        default=64,
        help='Image-based environments preprocessing, cut to current height. '
             '(default: %(default)s)')
    parser.add_argument(
        '--image_grayscale',
        action='store_true',
        default=False,
        help='Image-based environments preprocessing, flag to grayscale state image.')
    parser.add_argument(
        '--image_corners',
        type=str,
        default=None,
        help='Image-based environments preprocessing, image corners splitted by \'-\'.')
    parser.add_argument(
        '--n_frames',
        type=int,
        default=1,
        help='Number of memory frames to use. (default: %(default)s)')

    return parser


def typical_argsparse(args):
    if args.feature_network == "linear":
        network_args = {
            "layers": str2params(args.layers)
        }
        make_env_fn = make_env_wrapper(make_env, {"n_frames": args.n_frames})
    elif args.feature_network == "convolution":
        network_args = {
            "n_filters": str2params(args.n_filters),
            "kernels": str2params(args.kernels),
            "strides": str2params(args.strides)
        }

        corners = str2params(args.image_corners)
        if corners is not None:
            image_crop_x1, image_crop_x2, image_crop_y1, image_crop_y2 = corners
            crop_fn = lambda img: img[image_crop_x1:image_crop_x2, image_crop_y1:image_crop_y2]
        else:
            crop_fn = None

        image_preprocessing_params = {
            "width": args.image_width,
            "height": args.image_height,
            "grayscale": args.image_grayscale,
            "crop": crop_fn,
            "n_frames": args.n_frames
        }

        make_env_fn = make_env_wrapper(make_image_env, image_preprocessing_params)
    else:
        raise NotImplemented()

    network = network_wrapper(
        networks[args.feature_network], {
            **network_args, **{
            "activation_fn": activations[args.activation],
            "use_bn": args.use_bn,
            "dropout": args.dropout
        }})

    run_args = {
        "n_epochs": args.n_epochs,
        "n_sessions": args.n_sessions,
        "t_max": args.t_max
    }
    update_args = {
        "discount_factor": args.gamma,
        "reward_norm": args.reward_norm,
        "batch_size": args.batch_size,
        "time_major": args.time_major,
    }
    optimization_params = {
        "initial_lr": args.feature_lr,
        "decay_steps": args.lr_decay_steps,
        "lr_decay": args.lr_decay,
        "grad_clip": args.grad_clip
    }

    return network, run_args, update_args, optimization_params, make_env_fn
