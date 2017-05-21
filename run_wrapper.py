import numpy as np
import string
import os
import gym
from gym import wrappers
import tensorflow as tf

from rstools.utils.os_utils import save_history, save_model
from rstools.visualization.plotter import plot_all_metrics

from networks import activations, networks, network_wrapper, str2params
from wrappers import make_env, make_image_env_wrapper


def epsilon_greedy_policy(agent, sess, observations):
    probs = agent.predict_action(sess, observations)
    actions = [np.random.choice(len(row), p=row) for row in probs]
    return actions


def play_session(sess, agent, env, t_max=int(1e10)):
    total_reward = 0

    s = env.reset()
    for t in range(t_max):
        a = epsilon_greedy_policy(agent, sess, np.array([s], dtype=np.float32))[0]

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


def run_wrapper(
        n_games, learning_fn, update_fn, play_fn,
        env_name, make_env_fn, agent_cls,
        run_args, agent_agrs,
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4):
    env = make_env_fn(env_name, n_games)

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    agent = agent_cls(
        state_shape, n_actions, agent_agrs["network"],
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
            history = learning_fn(
                sess, agent, env,
                update_fn=update_fn,
                **run_args)
        except KeyboardInterrupt:
            print("Exiting training procedure")
        save_model(sess, saver, model_dir)

        if plot_stats:
            save_history(history, model_dir)
            plotter_dir = os.path.join(model_dir, "plotter")
            plot_all_metrics(history, save_dir=plotter_dir)

        if api_key is not None:
            env_name = env_name.replace("Deterministic", "")
            env = make_env_fn(env_name, 1)
            monitor_dir = os.path.join(model_dir, "monitor")
            env = gym.wrappers.Monitor(env, monitor_dir, force=True)
            sessions = [play_fn(sess, agent, env) for _ in range(300)]
            env.close()
            gym.upload(monitor_dir, api_key=api_key)


def typical_args(parser):
    parser.add_argument(
        '--env',
        type=str,
        default='CartPole-v0',  # BreakoutDeterministic-v0
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

    # feature network params
    parser.add_argument(
        '--feature_network',
        type=str,
        choices=["linear", "convolution"])
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

    # special args for linear network
    parser.add_argument(
        '--layers',
        type=str,
        default=None)

    # special args for convolution network:
    parser.add_argument(
        '--n_filters',
        type=str,
        default=None)
    parser.add_argument(
        '--kernels',
        type=str,
        default=None)
    parser.add_argument(
        '--strides',
        type=str,
        default=None)

    # typical optimization params
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
        default=5.0)

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


def typical_argsparse(args):
    if args.feature_network == "linear":
        network_args = {
            "layers": str2params(args.layers)
        }
    elif args.feature_network == "convolution":
        network_args = {
            "n_filters": str2params(args.n_filters),
            "kernels": str2params(args.kernels),
            "strides": str2params(args.strides)
        }
    else:
        raise NotImplemented()

    network = network_wrapper(
        networks[args.feature_network],
        dict(**network_args, **{
            "activation_fn": activations[args.activation],
            "use_bn": args.use_bn,
            "dropout": args.dropout
        }))

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

    image_preprocessing_params = {
        "width": args.image_width,
        "height": args.image_height,
        "grayscale": args.image_grayscale,
        "crop": lambda img: img[args.image_crop_x1:args.image_crop_x1,
                            args.image_crop_y1:args.image_crop_y1]
    }

    make_env_fn = make_env \
        if args.feature_network == "linear" \
        else make_image_env_wrapper(image_preprocessing_params)

    return network, run_args, update_args, optimization_params, make_env_fn