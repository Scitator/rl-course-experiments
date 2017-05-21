#!/usr/bin/python3

import tensorflow as tf
from tensorflow.contrib import rnn

from rstools.tf.optimization import build_model_optimization

from agent_networks import FeatureNet, QvalueNet, ValueNet
from agent_states import RecurrentHiddenState


class DqrnAgent(object):
    def __init__(self, state_shape, n_actions, network, cell, special=None):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.cell_size = cell.state_size

        self.is_end = tf.placeholder(shape=[None], dtype=tf.bool, name="is_end")

        self.special = special
        self.scope = special.get("scope", "dqrn")

        with tf.variable_scope(self.scope):
            self._build_graph(network, cell)

    def _build_graph(self, network, cell):
        self.feature_net = FeatureNet(
            self.state_shape, network,
            self.special.get("feature_net", {}))

        self.hidden_state = RecurrentHiddenState(self.feature_net.feature_state, cell)

        if self.special("double_network", False):
            self.qvalue_net = QvalueNet(
                self.hidden_state.state, self.n_actions,
                dict(**self.special.get("qvalue_net", {}), **{"advantage": True}))
            self.value_net = ValueNet(
                self.hidden_state.state,
                self.special.get("value_net", {}))

            build_model_optimization(
                self.value_net,
                self.special.get("value_net_optimization", None))
            model_loss = 0.5 * (self.qvalue_net.loss + self.value_net.loss)
        else:
            self.qvalue_net = QvalueNet(
                self.hidden_state.state, self.n_actions,
                self.special.get("qvalue_net", {}))
            model_loss = self.qvalue_net.loss

        build_model_optimization(
            self.qvalue_net,
            self.special.get("qvalue_net_optimization", None))

        build_model_optimization(
            self.hidden_state,
            self.special.get("hidden_state_optimization", None),
            loss=model_loss)
        build_model_optimization(
            self.feature_net,
            self.special.get("feature_net_optimization", None),
            loss=model_loss)

    def predict_qvalues(self, sess, state_batch):
        return sess.run(
            self.qvalue_net.predicted_qvalues,
            feed_dict={
                self.feature_net.states: state_batch,
                self.feature_net.is_training: False
            })

    def update_belief_state(self, sess, state_batch, done_batch):
        _ = sess.run(
            self.hidden_state.belief_update,
            feed_dict={
                self.feature_net.states: state_batch,
                self.hidden_state.is_end: done_batch,
                self.feature_net.is_training: False
            })

    def assign_belief_state(self, sess, new_belief):
        _ = sess.run(
            self.hidden_state.belief_assign,
            feed_dict={
                self.hidden_state.belief_out: new_belief
            })

    def get_belief_state(self, sess):
        return sess.run(self.hidden_state.belief_state)


def update_on_batch(
        sess, dqrn_agent,
        state_batch, action_batch, reward_batch, next_state_batch, done_batch,
        discount_factor=0.99, reward_norm=1.0):
    values_next = dqrn_agent.predict(sess, next_state_batch)
    td_target = reward_batch * reward_norm + \
                np.invert(done_batch).astype(np.float32) * \
                discount_factor * values_next.max(axis=1)

    loss, _, _ = sess.run(
        [dqrn_agent.qvalue_net.loss,
         dqrn_agent.qvalue_net.train_op,
         dqrn_agent.feature_net.train_op],
        feed_dict={
            dqrn_agent.feature_net.states: state_batch,
            dqrn_agent.feature_net.is_training: True,
            dqrn_agent.qvalue_net.actions: action_batch,
            dqrn_agent.qvalue_net.td_target: td_target,
            dqrn_agent.qvalue_net.is_training: True,
        })

    return loss


def update_wraper(discount_factor=0.99, reward_norm=1.0):
    def wrapper(sess, q_net, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        return update_on_batch(
            sess, q_net,
            state_batch, action_batch, reward_batch, next_state_batch, done_batch,
            discount_factor=discount_factor, reward_norm=reward_norm)

    return wrapper


def generate_sessions(sess, dqrn_agent, env_pool, t_max=1000, epsilon=0.25, update_fn=None):
    total_reward = 0.0
    total_loss = 0.0
    total_games = 0.0

    states = env_pool.pool_states()
    for t in range(t_max):
        actions = epsilon_greedy_policy(dqrn_agent, sess, states, epsilon)
        new_states, rewards, dones, _ = env_pool.step(actions)

        if update_fn is not None:
            total_loss += update_fn(
                sess, dqrn_agent,
                states, actions, rewards, new_states, dones)

        dqrn_agent.update_belief_state(sess, states, dones)
        states = new_states

        total_reward += rewards.mean()
        total_games += dones.sum()

    return total_reward, total_loss, total_games


def generate_session(
        sess, dqrn_agent, env, t_max=1000, epsilon=0.25, update_fn=None):
    total_reward = 0
    total_loss = 0.0
    s = env.reset()

    for t in range(t_max):
        a = epsilon_greedy_policy(dqrn_agent, sess, np.array([s], dtype=np.float32), epsilon)[0]

        new_s, r, done, _ = env.step(a)

        if update_fn is not None:
            total_loss += update_fn(
                sess, dqrn_agent,
                [s], [a], [r], [new_s], [done])

        total_reward += r

        dqrn_agent.update_belief_state(sess, [s], [done])
        s = new_s

        if done:
            break

    return total_reward, total_loss / float(t + 1), t


def dqrn_learning(
        sess, q_net, env, update_fn,
        n_epochs=1000, n_sessions=100, t_max=1000,
        initial_epsilon=0.25, final_epsilon=0.01):
    tr = trange(
        n_epochs,
        desc="",
        leave=True)

    epsilon = initial_epsilon
    n_epochs_decay = n_epochs * 0.8

    moi = ["loss", "reward", "steps"]
    history = {metric: np.zeros(n_epochs) for metric in moi}

    for i in tr:
        sessions = [
            generate_sessions(sess, q_net, env, t_max, epsilon=epsilon, update_fn=update_fn)
            for _ in range(n_sessions)]
        session_rewards, session_loss, session_steps = \
            map(np.array, zip(*sessions))

        if i < n_epochs_decay:
            epsilon -= (initial_epsilon - final_epsilon) / float(n_epochs_decay)

        history["reward"][i] = np.mean(session_rewards)
        history["loss"][i] = np.mean(session_loss)
        history["steps"][i] = np.mean(session_steps)

        desc = "\t".join(
            ["{} = {:.3f}".format(key, value[i]) for key, value in history.items()])
        tr.set_description(desc)

    return history


def conv_network(
        states,
        scope, reuse=False,
        is_training=True, activation_fn=tf.nn.elu):
    with tf.variable_scope(scope or "network", reuse=reuse):
        conv = tflayers.conv2d(
            states,
            num_outputs=32,
            kernel_size=8,
            stride=4,
            activation_fn=activation_fn)
        conv = tflayers.conv2d(
            conv,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=activation_fn)
        conv = tflayers.conv2d(
            conv,
            num_outputs=64,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn)

        flat = tflayers.flatten(conv)
        return flat


def network_wrapper(activation_fn=tf.nn.elu):
    def wrapper(states, scope=None, reuse=False, is_training=True):
        return conv_network(states, scope, reuse, is_training, activation_fn=activation_fn)

    return wrapper


def _parse_args():
    parser = argparse.ArgumentParser(description='Policy iteration example')
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
        '--gamma',
        type=float,
        default=0.99,
        help='Gamma discount factor')
    parser.add_argument(
        '--plot_stats',
        action='store_true',
        default=False)
    parser.add_argument(
        '--api_key',
        type=str,
        default=None)
    parser.add_argument(
        '--activation',
        type=str,
        default="elu")
    parser.add_argument(
        '--load',
        action='store_true',
        default=False)
    parser.add_argument(
        '--initial_epsilon',
        type=float,
        default=0.25,
        help='Gamma discount factor')
    parser.add_argument(
        '--gpu_option',
        type=float,
        default=0.45)

    parser.add_argument(
        '--initial_lr',
        type=float,
        default=1e-4)
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
        '--n_games',
        type=int,
        default=10)
    parser.add_argument(
        '--lstm_activation',
        type=str,
        default="tanh")

    parser.add_argument(
        '--reward_norm',
        type=float,
        default=1.0)

    args, _ = parser.parse_known_args()
    return args


def make_env(env, n_games=1, width=64, height=64, 
             grayscale=True, crop=lambda img: img[60:-30, 7:]):
        if n_games > 1:
            return EnvPool(
                PreprocessImage(
                    env,
                    width=width, height=height, grayscale=grayscale,
                    crop=crop), 
                n_games)
        else:
            return PreprocessImage(
                env,
                width=width, height=height, grayscale=grayscale,
                crop=crop)


def run(env, q_learning_args, update_args, agent_args,
        n_games, lstm_activation="tanh",
        plot_stats=False, api_key=None,
        load=False, gpu_option=0.4):
    env_name = env
    env = make_env(gym.make(env_name).env, n_games)

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    network = agent_args.get("network", None) or conv_network
    cell = rnn.LSTMCell(512, activation=activations[lstm_activation])
    q_net = DQRNAgent(
        state_shape, n_actions, network, cell=cell,
        special=agent_args)

    model_dir = "./logs_" + env_name.replace(string.punctuation, "_")
    model_dir += "_{}".format(lstm_activation)
    create_if_need(model_dir)

    # @TODO: very very hintly, need to find best solution
    # vars_of_interest = [
    #     v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #     if not v.name.startswith("belief_state")]
    vars_of_interest = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)
    saver = tf.train.Saver(var_list=vars_of_interest)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if not load:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "{}/model.ckpt".format(model_dir))

        stats = dqrn_learning(
            sess, q_net, env,
            update_fn=update_wraper(**update_args),
            **q_learning_args)
        create_if_need(model_dir)
        saver.save(
            sess, "{}/model.ckpt".format(model_dir),
            meta_graph_suffix='meta', write_meta_graph=True)
        # tf.train.write_graph(sess.graph_def, model_dir, "graph.pb", False)

    if plot_stats:
        stats_dir = os.path.join(model_dir, "stats")
        create_if_need(stats_dir)
        save_stats(stats, save_dir=stats_dir)

    if api_key is not None:
        tf.reset_default_graph()

        agent_args["n_games"] = 1
        q_net = DQRNAgent(
            state_shape, n_actions, network, cell=cell,
            special=agent_args)
        vars_of_interest = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(var_list=vars_of_interest)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.variables_initializer(
                [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                 if v.name.startswith("belief_state")]))
            # sess.run(tf.global_variables_initializer())
            saver.restore(sess, "{}/model.ckpt".format(model_dir))

            env_name = env_name.replace("Deterministic", "")
            env = make_env(gym.make(env_name))
            env = gym.wrappers.Monitor(env, "{}/monitor".format(model_dir), force=True)
            sessions = [generate_session(sess, q_net, env, int(1e10), epsilon=0.01, update_fn=None)
                        for _ in range(300)]
            env.close()
            gym.upload("{}/monitor".format(model_dir), api_key=api_key)


def main():
    args = _parse_args()
    network = network_wrapper(activations[args.activation])
    q_learning_args = {
        "n_epochs": args.n_epochs,
        "n_sessions": args.n_sessions,
        "t_max": args.t_max,
        "initial_epsilon": args.initial_epsilon
    }
    update_args = {
        "discount_factor": args.gamma,
        "reward_norm": args.reward_norm,
    }
    optimization_params = {
        "initial_lr": args.initial_lr,
        "decay_steps": args.lr_decay_steps,
        "lr_decay": args.lr_decay,
        "grad_clip": args.grad_clip
    }
    agent_args = {
        "n_games": args.n_games,
        "network": network,
        "feature_net_optimization": optimization_params,
        "qvalue_net_optimiaztion": optimization_params
    }
    run(args.env, q_learning_args, update_args, agent_args,
        args.n_games, args.lstm_activation,
        args.plot_stats, args.api_key,
        args.load, args.gpu_option)


if __name__ == '__main__':
    main()
