import tensorflow as tf


class FeatureNet(object):
    def __init__(self, state_shape, network, special=None):
        self.special = special or {}
        self.state_shape = state_shape

        self.states = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32, name="states")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.loss = None
        self.optimizer = None
        self.train_op = None

        self.scope = self.special.get("scope", "feature_network")

        self.hidden_state = network(
            self.states,
            scope=self.scope + "/hidden",
            reuse=self.special.get("reuse_hidden", False),
            is_training=self.is_training)


class PolicyNet(object):
    def __init__(self, hidden_state, n_actions, special=None):
        self.special = special or {}
        self.n_actions = n_actions

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.cumulative_rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="rewards")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optimizer = None
        self.train_op = None

        self.scope = self.special.get("scope", "policy_network")

        self.predicted_probs = self._probs(
            hidden_state,
            scope=self.scope + "/probs",
            reuse=self.special.get("reuse_probs", False))

        self.predicted_probs_for_actions = tf.gather(
            self.predicted_probs, self.actions)
        # one_hot_actions = tf.one_hot(self.actions, n_actions)
        # predicted_probs_for_actions = tf.reduce_sum(
        #     tf.multiply(self.predicted_probs, one_hot_actions),
        #     axis=-1)

        J = tf.reduce_mean(tf.log(self.predicted_probs_for_actions) * self.cumulative_rewards)
        self.loss = -J

        # a bit of regularization
        if self.special.get("entropy_loss", True):
            H = tf.reduce_mean(
                tf.reduce_sum(
                    self.predicted_probs * tf.log(self.predicted_probs),
                    axis=-1))
            self.loss += H * self.special.get("entropy_koef", 0.01)

    def _probs(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            probs = tf.layers.dense(
                hidden_state,
                units=self.n_actions,
                activation=tf.nn.softmax)
            return probs


class StateNet(object):
    def __init__(self, hidden_state, special=None):
        self.special = special or {}

        self.td_target = tf.placeholder(shape=[None], dtype=tf.float32, name="td_target")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optimizer = None
        self.train_op = None

        self.scope = self.special.get("scope", "state_network")

        self.predicted_values = tf.squeeze(
            self._state_value(
                hidden_state,
                scope=self.scope + "/state_value",
                reuse=self.special.get("reuse_state_value", False)),
            axis=1)

        self.loss = tf.losses.mean_squared_error(
            labels=self.td_target,
            predictions=self.predicted_values)

    def _state_value(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            qvalues = tf.layers.dense(
                hidden_state,
                units=1,
                activation=None)
            return qvalues


class QvalueNet(object):
    def __init__(self, hidden_state, n_actions, special=None):
        self.special = special or {}
        self.n_actions = n_actions

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.td_target = tf.placeholder(shape=[None], dtype=tf.float32, name="td_target")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optimizer = None
        self.train_op = None

        self.scope = self.special.get("scope", "qvalue_network")

        self.predicted_qvalues = self._qvalues(
            hidden_state,
            scope=self.scope + "/qvalue",
            reuse=self.special.get("reuse_state_value", False))

        self.predicted_qvalues_for_actions = tf.gather(
            self.predicted_qvalues, self.actions)
        # one_hot_actions = tf.one_hot(self.actions, n_actions)
        # self.predicted_qvalues_for_actions = tf.reduce_sum(
        #     tf.multiply(self.predicted_qvalues, one_hot_actions),
        #     axis=-1)

        self.loss = tf.losses.mean_squared_error(
            labels=self.td_target,
            predictions=self.predicted_qvalues_for_actions)

    def _qvalues(self, hidden_state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            qvalues = tf.layers.dense(
                hidden_state,
                units=self.n_actions,
                activation=None)
            if self.special.get("advantage", False):
                qvalues -= tf.reduce_mean(qvalues, axis=-1, keep_dims=True)
            return qvalues

