## Deep Q-Learning

### Algorithms & Readings

- [Deep Q-Learning (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Human-Level Control through Deep Reinforcement Learning](http://www.davidqiu.com:8888/research/nature14236.pdf)
- [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Deep Recurrent Q-Learning for Partially Observable MDPs (DRQN)](https://arxiv.org/abs/1507.06527)
- [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)


### Summary

- DQN: Q-Learning but with a Deep Neural Network as a function approximator.
- Using a non-linear Deep Neural Network is powerful, but training is unstable if we apply it naively.
- Trick 1 - Experience Replay: Store experience `(S, A, R, S_next)` in a replay buffer and sample minibatches from it to train the network. This decorrelates the data and leads to better data efficiency. In the beginning, the replay buffer is filled with random experience.
- Trick 2 - Target Network: Use a separate network to estimate the TD target. This target network has the same architecture as the function approximator but with frozen parameters. Every T steps (a hyperparameter) the parameters from the Q network are copied to the target network. This leads to more stable training because it keeps the target function fixed (for a while).
- Double DQN: Just like regular Q-Learning, DQN tends to overestimate values due to its max operation applied to both selecting and estimating actions. We get around this by using the Q network for selection and the target network for estimation when making updates.
