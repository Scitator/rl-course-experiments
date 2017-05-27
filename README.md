# RL course experiments

### Overview
This repository provides code implementations for popular Reinforcement Learning algorithms.

Main idea was to generalise main RL algorithms and provide unified interface for testing them on any gym environment. For simplicity, all main agent blocks are in `agents` folder.

For now, repository is under after-course refactoring. So, many documentation needed.

All code is written in Python 3 and uses RL environments from OpenAI Gym. Advanced techniques use Tensorflow for neural network implementations.

### Inspired by:
* [Berkeley CS188x](http://ai.berkeley.edu/home.html)
* [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning)
* [yandexdataschool/Practical_RL](https://github.com/yandexdataschool/Practical_RL)
* [yandexdataschool/AgentNet](https://github.com/yandexdataschool/AgentNet)

##### Additional thanks to [JustHeuristic](https://github.com/justheuristic) for Practical_RL course

### Table of Contents
* [Genetic algorithm](https://github.com/Scitator/rl-course-experiments/tree/master/GEN)
* [Dynamic Programming](https://github.com/Scitator/rl-course-experiments/tree/master/DP)
* [Cross Entropy Method](https://github.com/Scitator/rl-course-experiments/tree/master/CEM)
* [Monte Carlo Control](https://github.com/Scitator/rl-course-experiments/tree/master/MC)
* [Temporal Difference](https://github.com/Scitator/rl-course-experiments/tree/master/TD)
* [Deep Q-Networks](https://github.com/Scitator/rl-course-experiments/tree/master/DQN)
* [Policy Gradient](https://github.com/Scitator/rl-course-experiments/tree/master/PG) [refactoring]
* [Asynchronous Advantage Actor-Critic](https://github.com/Scitator/rl-course-experiments/tree/master/A3C) [refactoring]
* [Optimality Tightening](https://arxiv.org/abs/1611.01606) [TODO]
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) [TODO]
* Continuous action space [TODO]
* Monte Carlo Tree Search [TODO]

For more information, look at folder readme.

#### Special requirements

For simple script running you need to install my additional [repo](https://github.com/Scitator/rstools) with all kind of optimization stuff for neural networks:

`pip install git+https://github.com/Scitator/rstools`