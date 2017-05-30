# RL course experiments

### Overview
This repository provides code implementations for popular Reinforcement Learning algorithms.

Main idea was to generalise main RL algorithms and provide unified interface for testing them on any gym environment. 
For example, now your can create your own Double Dueling Deep Recurrent Q-Learning agent (Let's name it, 3DRQ). 
For simplicity, all main agent blocks are in `agents` folder. 

For now, repository is under after-course refactoring. So, many documentation needed.

All code is written in Python 3 and uses RL environments from OpenAI Gym. 
Advanced techniques use Tensorflow for neural network implementations.

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
* [Policy Gradient](https://github.com/Scitator/rl-course-experiments/tree/master/PG)
* [Asynchronous Advantage Actor-Critic](https://github.com/Scitator/rl-course-experiments/tree/master/A3C)
* [Optimality Tightening](https://arxiv.org/abs/1611.01606) [TODO]
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) [TODO]
* Continuous action space [TODO]
* Monte Carlo Tree Search [TODO]

For more information, look at folder readme.

#### Special requirements

For simple script running you need to install additional [repo](https://github.com/Scitator/rstools) with optimization stuff for neural networks:

`pip install git+https://github.com/Scitator/rstools`

#### Example usage

DQN:

```
PYTHONPATH=. python DQN/run_dqn.py --plot_history 
--feature_network linear --layers 16-16 --activation tanh --hidden_size 16 --hidden_activation tanh 
--n_epochs 500 --n_games 32 --batch_size 32 --t_max 100  
--qvalue_lr 0.00005 --feature_lr 0.00005 --value_lr 0.00005 --initial_epsilon 0.25 
--api_key <paste_your_gym_api_key_here>
```

Reinforce:

```
PYTHONPATH=. python PG/run_reinforce.py --plot_history 
--feature_network linear --layers 16-16 --activation tanh --hidden_size 16 --hidden_activation tanh 
--n_epochs 3200 --n_games 32 --batch_size 32 --t_max 50 
--entropy_factor 0.005 --policy_lr 0.00001 --feature_lr 0.00005 
--api_key <paste_your_gym_api_key_here>
```

Feed-Forward Asynchronous Advantage Actor-Critic:

```
PYTHONPATH=. python A3C/run_a3c.py --plot_history 
--feature_network linear --layers 16-16 --activation tanh --hidden_size 16 --hidden_activation tanh 
--n_epochs 200 --n_games 32 --batch_size 32 --t_max 10 --policy_lr 0.000001 
--api_key <paste_your_gym_api_key_here>
```

##### Metrics

- loss - typical neural network loss
- reward - typical environment reward, 
but because Environment Pool is always used not very informative for now
- steps - mean number of game ends per epoch session

##### If you have linux with NVIDIA GPU and no X server, but want to try gym

You need to reinstall NVIDIA drivers.

[issue source](https://github.com/openai/gym/issues/366)
[how-to guide](https://davidsanwald.github.io/2016/11/13/building-tensorflow-with-gpu-support.html)

and add `bash xvfb start; DISPLAY=:1` before run command. 

#### Contributing

##### write code

Found a bug or know how to write it simpler? 
Or maybe you want to create your own agent? 
Just follow PEP8 and make merge request.

##### ...or play a game

We have a lot of RL algorithms, and even more gym environments to test them. 
So, play a game, save
* agent parameters (so anyone can reproduce)
* agent itself (`model.ckpt*`)
* plots (they will be automatically generated with `--plot_history` flag)
* gym-link (main results)
* make merge request (solutions should be at `field/solutions.md`, for example `DQN/solutions.md`)
