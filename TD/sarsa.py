"""
SARSA Agent
This file builds upon the same functions as Q-learning agent (qlearning.py).

Here's usage example:
    from sarsa import SarsaAgent

    agent = SarsaAgent(
        alpha=0.1,epsilon=0.25,discount=0.99,
        getLegalActions = lambda s: actions_from_that_state)
    action = agent.getAction(state)
    agent.update(state, action, next_state, reward)
    agent.epsilon *= 0.99
"""
import random

import numpy as np
from collections import defaultdict


class SarsaAgent(object):
    """
      Classical SARSA agent.

      The two main methods are
      - self.getAction(state) - returns agent's action in that state
      - self.update(state,action,reward,nextState,nextAction) - returns agent's next action

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate aka gamma)

    """

    def __init__(self, alpha, epsilon, discount, getLegalActions):
        "We initialize agent and Q-values here."
        self.getLegalActions = getLegalActions
        self._qValues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
        """
        return self._qValues[state][action]

    def setQValue(self, state, action, value):
        """
          Sets the Qvalue for [state,action] to the given value
        """
        self._qValues[state][action] = value

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.
        """
        possibleActions = self.getLegalActions(state)

        # If there are no legal actions, return None
        if len(possibleActions) == 0:
            return None

        "*** this code works exactly as Q-learning ***"
        best_action = possibleActions[
            np.argmax([self.getQValue(state, a) for a in possibleActions])]
        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state, including exploration.
        """

        # Pick Action
        possibleActions = self.getLegalActions(state)
        action = None

        # If there are no legal actions, return None
        if len(possibleActions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        "*** Epsilon-greedy strategy exactly as Q-learning ***"
        if np.random.random() <= epsilon:
            action = random.choice(possibleActions)
        else:
            action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, nextAction, reward):
        """
          You should do your Q-Value update here
        """
        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        "*** YOUR CODE HERE ***"
        reference_qvalue = reward + gamma * self.getQValue(nextState, nextAction)

        updated_qvalue = (1 - learning_rate) * self.getQValue(state, action) + \
            learning_rate * reference_qvalue

        self.setQValue(state, action, updated_qvalue)
