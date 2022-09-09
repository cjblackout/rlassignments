import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 0  # initial value for states
        self.Q = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.
        pa = self.getPolicy(state)
        value = 0.0
        all_actions = self.actionFunction(state)
        for a in all_actions:
            pi = (self.epsilon / len(all_actions))
            if a == pa:
                pi = (1.0 - self.epsilon + self.epsilon / len(all_actions))
            value += pi * self.getQValue(state, a)
        return value
        

        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        # all actions from terminal state are considered Q=0 for the terminal state [RLBook P. 132]
        if state == (-1, -1):
            return 0.0
        if not state in self.Q:
            self.Q[state] = {action:0.0 for action in self.actionFunction(state)}
        return self.Q[state][action]
        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        # if state known, return argmax Q(s, a)
        if state in self.Q:
            return max(self.Q[state], key=self.Q[state].get)
        else:# if state unknown
            return self.getRandomAction(state)
        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            # *********
            return np.random.choice(all_actions)
            # *********
        else:
            return "exit"

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        if np.random.rand() < self.epsilon:
            return self.getRandomAction(state)
        else:
            return self.getPolicy(state)
        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.
        if not state in self.Q:
            # state is not in Q so initialize it to 0.0
            self.Q[state] = {action:0.0 for action in self.actionFunction(state)}
        if not nextState in self.Q and not nextState == (-1, -1):
            # nextState is not in Q so initialize it with 0.0
            self.Q[nextState]= {action:0.0 for action in self.actionFunction(nextState)}
        if nextState == (-1, -1):
            self.Q[state][action] += self.learningRate * (reward - self.Q[state][action])
        else:
            self.Q[state][action] += self.learningRate * (reward + self.discount * max([q for q in self.Q[nextState].values()]) - self.Q[state][action])
        # *********
