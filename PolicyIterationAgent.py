import numpy as np
from agent import Agent


# TASK 1

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your policy iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # Policy initialization
        # ******************
        # TODO 1.1.a)
        # self.V = ...
        self.V = {x:0.0 for x in states}
        # *******************

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}

        counter = 0

        while True:
            # Policy evaluation
            for i in range(iterations):
                newV = {}
                for s in states:
                    a = self.pi[s]
                    # *****************
                    # TODO 1.1.b)
                    # if...
                    #
                    # else:...
                    if a is None:
                        newV[s] = 0.0
                    else:
                # update value estimate
                # self.V=...
                        # assume pi(a) = 1 since there is only one action to be taken
                        newV[s] = sum(
                            [
                                p * ( self.mdp.getReward(s, a, sn) + self.discount * self.V[sn])
                                for sn, p in self.mdp.getTransitionStatesAndProbs(s, a)
                            ]
                        )
                self.V = newV
                # ******************

            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # TODO 1.1.c)
                    # self.pi[s] = ...

                    actionvals = {
                        a:sum(
                            [
                                p*(self.mdp.getReward(s, a, sn) + self.discount * self.V[sn])
                                for sn, p in self.mdp.getTransitionStatesAndProbs(s, a)
                            ])
                        for a in actions
                    }
                    
                    self.pi[s] = max(actionvals, key=actionvals.get)

                    # policy_stable =
                    policy_stable = policy_stable and self.pi[s] == old_action
                    # ****************
            counter += 1

            if policy_stable: break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        # TODO 1.2.
        return self.V[state]
        # ********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that policy iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # *********
        # TODO 1.3.
        return sum(
            [
                p * (self.mdp.getReward(state, action, sn) + self.discount * self.getValue(sn))
                for sn, p in self.mdp.getTransitionStatesAndProbs(state, action)
            ]
        )
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """
        # **********
        # TODO 1.4.
        return self.pi[state]
        # **********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents!
        """

        pass
