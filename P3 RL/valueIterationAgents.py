# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #see the github solution for this function
        #from github need to use self.velues to store grid position and values, 
        #find maximum q value among action
        #value in self.value  cannot be change during the iteration:
        #solved it using self.values = store
        for _ in range(self.iterations):
            store = {}
            for state in self.mdp.getStates():
                s = self.values[state]
                store[state] = s
                
            for state in self.mdp.getStates():
                Max = -float("inf")
                for a in self.mdp.getPossibleActions(state):
                    q = self.computeQValueFromValues(state, a)
                    if(Max < q):
                        Max = q
                        store[state] = Max 
            self.values = store
    
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]
 
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        
        for i in self.mdp.getTransitionStatesAndProbs(state, action):
            prob = i[1]
            next_state = i[0]
            reward = self.mdp.getReward(state, action, next_state)
            q = q + prob*(reward + self.discount*self.values[next_state])
            
        return q
        
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #from gtihub solutuion must return action or None, need to use computeQValueFromValues(self, state, action)
        if(self.mdp.isTerminal(state)):
            return None
        
        Max = -float("inf")
        best_action = None
        
        for a in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, a)
            if(q > Max):
                Max = q
                best_action = a
                
        return best_action
    
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        length = len(self.mdp.getStates())
       
        for i in range(self.iterations):
            index = i % length
            state = self.mdp.getStates()[index]
            if (state != "TERMINAL_STATE"):
                Max = -float("inf")
                for a in self.mdp.getPossibleActions(state):
                    q = self.computeQValueFromValues(state, a)
                    if(Max < q):
                        Max = q
                self.values[state] = Max
        
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #from github
        #use set()
        #in priorityqueue, not need produce key and value
        predecessors = {}
        for state in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(state):
                for i in self.mdp.getTransitionStatesAndProbs(state, a):
                    prob = i[1]
                    next_state = i[0]
                    if(prob > 0):
                        if(next_state not in predecessors):
                            predecessors[next_state] = set()
                            predecessors[next_state].add(state)
                        else:
                            predecessors[next_state].add(state)
                    
        queue = util.PriorityQueue()
        
        for state in self.mdp.getStates():
            if(self.mdp.isTerminal(state) != True):
                s = self.values[state]
                Max = -float("inf")
                for a in self.mdp.getPossibleActions(state):
                    q = self.computeQValueFromValues(state, a)
                    if(Max < q):
                        Max = q
                diff = abs(s - Max)
                queue.push(state, -diff)
                
        for _ in range(self.iterations):
            if queue.isEmpty():
                return
            else:   
                state = queue.pop()
                
                if(self.mdp.isTerminal(state) != True):
                    Max = -float("inf")
                    for a in self.mdp.getPossibleActions(state):
                        q = self.computeQValueFromValues(state, a)
                        if(Max < q):
                            Max = q
                    self.values[state] = Max
            
                for P in predecessors[state]:
                    p_v = self.values[P]
                    Max = -float("inf")
                    for a in self.mdp.getPossibleActions(P):
                        q = self.computeQValueFromValues(P, a)
                        if(Max < q):
                            Max = q
                    
                    diff = abs(Max - p_v)
                    
                    if(diff > self.theta):
                        queue.update(P, -diff)
        
                        
        
        
                    
       
