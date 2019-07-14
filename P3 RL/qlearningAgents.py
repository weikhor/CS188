# qlearningAgents.py
# ------------------
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

#Khor Chean Wei
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        #from github
        #indexed by state and action
        #not need define self.epsilon, self.alpha, self.discount, self.actionFn
        #because this object is inherited from ReinforcementAgent
        #value = []
        #for v1, v2 in args.items():
            #value.append(v2)
            
        #self.epsilon = value[2]
        #self.alpha = value[1]
        #self.discount = value[0]
        #self.actionFn = value[3]
        self.Q = {}
        
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if((state, action) not in self.Q):
            return 0
        else:
            return self.Q[state, action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #from github
        #not need use reward a this function
        
        #use len(self.getLegalActions(state)) for "TERMINAL_STATE"
        #very important in solving question 9
        
        if(len(self.getLegalActions(state)) == 0):
            return 0.0
        else: 
            Max = -float("inf")
            for a in self.getLegalActions(state):
                if(Max < self.getQValue(state, a)):
                    Max = self.getQValue(state, a)
            return Max
        
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        action = None
        
        if(len(self.getLegalActions(state)) == 0):
            return action
        else:
            Max = self.computeValueFromQValues(state)
            
            for a in self.getLegalActions(state):
                if(Max == self.getQValue(state, a)):
                    action = a
                    
            if(action == None):
                print(self.getLegalActions(state) , " ", Max)
                for a in self.getLegalActions(state):
                    print(self.getQValue(gstate, a))
                
                
            return action
                
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if(len(self.getLegalActions(state)) == 0):
            return action
        else:
            if(util.flipCoin(self.epsilon) == True):
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #from github
        #need to consider the TERMINAL_STATE
        
        #need to find the self.Q[state, action] even
        #when nextState is in Terminal state or non Terminal state
      
        if(len(self.getLegalActions(nextState)) == 0 ):
            self.Q[state, action] = (1 - self.alpha) * self.getQValue(state, action) +  self.alpha * reward
        else:
            Max = -float("inf")
            for a in self.getLegalActions(nextState):
                if(Max < self.getQValue(nextState, a)):
                    Max = self.getQValue(nextState, a)
            self.Q[state, action] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * Max)
        #print(self.Q[state, action])
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        #github answer 
        #need refer to function computeValueFromQValues(self, state) from parent class
        
        #after studying  the answer of function getQValue,update
        #i in self.weights[i] is not in number 
        #i in self.weights[i] is object obtained from self.featExtractor.getFeatures(state, action)
        #after changing the code, this question solves
        
    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        #print(self.featExtractor[state, action])
        #print(self.featExtractor.getFeatures(state, action))
        #print(action)
        #print(state, " " ,action
        
        ans = 0
        i = 0
        
        for k, v in self.featExtractor.getFeatures(state, action).items():
            ans = ans + self.weights[k]*v
            i = i + 1
        return ans
        
    

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        #from github answer
        #not need use same loop again for self.weights[i]
        
        if(len(self.getLegalActions(nextState)) == 0):
            diff = reward - self.getQValue(state, action)
        else:  
            Max = -float("inf")
            for a in self.getLegalActions(nextState):
                if(Max < self.getQValue(nextState, a)):
                        Max = self.getQValue(nextState, a)
                        
            diff = (reward + self.discount*Max) - self.getQValue(state, action)
       
        i = 0
        for k, v in self.featExtractor.getFeatures(state, action).items():
            self.weights[k] = self.weights[k] + self.alpha*diff*v
            i = i + 1
        
        

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            #pass
