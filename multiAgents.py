# multiAgents.py
# --------------
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

from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent


# Question 1 (not required)
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    # def getAction(self, gameState):
    #     """
    #     You do not need to change this method, but you're welcome to.
    #
    #     getAction chooses among the best options according to the evaluation function.
    #
    #     Just like in the previous project, getAction takes a GameState and returns
    #     some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
    #     """
    #     # Collect legal moves and successor states
    #     legalMoves = gameState.getLegalActions()
    #
    #     # Choose one of the best actions
    #     scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    #     bestScore = max(scores)
    #     bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    #     chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    #
    #     "Add more of your code here if you want to"
    #
    #     return legalMoves[chosenIndex]
    #
    # def evaluationFunction(self, currentGameState, action):
    #     """
    #     Design a better evaluation function here.
    #
    #     The evaluation function takes in the current and proposed successor
    #     GameStates (pacman.py) and returns a number, where higher numbers are better.
    #
    #     The code below extracts some useful information from the state, like the
    #     remaining food (newFood) and Pacman position after moving (newPos).
    #     newScaredTimes holds the number of moves that each ghost will remain
    #     scared because of Pacman having eaten a power pellet.
    #
    #     Print out these variables to see what you're getting, then combine them
    #     to create a masterful evaluation function.
    #     """
    #     # Useful information you can extract from a GameState (pacman.py)
    #     successorGameState = currentGameState.generatePacmanSuccessor(action)
    #     newPos = successorGameState.getPacmanPosition()
    #     newFood = successorGameState.getFood()
    #     newGhostStates = successorGameState.getGhostStates()
    #     newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #
    #     "*** YOUR CODE HERE ***"
    #     return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        optimalAction = self.maxValue(gameState=gameState, depth=0, agentIndex=0)[1]
        return optimalAction

    def is_game_end(self, gameState, depth, agentIndex):
        """
        Helper function to determine if we reached a leaf node in the state search tree

        Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.

        Outputs:
            [boolean]: win/loss state from the initialization of the problem.
            [list]: the list of legal actions.
            [int]: the depth to expand the search tree.
        """

        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agentIndex) is 0:
            return gameState.getLegalActions(agentIndex)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def maxValue(self, gameState, depth, agentIndex):
        """
        Helper function to determine the maximizing backed up optimalAction of a state.

        Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.

        Outputs:
            [tuple(float, string)]: finds the max minimax optimalAction for a gameState node and the corresponding action.
        """

        optimalAction = (float('-Inf'), None)
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successors = gameState.generateSuccessor(agentIndex, action)
            agentNums = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = (depth + 1) % agentNums
            succAction = self.minmax(gameState=successors, depth=expand, agentIndex=currentPlayer)
            optimalAction = max([optimalAction, (succAction, action)], key=lambda idx: idx[0])
        return optimalAction

    def minValue(self, gameState, depth, agentIndex):
        """
        Helper function to determine the minimizing backed up optimalAction of a state. T

        Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.

        Outputs:
            [tuple(float, string)]: finds the minimum minimax optimalAction for a gameState node and the corresponding action.
        """

        optimalAction = (float('+Inf'), None)
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successors = gameState.generateSuccessor(agentIndex, action)
            agentNums = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = (depth + 1) % agentNums
            succAction = self.minmax(gameState=successors, depth=expand, agentIndex=currentPlayer)
            optimalAction = min([optimalAction, (succAction, action)], key=lambda idx: idx[0])
        return optimalAction

    def minmax(self, gameState, depth, agentIndex):
        """
        Helper function that determines which agents's turn it is (MAX agent: Pacman, MIN agents: ghosts)
        and traverses the tree to the leaves and backs up the state's utility value.

        Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.

        Outputs:
            [float]: the utility value of State.
        """

        if self.is_game_end(gameState=gameState, depth=depth, agentIndex=agentIndex):
            return self.evaluationFunction(gameState)
        elif agentIndex is 0:
            return self.maxValue(gameState=gameState, depth=depth, agentIndex=agentIndex)[0]
        else:
            return self.minValue(gameState=gameState, depth=depth, agentIndex=agentIndex)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float('-Inf')  # MAX's best option on path to root
        beta = float('+Inf')  # MIN's best option on path to root
        depth = 0
        optimalAction = self.maxValue(gameState=gameState, depth=depth, agentIndex=0, alpha=alpha, beta=beta)
        return optimalAction[1]

    def is_game_end(self, gameState, depth, agentIndex):
        """
        Helper function to determine if we reached a leaf node in the state search tree

        Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.

        Outputs:
            [boolean]: win/loss state from the initialization of the problem.
            [list]: the list of legal actions.
            [int]: the depth to expand the search tree.
        """

        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agentIndex) is 0:
            return gameState.getLegalActions(agentIndex)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def maxValue(self, gameState, depth, agentIndex, alpha, beta):
        """
        Helper function that determine the maximizing backed up optimalAction of a state.
        Creating an iterable object such as a list and specify the key with which we make the comparison
        for the maximum optimalAction which is the float optimalAction in the first position of the tuple hence the idx[0].
        Additionally by using the alpha factor we can prune whole game-state subtrees.

        Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.
            alpha {[float]}: the float optimalAction which represents the minimum score that the maximizing player is assured of.
            beta {[float]}: the float optimalAction which represents the maximum score that the minimizing player is assured of.

        Outputs:
            [tuple(float, string)]: finds the max minimax optimalAction for a gameState node and the corresponding action.
        """

        optimalAction = (float('-Inf'), None)
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successors = gameState.generateSuccessor(agentIndex, action)
            agentNums = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = expand % agentNums
            succAction = self.minmax(
                gameState=successors, depth=expand, agentIndex=currentPlayer, alpha=alpha, beta=beta)
            optimalAction = max([optimalAction, (succAction, action)], key=lambda idx: idx[0])
            if optimalAction[0] > beta:
                return optimalAction
            alpha = max(alpha, optimalAction[0])
        return optimalAction

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        """
        Helper function that determine the minimizing backed up optimalAction of a state.
        Creating an iterable object such as a list and specify the key with which we make the comparison
        for the minimum optimalAction which is the float optimalAction in the first position of the tuple hence the idx[0].
        Additionally by using the beta factor we can prune whole game-state subtrees.

         Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.
            alpha {[float]}: the float optimalAction which represents the minimum score that the maximizing player is assured of.
            beta {[float]}: the float optimalAction which represents the maximum score that the minimizing player is assured of.

        Outputs:
            [tuple(float, string)]: finds the minimum minimax optimalAction for a gameState node and the corresponding action.
        """

        optimalAction = (float('+Inf'), None)
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successors = gameState.generateSuccessor(agentIndex, action)
            agentNum = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = expand % agentNum
            succAction = self.minmax(
                gameState=successors, depth=expand, agentIndex=currentPlayer, alpha=alpha, beta=beta)
            optimalAction = min([optimalAction, (succAction, action)], key=lambda idx: idx[0])
            if optimalAction[0] < alpha:
                return optimalAction
            beta = min(beta, optimalAction[0])
        return optimalAction

    def minmax(self, gameState, depth, agentIndex, alpha, beta):
        """
        Helper function that determines which agents's turn it is (MAX agent: Pacman, MIN agents: ghosts)
        and traverses the tree to the leaves and backs up the state's utility value.

        Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.
            alpha {[float]}: the float optimalAction which represents the minimum score that the maximizing player is assured of.
            beta {[float]}: the float optimalAction which represents the maximum score that the minimizing player is assured of.

        Outputs:
            [tuple(float, string)]: finds the max minimax optimalAction for a gameState node and the corresponding action.
        """

        if self.is_game_end(gameState=gameState, depth=depth, agentIndex=agentIndex):
            return self.evaluationFunction(gameState)
        elif agentIndex is 0:
            return self.maxValue(gameState=gameState, depth=depth, agentIndex=agentIndex, alpha=alpha, beta=beta)[0]
        else:
            return self.minValue(gameState=gameState, depth=depth, agentIndex=agentIndex, alpha=alpha, beta=beta)[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        optimalAction = self.maxValue(gameState=gameState, depth=0, agentIndex=0)[1]
        return optimalAction

    def is_end_game(self, gameState, depth, agentIndex):

        """
        Helper function to determine if we reached a leaf node in the state search tree

        Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.

        Outputs:
            [boolean]: win/loss state from the initialization of the problem.
            [list]: the list of legal actions.
            [int]: the depth to expand the search tree.
        """

        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agentIndex) is 0:
            return gameState.getLegalActions(agentIndex)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def maxValue(self, gameState, depth, agentIndex):
        """
        Helper function to determine the maximizing backed up optimalAction of a state.

        Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.

        Outputs:
            [tuple(float, string)]: finds the max minimax optimalAction for a gameState node and the corresponding action.
        """

        optimalAction = (float('-Inf'), None)
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successors = gameState.generateSuccessor(agentIndex, action)
            agentNum = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = (depth + 1) % agentNum
            succAction = self.expectMaxValue(gameState=successors, depth=expand, agentindex=currentPlayer)
            optimalAction = max(
                [optimalAction, (succAction, action)], key=lambda idx: idx[0])
        return optimalAction

    def expectedValue(self, gameState, depth, agentIndex):
        """
        Helper function that determine the average backed up optimalAction of a state.

                Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.

        Outputs:
            [float]: The average expected utilities optimalAction of children
        """

        optimalAction = list()
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successors = gameState.generateSuccessor(agentIndex, action)
            agentNum = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = (depth + 1) % agentNum
            succAction = self.expectMaxValue(gameState=successors, depth=expand, agentindex=currentPlayer)
            optimalAction.append(succAction)
        value = sum(optimalAction) / len(optimalAction)
        return value

    def expectMaxValue(self, gameState, depth, agentindex):
        """
        Helper function that determines which agents's turn it is (MAX agent: Pacman, MIN agents: ghosts)
        and traverses the tree to the leaves and backs up the state's utility value.

      Arguments:
            gameState {[MultiagentTreeState object]} : represents the state of the problem.
            depth {[int]}: the depth of the search tree.
            agent_idx {[int]}: the index of pacman or ghosts, testing for remaining legal actions.

        Outputs:
            [float]: the utility value of State.
        """

        if self.is_end_game(gameState=gameState, depth=depth, agentIndex=agentindex):
            return self.evaluationFunction(gameState)
        elif agentindex is 0:
            return self.maxValue(gameState=gameState, depth=depth, agentIndex=agentindex)[0]
        else:
            return self.expectedValue(gameState=gameState, depth=depth, agentIndex=agentindex)


# Question 5 (not required)
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

# Abbreviation
# better = betterEvaluationFunction
