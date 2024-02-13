# Teilnehmer: Hansen/Reschke
# Parameter maxWaitTime kann reduziert werden, um schneller zu testen (mit möglicherweise entsprechend reduzierter Genauigkeit)

from base_player import BasePlayer

import numpy as np
import time
import sys
import random
import copy

import pygame

from hexgame import HexBoard
from hexgame import HexGame
from random_player import RandomPlayer

class HexNode:
    """
    Class to represent a node in the Monte Carlo search tree.

    The node contains a representation of the current board, as well as some other information about the state of the game and the search tree.

    It also contains information about the value of the node in Monte Carlo tree search.
    """
    def __init__(self, board, current_player):
        self.parent = None
        self.children = list()
        self.UCTScore = 1
        self.thisBoard = board
        self.numberOfVisits = 0
        self.numberOfWins = 0
        self.currentPlayer = current_player
        self.lastMove = (0,0)
        self.AMAFNumberOfVisits = 0
        self.AMAFNumberOfWins = 0

    def setParent(self, node):
        self.parent = node

    def getParent(self):
        return self.parent

    def addChild(self, node):
        self.children.append(node)

    def getBoard(self):
        return self.thisBoard
    
    def getDiscountedUCTScore(self):
      """
      Returns the UCT score of the current node discounted by a factor which depends on the number of UCT visits.

      More visits results in a higher score.

      This is useful for RAVE evaluation where the number of UCT visits to the best node may be small.
      """
      decayFactor = 10
      return self.UCTScore - ((1 / (1 + decayFactor * np.log(np.maximum(self.numberOfVisits,1)))) * self.UCTScore)
    
    def getUCTScore(self):
        return self.UCTScore
    
    def getUCTVisits(self):
        return self.numberOfVisits
    
    def getAMAFVisits(self):
        return self.AMAFNumberOfVisits
    
    def getCurrentPlayer(self):
        return self.currentPlayer
    
    def updateUCT(self, result):
        if result:
            self.numberOfWins += 1
        self.numberOfVisits += 1
        self.UCTScore = self.numberOfWins / self.numberOfVisits

    def updateAMAF(self, result):
        if result:
            self.AMAFNumberOfWins += 1
        self.AMAFNumberOfVisits += 1

    def getRAVEScore(self):
        """
        Returns the RAVE score of the current node. The RAVE score is a combination of the AMAF and UCT win ratios. The exact score depends on the weighting factor, which decays
        proportional to the number of UCT visits. This means that the more the node is chosen first (rather than as a later AMAF move) the more confidence the algorithm places on 
        the UCT estimate.
        """
        explorationConstant = 1000
        AMAFWeightingFactor = self.calculateAMAFWeightingFactor()
        return (self.UCTScore + explorationConstant * np.sqrt(np.log(np.maximum(self.parent.getUCTVisits(),1) / np.maximum(self.numberOfVisits,1))) * (1 - AMAFWeightingFactor) + (self.AMAFNumberOfWins / np.maximum(self.AMAFNumberOfVisits,1)) * AMAFWeightingFactor)

    def numberOfChildren(self):
        return len(self.children)
    
    def calculateAMAFWeightingFactor(self):
        """
        Calculates and returns the AMAF weighting factor. This decreases with increasing number of visits.

        This means that the second term in the RAVE score, which is the product of this factor with the AMAF win ratio, also decreases in importance relative to the first term.
        """
        equivalenceParameter = 10000
        return np.sqrt(equivalenceParameter / ((3 * self.numberOfVisits) + equivalenceParameter))
    
    def setUCTScore(self, val):
        self.UCTScore = val

    def setAMAFWins(self, val):
        self.AMAFNumberOfWins = val

    def getLastMove(self):
        return self.lastMove
    
    def setLastMove(self, tup):
        self.lastMove = tup

class MCPlayer(BasePlayer):
  def __init__(self, swap_fun=lambda board, *args: False):
    self.currentPlayer = 1
    self.simulatedPlayer = 1
    self._swap_fun = swap_fun
    self.searchTree = list()
    self.threshhold = 3
    self.turnLimit = 100000
    self.maxWaitTime = 27
    self.thisBoard = None
    self.searchNumber = 100
    self.lastMove = (0,0)
    # Not necessary for puzzles
    self.useInfill = True
    # RAVE and attention do not currently add much value. I would like to try adjusting the parameters before the tournament.
    self.useRAVE = False
    self.useAttention = False

  def claim_swap(self, board, *args) -> bool:
    """
    Chooses whether to swap depending on first enemy move.

    Parameters
    ----------
    board : HexBoard
        Current state of the Hex board.
    args : tuple, optional
        Further parameters in tuple possibly required by heuristic.
    
    Returns
    -------
    res : bool
        Chooses whether or not to swap tiles.

    """

    """
    Based on known winning/losing positions from existing research. Swaps if the opponent's first move is to a known winning tile.
    """
    res = False
    #self.currentPlayer = 2
    #self.simulatedPlayer = self.currentPlayer
    (xdim, ydim) = board.dim()
    x5Positions = [(0,3),(0,4),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,0),(4,1)]
    x7Positions = [(0,3),(0,5),(0,6),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4),(2,5),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4),(4,5),(5,1),(5,2),(5,3),(5,4),(6,0),(6,1),(6,3)]
    x9Positions = [(1,7),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(3,2),(3,3),(3,4),(3,5),(3,6),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(5,2),(5,3),(5,4),(5,5),(5,6),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(7,1)]
    x11Positions = [(0,10),(1,9),(1,10),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),(5,10),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(6,8),(6,9),(6,10),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(7,10),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8),(8,9),(9,0),(9,1),(10,0)]
    x13Positions = [(0,12),(1,11),(1,12),(2,1),(2,2),(2,3),(2,4),(2,9),(2,10),(2,11),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(3,12),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),(5,10),(5,11),(5,12),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(6,12),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8),(8,9),(8,10),(8,11),(8,12),(9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9),(9,10),(9,11),(9,12),(10,1),(10,2),(10,3),(10,8),(10,9),(10,10),(10,11),(11,0),(11,1),(12,0)]
    x15Positions = [(0,14),(1,13),(1,14),(2,1),(2,2),(2,3),(2,4),(2,9),(2,10),(2,11),(2,12),(2,13),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(3,12),(3,13),(3,14),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(4,13),(4,14),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),(5,10),(5,11),(5,12),(5,13),(5,14),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(6,12),(6,13),(6,14),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12),(7,13),(7,14),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8),(8,9),(8,10),(8,11),(8,12),(8,13),(8,14),(9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9),(9,10),(9,11),(9,12),(9,13),(9,14),(10,0),(10,1),(10,2),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12),(10,13),(10,14),(11,0),(11,1),(11,2),(11,3),(11,4),(11,5),(11,6),(11,7),(11,8),(11,9),(11,10),(11,11),(11,12),(11,13),(11,14),(12,1),(12,2),(12,3),(12,4),(12,5),(12,10),(12,11),(12,12),(12,13),(13,0),(13,1),(14,0)]
    x17Positions = [(0,14),(1,13),(1,14),(2,1),(2,2),(2,3),(2,4),(2,9),(2,10),(2,11),(2,12),(2,13),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(3,12),(3,13),(3,14),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(4,13),(4,14),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),(5,10),(5,11),(5,12),(5,13),(5,14),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(6,12),(6,13),(6,14),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12),(7,13),(7,14),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8),(8,9),(8,10),(8,11),(8,12),(8,13),(8,14),(9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9),(9,10),(9,11),(9,12),(9,13),(9,14),(10,0),(10,1),(10,2),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12),(10,13),(10,14),(11,0),(11,1),(11,2),(11,3),(11,4),(11,5),(11,6),(11,7),(11,8),(11,9),(11,10),(11,11),(11,12),(11,13),(11,14),(12,1),(12,2),(12,3),(12,4),(12,5),(12,10),(12,11),(12,12),(12,13),(13,0),(13,1),(14,0)]
    x19Positions = [(0,18),(1,17),(1,18),(2,1),(2,2),(2,3),(2,4),(2,14),(2,15),(2,16),(2,17),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(3,12),(3,13),(3,14),(3,15),(3,16),(3,17),(3,18),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(4,13),(4,14),(4,15),(4,16),(4,17),(4,18),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),(5,10),(5,11),(5,12),(5,13),(5,14),(5,15),(5,16),(5,17),(5,18),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(6,12),(6,13),(6,14),(6,15),(6,16),(6,17),(6,18),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12),(7,13),(7,14),(7,15),(7,16),(7,17),(7,18),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8),(8,9),(8,10),(8,11),(8,12),(8,13),(8,14),(8,15),(8,16),(8,17),(8,18),(9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9),(9,10),(9,11),(9,12),(9,13),(9,14),(9,15),(9,16),(9,17),(9,18),(10,0),(10,1),(10,2),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12),(10,13),(10,14),(10,15),(10,16),(10,17),(10,18),(11,0),(11,1),(11,2),(11,3),(11,4),(11,5),(11,6),(11,7),(11,8),(11,9),(11,10),(11,11),(11,12),(11,13),(11,14),(11,15),(11,16),(11,17),(11,18),(12,0),(12,1),(12,2),(12,3),(12,4),(12,5),(12,6),(12,7),(12,8),(12,9),(12,10),(12,11),(12,12),(12,13),(12,14),(12,15),(12,16),(12,17),(12,18),(13,0),(13,1),(13,2),(13,3),(13,4),(13,5),(13,6),(13,7),(13,8),(13,9),(13,10),(13,11),(13,12),(13,13),(13,14),(13,15),(13,16),(13,17),(13,18),(14,0),(14,1),(14,2),(14,3),(14,4),(14,5),(14,6),(14,7),(14,8),(14,9),(14,10),(14,11),(14,12),(14,13),(14,14),(14,15),(14,16),(14,17),(14,18),(15,0),(15,1),(15,2),(15,3),(15,4),(15,5),(15,6),(15,7),(15,8),(15,9),(15,10),(15,11),(15,12),(15,13),(15,14),(15,15),(15,16),(15,17),(15,18),(16,1),(16,2),(16,3),(16,4),(16,14),(16,15),(16,16),(16,17)]
    if xdim == 5:
      positions = x5Positions
    elif xdim == 7:
      positions = x7Positions  
    elif xdim == 9:
      positions = x9Positions  
    elif xdim == 11:
      positions = x11Positions
    elif xdim == 13:
      positions = x13Positions
    elif xdim == 15:
      positions = x15Positions
    elif xdim == 17:
      positions = x17Positions
    elif xdim == 19:
      positions = x19Positions
    else:
      positions = list()
    for i in range(xdim):
      for j in range(ydim):
        if ((i, j) in positions) and (board.get_tile(i, j) == 3 - self.currentPlayer):
          res = True
          self.lastMove = (i,j)
    return res

  def choose_tile(self, board, *args) -> tuple:
    """
    Chooses a tile based on Monte Carlo tree search.

    In the case that a one-move win is possible, choose this move.

    Otherwise, we search the tree until the time limit ends. Each time that a leaf node is reached, we carry out Monte Carlo simulations and adjust the UCT score
    of all visited nodes.

    The final choice is the child node with the best UCT score, that is, the largest proportion of wins in the Monte Carlo simulations.

    If there are multiple moves with the best UCT score, choose one that completes an existing virtual connection.

    Parameters
    ----------
    board : HexBoard
        Current state of the Hex board.
    args : tuple, optional
        Further parameters in tuple possibly required by heuristic.
    
    Returns
    -------
    res : tuple (int, int)
        Returns coordinates of the move with the best UCT score.

    """
    (xdim, ydim) = board.dim()
    newMove = BasePlayer.random_choice(board, *args)
    self.currentPlayer = self.get_id()
    self.simulatedPlayer = self.get_id()

    thisBoard = board
    originalBoard = copy.deepcopy(thisBoard)

    if self.useInfill:
      thisBoard = self.InfillBoardState(thisBoard)

    currentBoardState = None
    for node in self.searchTree:
      if (node.getBoard().board == thisBoard.board).all():
        currentBoardState = node
        break
        #print('Found matching node in the search tree. Current board state: {}'.format(currentBoardState.getBoard().board))
    if (currentBoardState == None):
      currentBoardState = HexNode(thisBoard, self.currentPlayer)
      currentBoardState.setLastMove(self.lastMove)
      self.searchTree.append(currentBoardState)
    """
    First: Produce a list of all possible moves. Check whether a win is possible. If it's possible to immediately win, take this move.
    """
    validMoves = list()
    for i in range(xdim):
      for j in range(ydim):
        if thisBoard.get_tile(i, j) == 0:
          newMove = (i, j)
          validMoves.append(newMove)
    for move in validMoves:
      simulatedBoard = copy.deepcopy(thisBoard)
      (i, j) = move
      simulatedBoard.set_tile(i, j, self.currentPlayer)
      dummy_player1 = RandomPlayer()
      dummy_player2 = RandomPlayer()
      newGame = HexGame(simulatedBoard.board, dummy_player1, dummy_player2)
      if (newGame.check_finish() == self.currentPlayer):
        return move

    """
    Produce a list of all incomplete bridges.
    """
    virtualConnections = self.calculateAllBridges(thisBoard)
    #print('Bridges for current player: {}'.format(virtualConnections))

    """
    Carry out the Monte Carlo tree search until time runs out.
    """  
    elapsed = 0
    start = None
    for n in range(self.turnLimit):
      if elapsed < self.maxWaitTime:
        self.CarryOutSearch(currentBoardState)
        if n%self.searchNumber == 0:
          if start:
            end = time.time()
            elapsed = elapsed + end - start
            print('Elapsed time: {}'.format(elapsed))
          start = time.time()
    visitedNodes = list()
    """
    Examine all children of the current node and choose the one with the best UCT score. This is the next move.

    If there is a move with UCT score 1, prefer to complete an existing virtual connection (as determined by bridge templates).

    This is useful because the solver normally cannot otherwise distinguish between moves with UCT score 1.
    Hence given a game state with a guaranteed win it is indifferent between all moves.
    """  
    if currentBoardState.children:
      #print('Number of children: {}'.format(len(currentBoardState.children)))
      #for child in currentBoardState.children:
        #print('UCT score of this child: {}'.format(child.getUCTScore()))
      currentLevel = 0
      maxUCTScore = -1
      for child in currentBoardState.children:
        if ((child.getUCTScore() > 0.9999) & (child.getLastMove() in virtualConnections)):
          bestChild = child
          #print('Bridge with UCT score 1 found at: {}'.format(child.getLastMove()))
          break
        if (self.useRAVE == False):
          if (child.getUCTScore() > maxUCTScore):
            maxUCTScore = child.getUCTScore()
            bestChild = child
        if (self.useRAVE == True):
          if (child.getUCTScore() > maxUCTScore):
            #print('Switched to new node with UCT score {}'.format(child.getUCTScore()))
            maxUCTScore = child.getUCTScore()
            bestChild = child
      visitedNodes.append(currentBoardState)
      newBoardState = bestChild
      print('Searching at level: {}'.format(currentLevel))
      print('Size of search tree: {}'.format(len(self.searchTree)))
      for i in range(xdim):
        for j in range(ydim):
          if (currentBoardState.getBoard().get_tile(i, j) == 0) & (newBoardState.getBoard().get_tile(i, j) != 0):
            newMove = (i, j)
            print('Current player: {}. We played: {}. This move had total UCT score {} (discounted: {}) and RAVE score {} with {} UCT visits and {} AMAF visits.'.format(self.currentPlayer, newMove, newBoardState.getUCTScore(), newBoardState.getDiscountedUCTScore(), newBoardState.getRAVEScore(), newBoardState.getUCTVisits(), newBoardState.getAMAFVisits()))
    self.adjustTimeLimit(originalBoard)
    return newMove
  
  def CarryOutSearch(self, baseNode):
    """
    Carries out one loop of Monte Carlo tree search.

    In one loop, the program traverses the search tree to a leaf node. Starting at this node, a simulated game is carried out. The results are used to update the UCT
    values in the search tree.

    Parameters
    ----------
    baseNode: The node in the search tree at which the search begins.
    
    Returns
    -------
    Nothing. Updates are made within the function.

    """
    thisNode = baseNode
    visitedNodes = list()
    visitedNodes.append(thisNode)
    siblingsOfVisitedNodes = list()
    siblingsOfVisitedNodes.append(thisNode)
    (xdim, ydim) = thisNode.getBoard().dim()

    """
    Choose the best child of the current node. In general, we choose the child with the highest UCT score, but sometimes choose randomly to allow for exploration.
    """
    if (self.useRAVE == False):
      while thisNode.children:
        if (thisNode.getCurrentPlayer() == self.currentPlayer):
          maxUCTScore = -1
          for child in thisNode.children:
            if child.getUCTScore() > maxUCTScore:
              maxUCTScore = child.getUCTScore()
              bestChild = child
          if (random.uniform(0,1) < 0.10):
            thisNode = random.choice(thisNode.children)
          else:
            thisNode = bestChild
        elif (thisNode.getCurrentPlayer() == 3 - self.currentPlayer):
          minUCTScore = 2
          for child in thisNode.children:
            if child.getUCTScore() < minUCTScore:
              minUCTScore = child.getUCTScore()
              bestChild = child
          if (random.uniform(0,1) < 0.10):
            thisNode = random.choice(thisNode.children)
          else:
            thisNode = bestChild
        visitedNodes.append(thisNode)
    elif (self.useRAVE == True):
      while thisNode.children:
        if (thisNode.getCurrentPlayer() == self.currentPlayer):
          maxRAVEScore = -1
          bestChild = random.choice(thisNode.children)
          for child in thisNode.children:
            siblingsOfVisitedNodes.append(child)
            if (child.getRAVEScore() > maxRAVEScore):
              maxRAVEScore = child.getRAVEScore()
              bestChild = child
          if (random.uniform(0,1) < 0.10):
            thisNode = random.choice(thisNode.children)
          else:
            thisNode = bestChild
        elif (thisNode.getCurrentPlayer() == 3 - self.currentPlayer):
          minRAVEScore = 2
          bestChild = random.choice(thisNode.children)
          for child in thisNode.children:
            siblingsOfVisitedNodes.append(child)
            if (child.getRAVEScore() < minRAVEScore):
              minRAVEScore = child.getRAVEScore()
              bestChild = child
          if (random.uniform(0,1) < 0.10):
            thisNode = random.choice(thisNode.children)
          else:
            thisNode = bestChild
        visitedNodes.append(thisNode)      
    """
    Upon reaching a leaf, we carry out a simulation and update the UCT scores of all traversed nodes. If RAVE is enabled, we update the RAVE scores of all siblings.

    While updating the UCT scores we also create new child nodes for any node with a number of visits reaching the threshhold.
    """
    (winner, simulatedBoard) = self.CarryOutSimulation(thisNode)
    result = (winner == self.currentPlayer)
    for node in visitedNodes:
        node.updateUCT(result)
        nodeBoard = node.getBoard()
        if (node.numberOfVisits == self.threshhold):
            for i in range(xdim):
                for j in range(ydim):
                    if nodeBoard.get_tile(i, j) == 0:
                        newBoard = copy.deepcopy(nodeBoard)
                        newBoard.set_tile(i, j, node.getCurrentPlayer())
                        newChild = HexNode(newBoard, 3 - node.getCurrentPlayer())
                        if not (newChild.getCurrentPlayer() == self.currentPlayer):
                          newChild.setUCTScore(0)
                          newChild.setAMAFWins(0)
                        lastMove = (i,j)
                        newChild.setLastMove(lastMove)
                        self.searchTree.append(newChild)
                        newChild.setParent(thisNode)
                        node.addChild(newChild)
    if (self.useRAVE == True):
      for node in siblingsOfVisitedNodes:
        for i in range(xdim):
          for j in range(ydim):
            if (simulatedBoard.get_tile(i,j) == self.currentPlayer) and (node.getBoard().get_tile(i,j) == self.currentPlayer):
              node.updateAMAF(result)

  def CarryOutSimulation(self, thisNode):
    """
    Carries out a single Monte Carlo simulation.

    Parameters
    ----------
    thisNode: The current (leaf) node in the search tree.
    
    Returns
    -------
    Winner of the simulated game.

    """
    validMoves = list()
    simulatedBoard = copy.deepcopy(thisNode.getBoard())
    (xdim, ydim) = simulatedBoard.dim()
    self.simulatedPlayer = thisNode.getCurrentPlayer()
    for i in range(xdim):
      for j in range(ydim):
        if simulatedBoard.get_tile(i, j) == 0:
          newMove = (i, j)
          validMoves.append(newMove)
    """
    Play all possible moves in order, alternating between players. Use the analyzeMoves function to choose the next move.

    The attention mechanic is a heuristic which penalizes moves too far from the last move played. Move with a low attention score may be randomly rerolled.
    """
    while validMoves:
      nextMove = self.analyzeMoves(simulatedBoard)
      if (self.useAttention == True):
        attentionDecayFactor = self.calculateAttentionLoss(thisNode.getLastMove(), nextMove)
        while (random.uniform(0,1) > (1 - attentionDecayFactor)):
          nextMove = self.analyzeMoves(simulatedBoard)
          attentionDecayFactor = self.calculateAttentionLoss(thisNode.getLastMove(), nextMove)
      if nextMove:
        (i, j) = nextMove
        simulatedBoard.set_tile(i, j, self.simulatedPlayer)
        self.simulatedPlayer = 3 - self.simulatedPlayer
        moveToRemove = (i, j)
        if moveToRemove in validMoves:
            validMoves.remove(moveToRemove)
    dummy_player1 = RandomPlayer()
    dummy_player2 = RandomPlayer()
    newGame = HexGame(simulatedBoard.board, dummy_player1, dummy_player2)
    return (newGame.check_finish(), simulatedBoard)
  
  def analyzeMoves(self, board):
    """
    Chooses from among the available moves. Use a mix of heuristics and chance selection.

    Parameters
    ----------
    board: The current state of the board.
    
    Returns
    -------
    The coordinates of the chosen move.
    """

    """
    Use two heuristics to navigate playouts: savebridge and breakbridge.

    Savebridge: If a move of the opposing player threatens a bridge, play into the open space to form a connection.

    Breakbridge: If a move of the opposing player leaves a bridge threatened, play into the open space to break the virtual connection.

    If these two templates don't apply, choose randomly from among the remaining moves.
    """
    (xdim, ydim) = board.dim()
    threatenedBridges = list()
    breakableBridges = list()
    emptyTiles = list()
    #print('New simulation')
    for i in range(xdim):
      for j in range(ydim):
        if (board.get_tile(i,j) == self.simulatedPlayer):
          # Fall 1: Brücke rechts oben
          if (i-1 > 0) & (j+2 < ydim):
            if (board.get_tile(i-1,j+2) == self.simulatedPlayer):
              if (board.get_tile(i-1,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
              if (board.get_tile(i,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
          # Fall 2: Brücke rechts unten
          if (i+1 < xdim) & (j+1 < ydim):
            if (board.get_tile(i+1,j+1) == self.simulatedPlayer):
              if (board.get_tile(i,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
              if (board.get_tile(i+1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
          # Fall 3: Brücke lotrecht oben
          if (i-1 > 0) & (j+1 < ydim):
            if (board.get_tile(i-2,j+1) == self.simulatedPlayer):
              if (board.get_tile(i-1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
              if (board.get_tile(i-1,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
            # Fall 4: Am unteren Rand
          if (i == xdim - 2) & (j > 0) & (self.simulatedPlayer == 1):
            if (board.get_tile(i+1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
            if (board.get_tile(i+1,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
            # Fall 5: Am oberen Rand
          if (i == 1) & (j < ydim - 1) & (self.simulatedPlayer == 1):
            if (board.get_tile(i-1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
            if (board.get_tile(i-1,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
            # Fall 6: Am linken Rand
          if (i < xdim - 1) & (j == 1) & (self.simulatedPlayer == 2):
            if (board.get_tile(i+1,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i,j-1) == 0):
                move = (i,j-1)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
            if (board.get_tile(i,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
            # Fall 7: Am rechten Rand
          if (i > 0) & (j == ydim - 2) & (self.simulatedPlayer == 2):
            if (board.get_tile(i-1,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
            if (board.get_tile(i,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                #print('Found threatened bridge')
                threatenedBridges.append(move)
        if (board.get_tile(i,j) == (3 - self.simulatedPlayer)):
          # Fall 1: Brücke rechts oben
          if (i-1 > 0) & (j+2 < ydim):
            if (board.get_tile(i-1,j+2) == (3 - self.simulatedPlayer)):
              if (board.get_tile(i-1,j+1) == self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                breakableBridges.append(move)
              if (board.get_tile(i,j+1) == self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                breakableBridges.append(move)
          # Fall 2: Brücke rechts unten
          if (i+1 < xdim) & (j+1 < ydim):
            if (board.get_tile(i+1,j+1) == (3 - self.simulatedPlayer)):
              if (board.get_tile(i,j+1) == self.simulatedPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                breakableBridges.append(move)
              if (board.get_tile(i+1,j) == self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                breakableBridges.append(move)
          # Fall 3: Brücke lotrecht oben
          if (i-1 > 0) & (j+1 < ydim):
            if (board.get_tile(i-2,j+1) == (3 - self.simulatedPlayer)):
              if (board.get_tile(i-1,j) == self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                breakableBridges.append(move)
              if (board.get_tile(i-1,j+1) == self.simulatedPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                breakableBridges.append(move)
          # Fall 4: Am unteren Rand
          if (i == xdim - 2) & (j > 0) & (self.simulatedPlayer == 2):
            if (board.get_tile(i+1,j) == self.simulatedPlayer) & (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                #print('Found threatened bridge')
                breakableBridges.append(move)
            if (board.get_tile(i+1,j-1) == self.simulatedPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                #print('Found threatened bridge')
                breakableBridges.append(move)
          # Fall 5: Am oberen Rand
          if (i == 1) & (j < ydim - 1) & (self.simulatedPlayer == 2):
            if (board.get_tile(i-1,j) == self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i+1,j-1)
                #print('Found threatened bridge')
                breakableBridges.append(move)
            if (board.get_tile(i-1,j+1) == self.simulatedPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                #print('Found threatened bridge')
                breakableBridges.append(move)
          # Fall 6: Am linken Rand
          if (i < xdim - 1) & (j == 1) & (self.simulatedPlayer == 1):
            if (board.get_tile(i+1,j-1) == self.simulatedPlayer) & (board.get_tile(i,j-1) == 0):
                move = (i,j-1)
                #print('Found threatened bridge')
                breakableBridges.append(move)
            if (board.get_tile(i,j-1) == self.simulatedPlayer) & (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                #print('Found threatened bridge')
                breakableBridges.append(move)
          # Fall 7: Am rechten Rand
          if (i > 0) & (j == ydim - 2) & (self.simulatedPlayer == 1):
            if (board.get_tile(i-1,j+1) == self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                #print('Found threatened bridge')
                breakableBridges.append(move)
            if (board.get_tile(i,j+1) == self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                #print('Found threatened bridge')
                breakableBridges.append(move)
        elif (board.get_tile(i,j) == 0):
          move = (i,j)
          emptyTiles.append(move)
    """
    If at least one threatened bridge was found, choose randomly from the moves which save one of these bridges.

    Otherwise, choose randomly from among all other valid moves.

    If attention is used, rechoose with probability equal to the decay factor.
    """
    if threatenedBridges:
      return random.choice(threatenedBridges)
    elif breakableBridges:
      return random.choice(breakableBridges)
    elif emptyTiles:
      return random.choice(emptyTiles)
    else:
      return None
    
  def calculateAllBridges(self, board):
    """
    Find all bridges which could be completed by the current player.

    Parameters
    ----------
    board: The current state of the board.
    
    Returns
    -------
    virtualConnections: List of all uncompleted bridges on the board.
    """

    """
    Similar to analyzeMoves with the distinction that the bridge must not be threatened, and that all virtual connections are returned.
    """
    (xdim, ydim) = board.dim()
    virtualConnections = list()
    for i in range(xdim):
      for j in range(ydim):
        if (board.get_tile(i,j) == self.currentPlayer):
          # Fall 1: Brücke rechts oben
          if (i-1 > 0) & (j+2 < ydim):
            if (board.get_tile(i-1,j+2) == self.currentPlayer):
              if (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                virtualConnections.append(move)
              if (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                virtualConnections.append(move)
          # Fall 2: Brücke rechts unten
          if (i+1 < xdim) & (j+1 < ydim):
            if (board.get_tile(i+1,j+1) == self.currentPlayer):
              if (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                virtualConnections.append(move)
              if (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                virtualConnections.append(move)
          # Fall 3: Brücke lotrecht oben
          if (i-1 > 0) & (j+1 < ydim):
            if (board.get_tile(i-2,j+1) == self.simulatedPlayer):
              if (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                virtualConnections.append(move)
              if (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                virtualConnections.append(move)
            # Fall 4: Am unteren Rand
          if (i == xdim - 2) & (j > 0) & (self.simulatedPlayer == 1):
            if (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                virtualConnections.append(move)
            if (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                virtualConnections.append(move)
            # Fall 5: Am oberen Rand
          if (i == 1) & (j < ydim - 1) & (self.simulatedPlayer == 1):
            if (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                virtualConnections.append(move)
            if (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                virtualConnections.append(move)
            # Fall 6: Am linken Rand
          if (i < xdim - 1) & (j == 1) & (self.simulatedPlayer == 2):
            if (board.get_tile(i,j-1) == 0):
                move = (i,j-1)
                virtualConnections.append(move)
            if (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                virtualConnections.append(move)
            # Fall 7: Am rechten Rand
          if (i > 0) & (j == ydim - 2) & (self.simulatedPlayer == 2):
            if (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                virtualConnections.append(move)
            if (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                virtualConnections.append(move)
    return virtualConnections
    
  def InfillBoardState(self, board):
    """
    Use patterns to compute dead cells. Dead cells are cells which have no impact on the game and hence can be set to either color.

    Parameters
    ----------
    board: The current state of the board.
    
    Returns
    -------
    thisBoard: 'Infilled' hex board --- copy of the current board with dead cells filled by one of the two players.
    """
    #print('Calculating infill...')
    infilledTiles = list()
    thisBoard = copy.deepcopy(board)
    (xdim, ydim) = thisBoard.dim()
    for i in range(xdim):
      for j in range(ydim):
        if thisBoard.get_tile(i,j) == 0:
          if (i > 0) & (j > 0) & (i + 1 < xdim) & (j + 1 < ydim):
            # Muster 1.1: 4 tiles of the same color, first player
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i,j+1) == 1):
              #print('Match found! (test case 1) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 1.2: 4 tiles of the same color, first player
            if (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i+1,j) == 1):
              #print('Match found! (test case 2) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 1.3: 4 tiles of the same color, first player
            if (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i+1,j-1) == 1):
              #print('Match found! (test case 3) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 1.4: 4 tiles of the same color, first player
            if (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i,j-1) == 1):
              #print('Match found! (test case 4) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 1.5: 4 tiles of the same color, first player
            if (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1):
              #print('Match found! (test case 5) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 1.6: 4 tiles of the same color, first player
            if (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i-1,j+1) == 1):
              #print('Match found! (test case 6) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 2.1: 4 tiles of the same color, second player
            if (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i,j+1) == 2):
              #print('Match found! (test case 7) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 2.2: 4 tiles of the same color, second player
            if (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i+1,j) == 2):
              #print('Match found! (test case 8 ) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 2.3: 4 tiles of the same color, second player
            if (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j-1) == 2):
              #print('Match found! (test case 9) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 2.4: 4 tiles of the same color, second player
            if (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i,j-1) == 2):
              #print('Match found! (test case 10) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 2.5: 4 tiles of the same color, second player
            if (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i-1,j) == 2):
              #print('Match found! (test case 11) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 2.6: 4 tiles of the same color, second player
            if (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i-1,j+1) == 2):
              #print('Match found! (test case 12) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 3.1: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i-1,j) == 2):
              #print('Match found! (test case 13) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 3.2: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i-1,j+1) == 2):
              #print('Match found! (test case 14) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 3.3: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i,j+1) == 2):
              #print('Match found! (test case 15) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 3.4: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i+1,j) == 2):
              #print('Match found! (test case 16) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 3.5: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i+1,j-1) == 2):
              #print('Match found! (test case 17) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 3.6: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i,j-1) == 2):
              #print('Match found! (test case 18) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 4.1: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i-1,j) == 1):
              #print('Match found! (test case 19) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 4.2: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i-1,j+1) == 1):
              #print('Match found! (test case 20) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 4.3: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i,j+1) == 1):
              #print('Match found! (test case 21) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 4.4: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i+1,j) == 1):
              #print('Match found! (test case 22) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 4.5: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i+1,j-1) == 1):
              #print('Match found! (test case 23) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 4.6: 3 tiles of one color opposite 1 of the other color
            if (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i,j-1) == 1):
              #print('Match found! (test case 24) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
              move = (i,j)
              infilledTiles.append(move)
            # Muster 5.1: 2 tiles of one color opposite 2 of the other color
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j+1) == 2):
              #print('Match found! (test case 25) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 5.2: 2 tiles of one color opposite 2 of the other color
            if (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j-1) == 2):
              #print('Match found! (test case 26) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 5.3: 2 tiles of one color opposite 2 of the other color
            if (thisBoard.get_tile(i-1,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j+1) == 2):
              #print('Match found! (test case 27) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 5.4: 2 tiles of one color opposite 2 of the other color
            if (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i-1,j) == 2):
              #print('Match found! (test case 28) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 5.5: 2 tiles of one color opposite 2 of the other color
            if (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i-1,j) == 2):
              #print('Match found! (test case 29) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 5.6: 2 tiles of one color opposite 2 of the other color
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i,j+1) == 2):
              #print('Match found! (test case 30) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 6: tiles directly and above left (edge pattern)  
          if (i == xdim-1) & (j > 0):
            if (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i,j-1) == 1):
              #print('Match found! (test case 31) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 7: tiles to both sides (edge pattern)
          if (i == xdim-1) & (j > 0) & (j+1 < ydim):
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i,j+1) == 1):
              #print('Match found! (test case 32) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 8: tiles above left and directly right (edge pattern)
          if (i == xdim-1) & (j < ydim-1):
            if (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i,j+1) == 1):
              #print('Match found! (test case 33) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 9: tiles above and to both sides
          if (i == xdim-1) & (j < ydim-1):
            if (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i-1,j+1) == 2):
              #print('Match found! (test case 34) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
              move = (i,j)
              infilledTiles.append(move)
            # Muster 10: acute corner killed by single tile
          if (i == xdim-1) & (j == ydim-1):
            if (thisBoard.get_tile(i,j-1) == 1):
              #print('Match found! (test case 35) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # upper edge patterns
            # Muster 11: tiles directly and below right (edge pattern)
          if (i == 0) & (j+1 < ydim):
            if (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i,j+1) == 1):
              #print('Match found! (test case 36) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 12: tiles to both sides (edge pattern)
          if (i == 0) & (j > 0) & (j+1 < ydim):
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i,j+1) == 1):
              #print('Match found! (test case 37) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 13: tiles directly to the left and below to the right (edge pattern)
          if (i == 0) & (j > 0):
            if (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j-1) == 1):
              #print('Match found! (test case 38) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
              move = (i,j)
              infilledTiles.append(move)
            # Muster 14: tiles below to the left and right
          if (i == 0) & (j > 0):
            if (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j-1) == 2):
              #print('Match found! (test case 39) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 15: acute corner killed by single tile
          if (i == 0) & (j == 0):
            if (thisBoard.get_tile(i,j+1) == 1):
              #print('Match found! (test case 40) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # left edge patterns
            # Muster 16: counterpart to (6) on the left edge
          if (i < xdim - 1) & (j == 0):
            if (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j+1) == 2):
              #print('Match found! (test case 41) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 17: counterpart to (7) on the left edge
          if (i > 0) & (j == 0):
            if (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i,j+1) == 1):
              #print('Match found! (test case 42) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
            # Muster 18: counterpart to (8) on the left edge
          if (i == xdim-1) & (j < ydim-1):
            if (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i,j+1) == 2):
              #print('Match found! (test case 43) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 19: counterpart to (9) on the left edge
          if (i == xdim-1) & (j < ydim-1):
            if (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i-1,j+1) == 1):
              #print('Match found! (test case 44) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
              move = (i,j)
              infilledTiles.append(move)
            # Muster 20: counterpart to (10) on the left edge
          if (i == xdim-1) & (j == ydim-1):
            if (thisBoard.get_tile(i,j-1) == 2):
              #print('Match found! (test case 45) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
              move = (i,j)
              infilledTiles.append(move)
        # Ladder template --- not used
        #if (thisBoard.get_tile(i,j) == self.currentPlayer) & (self.currentPlayer == 2):
          #if (j > 0) & (j < ydim - 1) & (i > 0):
            #if (thisBoard.get_tile(i-1,j) == 3 - self.currentPlayer) & (thisBoard.get_tile(i-1,j-1) == 0) & (thisBoard.get_tile(i,j-1) == 0):
              #move = (i,j-1)
              #print('Continue ladder: {}'.format(move))
          #if (j < ydim - 2) & (i > 0):
            #if (thisBoard.get_tile(i-1,j+1) == 3 - self.currentPlayer) & (thisBoard.get_tile(i-1,j+2) == 0) & (thisBoard.get_tile(i,j+1) == 0):
              #move = (i,j+1)
              #print('Continue ladder: {}'.format(move))
    #print('Infilled tiles: {}'.format(infilledTiles))
    return thisBoard
  
  def calculateAttentionLoss(self, lastMove, nextMove):
    """
    Calculate the loss of 'attention' based on distance to the last move. Distance is the 'taxicab' distance, i.e., the number of moves necessary to connect the two tiles minus 1.

    Parameters
    ----------
    lastMove: Coordinates of the last played move.

    nextMove: Coordinates of the chosen next move
    
    Returns
    -------
    A numerical factor which increases with increasing distance of the next move from the last one played.

    The attention loss is 0 for the first two fields and then linear thereafter and is capped at 0.6.
    """
    decayRate = 0.1
    (i,j) = lastMove
    (ii,jj) = nextMove
    return np.minimum((np.maximum((np.absolute(i-ii) + np.absolute(j-jj) - 2), 0) * decayRate), 0.6)
  
  def adjustTimeLimit(self, board):
    (xdim, ydim) = board.dim()
    numberFilled = 0
    for i in range(xdim):
      for j in range(ydim):
        if not (board.get_tile(i,j) == 0):
          numberFilled += 1
    if (numberFilled >= ((xdim * ydim - 1) / 3)):
      self.maxWaitTime = 17
      #print('More than 1/3 of all tiles filled: Adjusting to 20 seconds wait time.')
    if (numberFilled >= ((xdim * ydim - 1) / 2)):
      self.maxWaitTime = 12
      #print('More than 1/2 of all tiles filled: Adjusting to 15 seconds wait time.')