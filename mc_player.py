from base_player import BasePlayer

import numpy as np
import time
import sys
import random
import copy

import pygame

from hexgame import HexBoard
from HexNode import HexNode
from hexgame import HexGame
from random_player import RandomPlayer

class MCPlayer(BasePlayer):
  def __init__(self, swap_fun=lambda board, *args: False):
    self.currentPlayer = 1
    self.simulatedPlayer = 1
    self._swap_fun = swap_fun
    self.searchTree = list()
    self.threshhold = 3
    self.turnLimit = 50000
    self.maxWaitTime = 45
    self.totalMoveCount = 0
    self.thisBoard = None
    self.searchNumber = 50

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
    self.currentPlayer = 2
    self.simulatedPlayer = self.currentPlayer
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
    return res

  def choose_tile(self, board, *args) -> tuple:
    """
    Chooses a tile based on Monte Carlo tree search.

    In the case that a one-move win is possible, choose this move.

    Otherwise, we search the tree until the time limit ends. Each time that a leaf node is reached, we carry out Monte Carlo simulations and adjust the UCT score
    of all visited nodes.

    The final choice is the child node with the best UCT score, that is, the largest proportion of wins in the Monte Carlo simulations.

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

    thisBoard = board
    simulatedBoard = self.InfillBoardState(thisBoard)
    currentBoardState = HexNode(thisBoard, self.currentPlayer)
    """
      First: Produce a list of all possible moves. Check whether a win is possible.
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
      If no win is possible: Carry out the Monte Carlo tree search until time runs out.
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
          start = time.time()
    visitedNodes = list()
    """
      Examine all children of the current node and choose the one with the best UCT score. This is the next move.
     """  
    if currentBoardState.children:
      maxUCTScore = -1
      for child in currentBoardState.children:
        if child.getUCTScore() > maxUCTScore:
          maxUCTScore = child.getUCTScore()
          bestChild = child
      visitedNodes.append(currentBoardState)
      newBoardState = bestChild
      for i in range(xdim):
        for j in range(ydim):
          if (currentBoardState.getBoard().get_tile(i, j) == 0) & (newBoardState.getBoard().get_tile(i, j) != 0):
            newMove = (i, j)
            print('Current player: {}. Zug: {} mit UCT-Wert: {}'.format(self.currentPlayer, newMove, maxUCTScore))
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

    """
    Choose the best child of the current node. In general, we choose the child with the highest UCT score, but sometimes choose randomly to allow for exploration.
    """
    while thisNode.children:
      if (thisNode.getCurrentPlayer() == self.currentPlayer):
        maxUCTScore = -1
        for child in thisNode.children:
          if child.getUCTScore() > maxUCTScore:
            maxUCTScore = child.getUCTScore()
            bestChild = child
        if (random.uniform(0,1) < 0.05):
          thisNode = random.choice(thisNode.children)
        else:
          thisNode = bestChild
      elif (thisNode.getCurrentPlayer() == 3 - self.currentPlayer):
        minUCTScore = 2
        for child in thisNode.children:
          if child.getUCTScore() < minUCTScore:
            minUCTScore = child.getUCTScore()
            bestChild = child
        if (random.uniform(0,1) < 0.05):
          thisNode = random.choice(thisNode.children)
        else:
          thisNode = bestChild
      visitedNodes.append(thisNode)
    """
    Upon reaching a leaf, we carry out a simulation and update the UCT scores of all traversed nodes.
    """
    winner = self.CarryOutSimulation(thisNode)
    result = (winner == self.currentPlayer)
    for node in visitedNodes:
        node.updateUCT(result)
        nodeBoard = node.getBoard()
        if node.numberOfVisits == self.threshhold:
            (xdim, ydim) = nodeBoard.dim()
            for i in range(xdim):
                for j in range(ydim):
                    if nodeBoard.get_tile(i, j) == 0:
                        newBoard = copy.deepcopy(nodeBoard)
                        newBoard.set_tile(i, j, node.getCurrentPlayer())
                        newChild = HexNode(newBoard, 3 - node.getCurrentPlayer())
                        if not (newChild.getCurrentPlayer() == self.currentPlayer):
                          newChild.setUCTScore(0)
                        self.searchTree.append(newChild)
                        newChild.setParent(thisNode)
                        node.addChild(newChild)

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
    """
    while validMoves:
      (moves, threatenedBridgePresent) = self.analyzeMoves(simulatedBoard)
      nextMove = random.choice(moves)
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
    return newGame.check_finish()
  
  def analyzeMoves(self, board):
    """
    Carries out a single Monte Carlo simulation.

    Parameters
    ----------
    board: The current state of the board.
    
    Returns
    -------
    The coordinates of the chosen move.
    """

    """
    Use the savebridge heuristic as described in the MoHex paper. If a move of the opposing player threatens a bridge, play into the open space to form a connection.
    """
    (xdim, ydim) = board.dim()
    threatenedTiles = list()
    emptyTiles = list()
    for i in range(xdim):
      for j in range(ydim):
        if (board.get_tile(i,j) == self.simulatedPlayer):
          # Fall 1: Brücke rechts oben
          if (i-1 > 0) & (j+2 < ydim):
            if (board.get_tile(i-1,j+2) == self.simulatedPlayer):
              if (board.get_tile(i-1,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                threatenedTiles.append(move)
              if (board.get_tile(i,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                threatenedTiles.append(move)
          # Fall 2: Brücke rechts unten
          if (i+1 < xdim) & (j+1 < ydim):
            if (board.get_tile(i+1,j+1) == self.simulatedPlayer):
              if (board.get_tile(i,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                threatenedTiles.append(move)
              if (board.get_tile(i+1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                threatenedTiles.append(move)
          # Fall 3: Brücke lotrecht oben
          if (i-1 > 0) & (j+1 < ydim):
            if (board.get_tile(i-2,j+1) == self.simulatedPlayer):
              if (board.get_tile(i-1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                threatenedTiles.append(move)
              if (board.get_tile(i-1,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                threatenedTiles.append(move)
            # Fall 4: Am unteren Rand
          if (i == xdim - 2) & (j > 0) & (self.currentPlayer == 1):
            if (board.get_tile(i+1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                threatenedTiles.append(move)
            if (board.get_tile(i+1,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                threatenedTiles.append(move)
            # Fall 5: Am oberen Rand
          if (i == 1) & (j < ydim - 1) & (self.currentPlayer == 1):
            if (board.get_tile(i-1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                threatenedTiles.append(move)
            if (board.get_tile(i+1,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                threatenedTiles.append(move)
            # Fall 6: Am linken Rand
          if (i < xdim - 1) & (j == 1) & (self.currentPlayer == 2):
            if (board.get_tile(i+1,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i,j-1) == 0):
                move = (i,j-1)
                threatenedTiles.append(move)
            if (board.get_tile(i,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                threatenedTiles.append(move)
            # Fall 7: Am rechten Rand
          if (i > 0) & (j == ydim - 2) & (self.currentPlayer == 2):
            if (board.get_tile(i-1,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                threatenedTiles.append(move)
            if (board.get_tile(i,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                threatenedTiles.append(move)
        elif board.get_tile(i,j) == 0:
          move = (i,j)
          emptyTiles.append(move)
    """
    If at least one threatened bridge was found, choose randomly from the moves which save one of these bridges.

    Otherwise, choose randomly from among all other valid moves.
    """
    if threatenedTiles:
      return (threatenedTiles, True)
    elif emptyTiles:
      return (emptyTiles, False)
    else:
      return None
    
  def InfillBoardState(self, board):
    print('Calculating infill...')
    thisBoard = copy.deepcopy(board)
    (xdim, ydim) = thisBoard.dim()
    for i in range(xdim):
      for j in range(ydim):
        if thisBoard.get_tile(i,j) == 0:
          if (i > 0) & (j > 0) & (i + 1 < xdim) & (j + 1 < ydim):
            # Muster 1.1: 'Kappe' des ersten Spielers
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i,j+1) == 1):
              print('Match found! (test case 1) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 1.2: 'Kappe' des ersten Spielers
            if (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i+1,j) == 1):
              print('Match found! (test case 2) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 1.3: 'Kappe' des ersten Spielers
            if (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i+1,j-1) == 1):
              print('Match found! (test case 3) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 1.4: 'Kappe' des ersten Spielers 1
            if (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i,j-1) == 1):
              print('Match found! (test case 4) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 1.5: 'Kappe' des ersten Spielers 1
            if (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1):
              print('Match found! (test case 5) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 1.6: 'Kappe' des ersten Spielers 1
            if (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i-1,j+1) == 1):
              print('Match found! (test case 6) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 2.1: 'Kappe' des zweiten Spielers
            if (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i,j+1) == 2):
              print('Match found! (test case 7) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 2.2: 'Kappe' des zweiten Spielers
            if (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i+1,j) == 2):
              print('Match found! (test case 8 ) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 2.3: 'Kappe' des zweiten Spielers
            if (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j-1) == 2):
              print('Match found! (test case 9) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 2.4: 'Kappe' des zweiten Spielers
            if (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i,j-1) == 2):
              print('Match found! (test case 10) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 2.5: 'Kappe' des zweiten Spielers
            if (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i-1,j) == 2):
              print('Match found! (test case 11) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 2.6: 'Kappe' des zweiten Spielers
            if (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i-1,j+1) == 2):
              print('Match found! (test case 12) ({},{})'.format(i,j))
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 3.1: 'Pfeil' des ersten Spielers 1
            if (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i-1,j) == 2):
              print('Match found! (test case 13)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 3.2: 'Pfeil' des ersten Spielers 2
            if (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i-1,j+1) == 2):
              print('Match found! (test case 14)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 3.3: 'Pfeil' des ersten Spielers 3
            if (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i,j+1) == 2):
              print('Match found! (test case 15)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 3.4: 'Pfeil' des ersten Spielers 4
            if (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i+1,j) == 2):
              print('Match found! (test case 16)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 3.5: 'Pfeil' des ersten Spielers 5
            if (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i+1,j-1) == 2):
              print('Match found! (test case 17)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 3.6: 'Pfeil' des zweiten Spielers 6
            if (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i,j-1) == 2):
              print('Match found! (test case 18)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 4.1: 'Pfeil' des zweiten Spielers 1
            if (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i-1,j) == 1):
              print('Match found! (test case 19)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 4.2: 'Pfeil' des zweiten Spielers 2
            if (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i-1,j+1) == 1):
              print('Match found! (test case 20)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 4.3: 'Pfeil' des zweiten Spielers 3
            if (thisBoard.get_tile(i+1,j-1) == 2) & (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i,j+1) == 1):
              print('Match found! (test case 21)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 4.4: 'Pfeil' des zweiten Spielers 4
            if (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i+1,j) == 1):
              print('Match found! (test case 22)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 4.5: 'Pfeil' des zweiten Spielers 5
            if (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i+1,j-1) == 1):
              print('Match found! (test case 23)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 4.6: 'Pfeil' des zweiten Spielers 6
            if (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j+1) == 2) & (thisBoard.get_tile(i,j-1) == 1):
              print('Match found! (test case 24)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
            # Muster 5.1: 'Fliege' 1
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j+1) == 2):
              print('Match found! (test case 25)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 5.2: 'Fliege' 2
            if (thisBoard.get_tile(i-1,j+1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j-1) == 2):
              print('Match found! (test case 26)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 5.3: 'Fliege' 3
            if (thisBoard.get_tile(i-1,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j+1) == 2):
              print('Match found! (test case 27)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 5.4: 'Fliege' 4
            if (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i,j+1) == 1) & (thisBoard.get_tile(i,j-1) == 2) & (thisBoard.get_tile(i-1,j) == 2):
              print('Match found! (test case 28)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 5.5: 'Fliege' 5
            if (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i-1,j) == 2):
              print('Match found! (test case 29)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 5.6: 'Fliege' 6
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i+1,j-1) == 1) & (thisBoard.get_tile(i-1,j+1) == 2) & (thisBoard.get_tile(i,j+1) == 2):
              print('Match found! (test case 30)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 6: am Rand, Steine links und oben links  
          if (i == xdim-1) & (j > 0):
            if (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i,j-1) == 1):
              print('Match found! (test case 31)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 7: am Rand, Steine Links und rechts
          if (i == xdim-1) & (j > 0) & (j+1 < ydim):
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i,j+1) == 1):
              print('Match found! (test case 32)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 8: am Rand, Steine oben links and rechts
          if (i == xdim-1) & (j < ydim-1):
            if (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i,j+1) == 1):
              print('Match found! (test case 33)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
            # Muster 9: am Rand, Steine oben links und oben rechts
          if (i == xdim-1) & (j < ydim-1):
            if (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i-1,j+1) == 2):
              print('Match found! (test case 34)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
            # Muster 10: im akuten Winkel
          if (i == xdim-1) & (j == ydim-1):
            if (thisBoard.get_tile(i,j-1) == 1):
              print('Match found! (test case 35)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # am oberen Rand
            # Muster 11: am Rand, Steine rechts und unten rechts 
          if (i == 0) & (j+1 < ydim):
            if (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i,j+1) == 1):
              print('Match found! (test case 36)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 12: am Rand, Steine Links und rechts
          if (i == 0) & (j > 0) & (j+1 < ydim):
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i,j+1) == 1):
              print('Match found! (test case 37)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 13: am Rand, Steine unten rechts und links
          if (i == 0) & (j > 0):
            if (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i,j-1) == 1):
              print('Match found! (test case 38)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
            # Muster 14: am oberen Rand, Steine unten links und unten rechts
          if (i == 0) & (j > 0):
            if (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j-1) == 2):
              print('Match found! (test case 39)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
            # Muster 15: im akuten Winkel
          if (i == 0) & (j == 0):
            if (thisBoard.get_tile(i,j+1) == 1):
              print('Match found! (test case 40)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # am linken Rand
            # Muster 16: am Rand, Steine links und oben links  
          if (i == xdim-1) & (j > 0):
            if (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i,j-1) == 1):
              print('Match found! (test case 41)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 17: am Rand, Steine Links und rechts
          if (i == xdim-1) & (j > 0) & (j+1 < ydim):
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i,j+1) == 1):
              print('Match found! (test case 42)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 18: am Rand, Steine oben links and rechts
          if (i == xdim-1) & (j < ydim-1):
            if (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i,j+1) == 1):
              print('Match found! (test case 43)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
            # Muster 19: am Rand, Steine oben links und oben rechts
          if (i == xdim-1) & (j < ydim-1):
            if (thisBoard.get_tile(i-1,j) == 2) & (thisBoard.get_tile(i-1,j+1) == 2):
              print('Match found! (test case 44)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
            # Muster 20: im akuten Winkel
          if (i == xdim-1) & (j == ydim-1):
            if (thisBoard.get_tile(i,j-1) == 1):
              print('Match found! (test case 45)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
        move = (0, 0)
        if thisBoard.get_tile(i,j) == self.currentPlayer:
          # Fall 1: Brücke rechts oben
          if (i-1 > 0) & (j+2 < ydim):
            if (board.get_tile(i-1,j+2) == self.currentPlayer):
              if (board.get_tile(i-1,j+1) == 3 - self.currentPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                print('Threatened bridge found: {}'.format(move))
              if (board.get_tile(i,j+1) == 3 - self.currentPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                print('Threatened bridge found: {}'.format(move))
          # Fall 2: Brücke rechts unten
          if (i+1 < xdim) & (j+1 < ydim):
            if (board.get_tile(i+1,j+1) == self.currentPlayer):
              if (board.get_tile(i,j+1) == 3 - self.currentPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                print('Threatened bridge found: {}'.format(move))
              if (board.get_tile(i+1,j) == 3 - self.currentPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                print('Threatened bridge found: {}'.format(move))
          # Fall 3: Brücke lotrecht oben
          if (i-1 > 0) & (j+1 < ydim):
            if (board.get_tile(i-2,j+1) == self.currentPlayer):
              if (board.get_tile(i-1,j) == 3 - self.currentPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                print('Threatened bridge found: {}'.format(move))
              if (board.get_tile(i-1,j+1) == 3 - self.currentPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                print('Threatened bridge found: {}'.format(move))
            # Fall 4: Am unteren Rand
          if (i == xdim - 2) & (j > 0) & (self.currentPlayer == 1):
            if (board.get_tile(i+1,j) == 3 - self.currentPlayer) & (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                print('Threatened bridge found: {}'.format(move))
            if (board.get_tile(i+1,j-1) == 3 - self.currentPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                print('Threatened bridge found: {}'.format(move))
            # Fall 5: Am oberen Rand
          if (i == 1) & (j < ydim - 1) & (self.currentPlayer == 1):
            if (board.get_tile(i-1,j) == 3 - self.currentPlayer) & (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                print('Threatened bridge found: {}'.format(move))
            if (board.get_tile(i+1,j-1) == 3 - self.currentPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                print('Threatened bridge found: {}'.format(move))
            # Fall 6: Am linken Rand
          if (i < xdim - 1) & (j == 1) & (self.currentPlayer == 2):
            if (board.get_tile(i+1,j-1) == 3 - self.currentPlayer) & (board.get_tile(i,j-1) == 0):
                move = (i,j-1)
                print('Threatened bridge found: {}'.format(move))
            if (board.get_tile(i,j-1) == 3 - self.currentPlayer) & (board.get_tile(i+1,j-1) == 0):
                move = (i+1,j-1)
                print('Threatened bridge found: {}'.format(move))
            # Fall 7: Am rechten Rand
          if (i > 0) & (j == ydim - 2) & (self.currentPlayer == 2):
            if (board.get_tile(i-1,j+1) == 3 - self.currentPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                print('Threatened bridge found: {}'.format(move))
            if (board.get_tile(i,j+1) == 3 - self.currentPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                print('Threatened bridge found: {}'.format(move))
        elif thisBoard.get_tile(i,j) == 3 - self.currentPlayer:
          # Fall 1: Brücke rechts oben
          if (i-1 > 0) & (j+2 < ydim):
            if (board.get_tile(i-1,j+2) == 3 - self.currentPlayer):
              if (board.get_tile(i-1,j+1) == self.currentPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                print('Threatened bridge found: {}'.format(move))
              if (board.get_tile(i,j+1) == self.currentPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                print('Threatened bridge found: {}'.format(move))
          # Fall 2: Brücke rechts unten
          if (i+1 < xdim) & (j+1 < ydim):
            if (board.get_tile(i+1,j+1) == 3 - self.currentPlayer):
              if (board.get_tile(i,j+1) == self.currentPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                print('Threatened bridge found: {}'.format(move))
              if (board.get_tile(i+1,j) == self.currentPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                print('Threatened bridge found: {}'.format(move))
          # Fall 3: Brücke lotrecht oben
          if (i-1 > 0) & (j+1 < ydim):
            if (board.get_tile(i-2,j+1) == 3 - self.currentPlayer):
              if (board.get_tile(i-1,j) == self.currentPlayer) & (board.get_tile(i-1,j+1) == 0):
                move = (i-1,j+1)
                print('Threatened bridge found: {}'.format(move))
              if (board.get_tile(i-1,j+1) == self.currentPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                print('Threatened bridge found: {}'.format(move))    
    return thisBoard  