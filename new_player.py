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

class NewPlayer(BasePlayer):
  def __init__(self, swap_fun=lambda board, *args: False):
    self.currentPlayer = 2
    self._swap_fun = swap_fun
    self.searchTree = list()
    self.threshhold = 5
    self.simulatedPlayer = 2
    self.turnLimit = 10000

  def choose_tile(self, board, *args) -> tuple:
    """
    Chooses a tile based on given heuristic / algorithm.

    Parameters
    ----------
    board : HexBoard
        Current state of the Hex board.
    args : tuple, optional
        Further parameters in tuple possibly required by heuristic.
    
    Returns
    -------
    res : tuple (int, int)
        Indices of resulting tile.

    """
    (xdim, ydim) = board.dim()
    newMove = BasePlayer.random_choice(board, *args)

    thisBoard = board
    currentBoardState = HexNode(thisBoard, self.currentPlayer)
    for n in range(self.turnLimit):
      self.CarryOutSearch(currentBoardState)
    visitedNodes = list()
    if currentBoardState.children:
      maxUCTScore = -1
      for child in currentBoardState.children:
        #print('Searching the node:')
        #print(child.getBoard().board)
        #print('With UCT score:')
        #print(child.getUCTScore())
        if child.getUCTScore() > maxUCTScore:
          maxUCTScore = child.getUCTScore()
          bestChild = child
      visitedNodes.append(currentBoardState)
      newBoardState = bestChild
      for i in range(xdim):
        for j in range(ydim):
          if (currentBoardState.getBoard().get_tile(i, j) == 0) & (newBoardState.getBoard().get_tile(i, j) != 0):
            newMove = (i, j)
            print('Moved at:')
            print(newMove)
            print('with UCT score:')
            print(maxUCTScore)
    self.searchTree = list()
    return newMove
  
  def CarryOutSearch(self, baseNode):
    thisNode = baseNode
    visitedNodes = list()
    visitedNodes.append(thisNode)
    while thisNode.children:
      maxUCTScore = -1
      for child in thisNode.children:
        if child.getUCTScore() > maxUCTScore:
          maxUCTScore = child.getUCTScore()
          bestChild = child
      thisNode = random.choice(thisNode.children)
      #thisNode = bestChild
      visitedNodes.append(thisNode)
    result = (self.CarryOutSimulation(thisNode) == self.currentPlayer)
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
                        self.searchTree.append(newChild)
                        newChild.setParent(thisNode)
                        node.addChild(newChild)

  def CarryOutSimulation(self, thisNode):
    validMoves = list()
    simulatedBoard = copy.deepcopy(thisNode.getBoard())
    (xdim, ydim) = simulatedBoard.dim()
    for i in range(xdim):
        for j in range(ydim):
            if simulatedBoard.get_tile(i, j) == 0:
              newMove = (i, j)
              validMoves.append(newMove)
    random.shuffle(validMoves)
    if validMoves:
        for move in validMoves:
          (i, j) = move
          simulatedBoard.set_tile(i, j, self.simulatedPlayer)
          self.simulatedPlayer = 3 - self.simulatedPlayer # Switch players
    dummy_player1 = RandomPlayer()
    dummy_player2 = RandomPlayer()
    newGame = HexGame(simulatedBoard.board, dummy_player1, dummy_player2)
    return newGame.check_finish()
  
"""   def ThreatenedBridge(self, board):
    thisBoard = board
    (xdim, ydim) = thisBoard.dim()
    for i in range(xdim):
      for j in range(ydim):
        if thisBoard.get_tile(i,j) == self.currentPlayer: """

  
"""   def BridgeCheck(self, move, board):
     
  
  def FindBridge(self, board):
    # Iterate over all board positions to find a potential bridge
    for i in range(board.dim()[0]):
      for j in range(board.dim()[1]):
        if board.get_tile(i, j) == 0:
          # Place a stone and check if it forms a bridge
          board.set_tile(i, j, self.currentPlayer)
          if self.CheckBridge(board, i, j):
            # If it forms a bridge, return the position
            return i, j
          else:
            # If not, undo the move
            board.set_tile(i, j, 0)
    return None """