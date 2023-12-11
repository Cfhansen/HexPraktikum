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
    while validMoves:
      print('Number of valid moves:')
      print(len(validMoves))
      nextMove = self.analyzeMoves(simulatedBoard)
      if nextMove:
        # Play the move
        (i, j) = nextMove
        simulatedBoard.set_tile(i, j, self.simulatedPlayer)
        self.simulatedPlayer = 3 - self.simulatedPlayer # Switch players
        moveToRemove = (i, j)
        validMoves.remove(moveToRemove)
    dummy_player1 = RandomPlayer()
    dummy_player2 = RandomPlayer()
    newGame = HexGame(simulatedBoard.board, dummy_player1, dummy_player2)
    return newGame.check_finish()
  
  def analyzeMoves(self, board):
    thisBoard = board
    (xdim, ydim) = thisBoard.dim()
    threatenedTiles = list()
    emptyTiles = list()
    for i in range(xdim):
      for j in range(ydim):
        if thisBoard.get_tile(i,j) == self.simulatedPlayer:
          # Case 1: Bridge to the upper right
          if (i+2 < xdim) & (j+1 < ydim):
            if board.get_tile(i+2,j+1) == self.simulatedPlayer:
              if (board.get_tile(i+1,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                threatenedTiles.append(move)
              elif (board.get_tile(i+1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j+1) == 0):
                move = (i+1,j+1)
                threatenedTiles.append(move)
          # Case 2: Bridge to the lower right
          if (i+1 < xdim) & (j-1 > 0):
            if board.get_tile(i+1,j-1) == self.simulatedPlayer:
              if (board.get_tile(i+1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i,j-1) == 0):
                move = (i,j-1)
                threatenedTiles.append(move)
              elif (board.get_tile(i,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                threatenedTiles.append(move)
          # Case 3: Bridge directly below
          if (i-1 > 0) & (j-2 > 0):
            if board.get_tile(i-1,j-2) == self.simulatedPlayer:
              if (board.get_tile(i,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j-1) == 0):
                move = (i-1,j-1)
                threatenedTiles.append(move)
              elif (board.get_tile(i-1,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i,j-1) == 0):
                move = (i,j-1)
                threatenedTiles.append(move)
          # Case 4: Bridge below and to the left
          if (i-2 > 0) & (j-1 > 0):
            if board.get_tile(i-1,j) == self.simulatedPlayer:
              if (board.get_tile(i-1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j-1) == 0):
                move = (i-1,j-1)
                threatenedTiles.append(move)
              elif (board.get_tile(i-1,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                threatenedTiles.append(move)
          # Case 5: Bridge above and to the left
          if (i-1 > 0) & (j+1 < ydim):
            if board.get_tile(i-1,j+1) == self.simulatedPlayer:
              if (board.get_tile(i-1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                threatenedTiles.append(move)
              elif (board.get_tile(i,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                threatenedTiles.append(move)
          # Case 6: Bridge directly above
          if (i+1 < xdim) & (j+2 < ydim):
            if board.get_tile(i+1,j+2) == self.simulatedPlayer:
              if (board.get_tile(i,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j+1) == 0):
                move = (i+1,j+1)
                threatenedTiles.append(move)
              elif (board.get_tile(i+1,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                threatenedTiles.append(move)
        elif thisBoard.get_tile(i,j) == 0:
          move = (i,j)
          emptyTiles.append(move)
    if threatenedTiles:
      return random.choice(threatenedTiles)
    elif emptyTiles:
      return random.choice(emptyTiles)
    else:
      return None