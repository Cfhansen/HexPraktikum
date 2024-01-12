# Teilnehmer: Hansen/Reschke
# Testen des Codes:
# from new_player import NewPlayer
# player1 = NewPlayer()
# (or:) player2 = NewPlayer()

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

class NewPlayerOld(BasePlayer):
  def __init__(self, swap_fun=lambda board, *args: False):
    self.currentPlayer = 1
    self.simulatedPlayer = 1
    self._swap_fun = swap_fun
    self.searchTree = list()
    self.threshhold = 10
    self.turnLimit = 50000
    self.maxWaitTime = 5
    self.totalMoveCount = 0
    self.thisBoard = None
    self.searchNumber = 500

  def claim_swap(self, board, *args) -> bool: #Ignoriere die vorgegebene Swapfunktion; nutze stattdessen die hier gegebene kustomisierte Funktion
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
          print('Wird geswappt: Gegner hat {} gespielt'.format((i, j)))
    return res

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
    thisBoard = self.InfillBoardState(thisBoard)
    currentBoardState = HexNode(thisBoard, self.currentPlayer)
    elapsed = 0
    start = None
    for n in range(self.turnLimit):
      if elapsed < self.maxWaitTime:
        self.CarryOutSearch(currentBoardState)
        if n%self.searchNumber == 0: # Prüfe die vergangene Zeit jedes Mal nur nach einer bestimmten vom Parameter angegebenen Anzahl an Spielen
          if start:
            end = time.time()
            elapsed = elapsed + end - start
            print('Suchzeit für diesen Zug: {} mit bislang {} simulierten Spielen.'.format(elapsed, n))
          start = time.time()
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
            print('Zug: {} mit UCT-Wert: {}'.format(newMove, maxUCTScore))
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
                 
  def InfillBoardState(self, board):
    print('Calculating infill...')
    thisBoard = copy.deepcopy(board)
    (xdim, ydim) = thisBoard.dim()
    for i in range(xdim):
      for j in range(ydim):
        if thisBoard.get_tile(i,j) == 0:
          if (i > 0) & (j > 0) & (i + 1 < xdim) & (j + 1 < ydim):
            # Muster 1: 'Kappe' des ersten Spielers
            if (thisBoard.get_tile(i-1,j-1) == 1) & (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i+1,j) == 1) & (thisBoard.get_tile(i+1,j+1) == 1):
              print('Match found! (cap)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 2: 'Kappe' des zweiten Spielers
            if thisBoard.get_tile(i-1,j-1) == 2 & thisBoard.get_tile(i,j-1) == 2 & thisBoard.get_tile(i+1,j) == 2 & thisBoard.get_tile(i+1,j+1) == 2:
              print('Match found! (cap)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 3: 'Pfeil' des ersten Spielers
            if thisBoard.get_tile(i-1,j-1) == 1 & thisBoard.get_tile(i-1,j) == 1 & thisBoard.get_tile(i,j+1) == 1 & thisBoard.get_tile(i+1,j) == 2:
              print('Match found! (arrow)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 4: 'Pfeil' des zweiten Spielers
            if thisBoard.get_tile(i,j-1) == 2 & thisBoard.get_tile(i+1,j) == 2 & thisBoard.get_tile(i+1,j+1) == 2 & thisBoard.get_tile(i-1,j) == 1:
              print('Match found! (arrow)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)     
            # Muster 5: 'tote Brücke'
            if (thisBoard.get_tile(i-1,j-1) == 1) & (thisBoard.get_tile(i-1,j) == 1) & (thisBoard.get_tile(i+1,j) == 2) & (thisBoard.get_tile(i+1,j+1) == 2):
              print('Match found! (dead bridge)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 6: am Rand, Steine links und oben links  
          if (i == xdim-1) & (j > 0):
            if (thisBoard.get_tile(i-1,j-1) == 1) & (thisBoard.get_tile(i,j-1) == 1):
              print('Match found! (self block on bottom edge)')
              thisBoard.set_tile(i, j, self.simulatedPlayer) 
            # Muster 7: am Rand, Steine Links und rechts
          if (i == xdim) & (j > 0) & (i+1 < xdim) & (j+1 < ydim):
            if (thisBoard.get_tile(i-1,j-1) == 1) & (thisBoard.get_tile(i+1,j+1) == 1):
              print('Match found! (self sandwich on bottom edge)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
            # Muster 8: am Rand, Steine oben links and rechts
          if (j == ydim) & (i+1 < xdim) & (j+1 < ydim):
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i+1,j+1) == 2):
              print('Match found! (double block on bottom edge)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
            # Muster 9: am Rand, Steine oben links und oben rechts
          if (j == ydim) & (i+1 < xdim):
            if (thisBoard.get_tile(i,j-1) == 1) & (thisBoard.get_tile(i+1,j) == 1):
              print('Match found! (blocked by opponent on bottom edge)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)  
            # Muster 10: im Winkel
          if (j == ydim) & (i == xdim):
            if (thisBoard.get_tile(i-1,j-1) == 2):
              print('Match found! (corner block)')
              thisBoard.set_tile(i, j, self.simulatedPlayer)
    return thisBoard      

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
      nextMove = self.analyzeMoves(simulatedBoard)
      if nextMove:
        # Zug wird gespielt
        (i, j) = nextMove
        simulatedBoard.set_tile(i, j, self.simulatedPlayer)
        self.simulatedPlayer = 3 - self.simulatedPlayer # Wechsle auf den anderen Spieler
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
          # Fall 1: Brücke rechts oben
          if (i+2 < xdim) & (j+1 < ydim):
            if board.get_tile(i+2,j+1) == self.simulatedPlayer:
              if (board.get_tile(i+1,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                threatenedTiles.append(move)
              elif (board.get_tile(i+1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j+1) == 0):
                move = (i+1,j+1)
                threatenedTiles.append(move)
          # Fall 2: Brücke rechts unten
          if (i+1 < xdim) & (j-1 > 0):
            if board.get_tile(i+1,j-1) == self.simulatedPlayer:
              if (board.get_tile(i+1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i,j-1) == 0):
                move = (i,j-1)
                threatenedTiles.append(move)
              elif (board.get_tile(i,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i+1,j) == 0):
                move = (i+1,j)
                threatenedTiles.append(move)
          # Fall 3: Brücke lotrecht unten
          if (i-1 > 0) & (j-2 > 0):
            if board.get_tile(i-1,j-2) == self.simulatedPlayer:
              if (board.get_tile(i,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j-1) == 0):
                move = (i-1,j-1)
                threatenedTiles.append(move)
              elif (board.get_tile(i-1,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i,j-1) == 0):
                move = (i,j-1)
                threatenedTiles.append(move)
          # Fall 4: Brücke links unten
          if (i-2 > 0) & (j-1 > 0):
            if board.get_tile(i-1,j) == self.simulatedPlayer:
              if (board.get_tile(i-1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j-1) == 0):
                move = (i-1,j-1)
                threatenedTiles.append(move)
              elif (board.get_tile(i-1,j-1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                threatenedTiles.append(move)
          # Fall 5: Brücke links oben
          if (i-1 > 0) & (j+1 < ydim):
            if board.get_tile(i-1,j+1) == self.simulatedPlayer:
              if (board.get_tile(i-1,j) == 3 - self.simulatedPlayer) & (board.get_tile(i,j+1) == 0):
                move = (i,j+1)
                threatenedTiles.append(move)
              elif (board.get_tile(i,j+1) == 3 - self.simulatedPlayer) & (board.get_tile(i-1,j) == 0):
                move = (i-1,j)
                threatenedTiles.append(move)
          # Fall 6: Brücke lotrecht oben
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