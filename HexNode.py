""" import numpy as np
import time
import random
import copy

import pygame
from hexgame import HexBoard
from hexgame import HexGame
from hexapp import HexApp
from random_player import RandomPlayer

class HexNode:
    def __init__(self, board, current_player):
        self.parent = None
        self.children = list()
        self.UCTScore = 0
        self.thisBoard = board
        self.numberOfVisits = 1
        self.numberOfWins = 0
        self.currentPlayer = current_player
        self.AMAFScore = 0
        self.AMAFNumberOfWins = 0
        self.AMAFNumberOfVisits = 1

    def setParent(self, node):
        self.parent = node

    def getParent(self):
        return self.parent

    def addChild(self, node):
        self.children.append(node)

    def getChildren(self):
        return self.children

    def getBoard(self):
        return self.thisBoard
    
    def getUCTScore(self):
        return self.UCTScore
    
    def getCurrentPlayer(self):
        return self.currentPlayer
    
    def updateUCT(self, result):
        if (result == self.currentPlayer):
            self.numberOfWins += 1
        self.numberOfVisits += 1
        self.UCTScore = self.numberOfWins / self.numberOfVisits

    def numberOfChildren(self):
        return len(self.children)
    
    def updateAMAF(self, result):
        if (result == self.currentPlayer):
            self.AMAFNumberOfWins += 1
        self.AMAFNumberOfVisits += 1
        self.UCTScore = self.AMAFNumberOfWins / self.AMAFNumberOfVisits

    def getAMAFScore(self):
        return self.AMAFScore
    
    def getRAVEScore(self):
        return (self.numberOfWins + self.AMAFNumberOfWins) / (self.numberOfVisits + self.AMAFNumberOfVisits) """

import numpy as np
import time
import random
import copy

import pygame
from hexgame import HexBoard
from hexgame import HexGame
from hexapp import HexApp
from random_player import RandomPlayer

class HexNode:
    def __init__(self, board, current_player):
        self.parent = None
        self.children = list()
        self.UCTScore = 1
        self.thisBoard = board
        self.numberOfVisits = 0
        self.numberOfWins = 0
        self.currentPlayer = current_player

    def setParent(self, node):
        self.parent = node

    def getParent(self):
        return self.parent

    def addChild(self, node):
        self.children.append(node)

    def getBoard(self):
        return self.thisBoard
    
    def getUCTScore(self):
        return self.UCTScore
    
    def getUCTVisits(self):
        return self.numberOfWins
    
    def getCurrentPlayer(self):
        return self.currentPlayer
    
    def updateUCT(self, result):
        if result:
            self.numberOfWins += 1
        self.numberOfVisits += 1
        self.UCTScore = self.numberOfWins / self.numberOfVisits

    def numberOfChildren(self):
        return len(self.children)
    
    def setUCTScore(self, val):
        self.UCTScore = val