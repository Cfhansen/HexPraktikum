import numpy as np
import time
import random
import copy

import pygame
from hexgame import HexBoard
from hexgame import HexGame
from hexapp import HexApp
from random_player import RandomPlayer

class HexMove:
    def __init__(self, i, j):
        self.UCTScore = 0
        self.numberOfVisits = 1
        self.numberOfWins = 0
        self.AMAFScore = 0
        self.AMAFNumberOfWins = 0
        self.AMAFNumberOfVisits = 1
        self.icoord = i
        self.jcoord = j
    
    def getUCTScore(self):
        return self.UCTScore
    
    def updateUCT(self, result):
        if result:
            self.numberOfWins += 1
        self.numberOfVisits += 1
        self.UCTScore = self.numberOfWins / self.numberOfVisits

    def getAMAFScore(self):
        return self.AMAFScore
    
    def updateAMAF(self, result):
        if result:
            self.AMAFNumberOfWins += 1
        self.AMAFNumberOfVisits += 1
        self.UCTScore = self.AMAFNumberOfWins / self.AMAFNumberOfVisits
    
    def getRAVEScore(self):
        if ((self.numberOfVisits + self.AMAFNumberOfVisits) > 0):
            return (self.numberOfWins + self.AMAFNumberOfWins) / (self.numberOfVisits + self.AMAFNumberOfVisits)
        else:
            return 0