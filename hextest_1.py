import numpy
import pygame
import random

from hexgame import HexBoard
from base_player import BasePlayer
from mc_player import NewPlayer
from new_player_old import NewPlayerOld
from HexNode import HexNode

def TestCarryOutSimulation():
    thisBoard = HexBoard(numpy.zeros((5,5)))
    thisNode = HexNode(thisBoard,2)
    thisPlayer = NewPlayer2(thisBoard)
    print(thisPlayer.CarryOutSimulation(thisNode))

def TestCarryOutSearch():
    newArray = numpy.zeros((5,5))
    newArray[1,1] = 2
    thisBoard = HexBoard(newArray)
    thisNode = HexNode(thisBoard,2)
    thisPlayer = NewPlayer2(thisBoard)
    turnLimit = 10000
    for n in range(turnLimit):
        thisPlayer.CarryOutSearch(thisNode)
    print(len(thisNode.children))
    for node in thisNode.children:
        print(node.getBoard().board)
        print(node.getUCTScore())
        print(node.numberOfWins)
        print(node.numberOfVisits)
        for node2 in node.children:
            print(node2.getBoard().board)
            print(node2.getUCTScore())
            print(node2.numberOfWins)
            print(node2.numberOfVisits)
    maxUCTScore = -1
    for node2 in node.children:
        if node2.getUCTScore() > maxUCTScore:
            print('new max UCT score')
            maxUCTScore = node2.getUCTScore()
            print(node2.getUCTScore())
            print(node2.getBoard().board)
    depthOfSearchTree = 1
    currentNode = thisNode
    while currentNode.children:
        currentNode = random.choice(currentNode.children)
        print(currentNode.numberOfVisits)
        depthOfSearchTree +=1
    print('Depth of search tree:')
    print(depthOfSearchTree)
    print(currentNode.getBoard().board)

def TestUpdateChildren():
    thisBoard = HexBoard(numpy.zeros((5,5)))
    thisNode = SearchTreeNode(thisBoard,2)
    thisPlayer = NewPlayer(thisBoard)
    thisNode.updateChildren()
    print(len(thisNode.children))
    for child in thisNode.children:
        print(child.getBoard())
        child.updateChildren()
        print(len(thisNode.children))
        for childofchild in child.children:
            print(childofchild.getBoard())

def TestSearchTree():
    thisBoard = HexBoard(numpy.zeros((5,5)))
    thisPlayer = NewPlayer(thisBoard)
    thisPlayer.buildSearchTree()

def TestListUpdate():
    thisList = ['a','b','c']
    thisList.remove('b')
    thisList.insert(0,'b')
    print(thisList)

def TestDim():
    thisBoard = HexBoard(numpy.zeros((5,5)))
    (xdim, ydim) = thisBoard.dim()
    for i in range(xdim):
        print(i)
    print(ydim)
    print(thisBoard.get_tile(4,4))

def TestInfill():
    thisBoard = HexBoard(numpy.zeros((5,5)))
    thisBoard.set_tile(2,2,1)
    thisBoard.set_tile(2,4,1)
    thisBoard.set_tile(1,4,1)
    thisBoard.set_tile(1,3,1)
    print(thisBoard)
    swap1 = lambda board, *args: BasePlayer.pygame_manual_swap(board, *args)
    player = NewPlayerOld(swap1)
    thisBoard = player.InfillBoardState(thisBoard)
    print(thisBoard)
    print(thisBoard.get_tile(2,3))
    print(range(thisBoard.dim()[0]))

def TestRandom():
    print(random.uniform(0,1))

        
#TestListUpdate()
#TestCarryOutSearch()
#TestCarryOutSimulation()
#TestUpdateChildren()
#TestDim()
#TestSearchTree()
#TestInfill()
TestRandom()