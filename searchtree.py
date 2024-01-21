import anytree
from hexmain import HexBoard

class SearchNode(Node):
  def __init__(self):
    self.parent = None
    self.children = None
    this_board = HexBoard()
    wins = 0
    losses = 0
    score = 0
  def updateScore():
    if losses > 0:
      score = wins / losses
  def traverse():
    for child in children:
      best_score = 0
      if best_score > child.score:
        best_score = child.score
        best_child = child
        child.traverse()
        updateScore()

    
    