from base_player import BasePlayer
from random_player import RandomPlayer
from human_player import HumanPlayer
from new_player import NewPlayer
from hexapp import HexApp
from hexglobals import PLAYER

import numpy as np

if __name__ == "__main__":
  # First create players with respective strategies
  swap1 = lambda board, *args : BasePlayer.pygame_manual_swap(board, *args)
  swap2 = lambda board, *args : BasePlayer.random_swap(board, *args)

  player1 = RandomPlayer(swap1)
  player2 = NewPlayer(swap2)

  # Set up empty board
  dim = 11
  board = np.zeros((dim, dim), dtype=int)

  # Manual changes
  # board[0,0] = PLAYER["red"]
  # board[2,2] = PLAYER["red"]

  play = True
  while play != False:
    # Game will be completely restarted if play == True
    hex_app = HexApp(board, player1, player2, 1200, 800)
    play = hex_app.execute()