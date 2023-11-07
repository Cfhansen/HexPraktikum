from base_player import BasePlayer

import numpy as np

import pygame

import multiprocessing
import time

class NewPlayer(BasePlayer):
  def choose_tile(self, board, *args) -> tuple:
    # Start process
    p = multiprocessing.Process(target=choose_tile_child, name="Choose tile", args=(10,))
    p.start()

    # Wait 10 seconds
    time.sleep(10)
    # Terminate
    p.terminate()
    # Cleanup
    p.join()

  def choose_tile_child(self, board, *args) -> tuple:
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
    global row

    i = row
    iter = 1
    while (np.all(board.get_row(i) > 0) | np.any(board.get_row(i) == 1)) & (iter < 100):
      iter = iter + 1
      i = np.random.randint(ydim)
      row = i

    # Then choose random column index to obtain new tile
    j = np.random.randint(xdim)
    while board.get_tile(i,j) > 0:
      j = np.random.randint(xdim)
    res = (i,j)

    time.sleep(0.5) # Simulate thinking

    return BasePlayer.random_choice(board, *args)
