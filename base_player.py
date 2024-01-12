import numpy as np
import time

import pygame


class BasePlayer:
    def __init__(self, swap_fun=lambda board, *args: False):
        """
        Initializes base player instance with user-defined "swap player" algorithm.

        Parameters
        ----------
        swap_fun : callable
            The heuristic / algorithm used to determine if swap is to be applied

              ``swap_fun(board, *args) -> bool``

            where ``board`` is the current state of the Hex board, and
            ``args`` is a tuple of further parameters possibly required by
            the underlying algorithm / heuristic (optional).

        """
        self._swap_fun = swap_fun
        self.id = 0  # id can be interpreted as color of player

    def set_id(self, new_id):
        self.id = new_id

    def get_id(self):
        return self.id

    def choose_tile(self, board, *args) -> tuple:
        """
        Chooses a tile based on given heuristic / algorithm.

        Parameters
        ----------
        board : HexBoard
            Current state of the Hex board.
        args : tuple, optional
            Further parameters in tuple possibly required by heuristic.

        Raises Exception.

        """
        raise NotImplementedError("Please derive a class and implement this method.")

    def claim_swap(self, board, *args) -> bool:
        """
        Determines whether to claim a swap on given heuristic / algorithm.

        Parameters
        ----------
        board : HexBoard
            Current state of the Hex board.
        args : tuple, optional
            Further parameters in tuple possibly required by heuristic.

        Returns
        -------
        res : bool
            Whether to perform swap (True) or not (False)

        """

        res = self._swap_fun(board, *args)
        return res

    @classmethod
    def random_choice(cls, board, *args) -> tuple:
        """
        Implements random tile picking "heuristic".

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

        # First choose random row index (must not be already occupied!)
        i = np.random.randint(ydim)
        while np.all(board.get_row(i) > 0):
            i = np.random.randint(ydim)

        # Then choose random column index to obtain new tile
        j = np.random.randint(xdim)
        while board.get_tile(i, j) > 0:
            j = np.random.randint(xdim)
        res = (i, j)

        time.sleep(0.5)  # Simulate thinking

        return res

    @classmethod
    def random_swap(cls, board, *args) -> tuple:
        """
        Implements random choice of claiming a swap

        Parameters
        ----------
        board : HexBoard
            Current state of the Hex board.
        args : tuple, optional
            Further parameters in tuple possibly required by heuristic.

        Returns
        -------
        res : bool
            Claims swap (True) or not (False)

        """

        res = (np.random.rand(1) >= 0.5)
        if res:
            print("Swap rule applied")

        time.sleep(0.5)  # Simulate thinking

        return res

    @classmethod
    def pygame_manual_swap(cls, board, *args):
        """
        Implements user choice via keyboard input of claiming a swap

        Parameters
        ----------
        board : HexBoard
            Current state of the Hex board.
        args : tuple, optional
            Further parameters in tuple possibly required by heuristic.

        Returns
        -------
        res : bool
            Claims swap (True) or not (False)

        """

        # This blocks the whole event loop until a key is pressed:
        # NOTE: The same strategy could of course also be applied to the tile selection!
        print("Shall swap rule be applied? 1: Yes; otherwise: No")
        while True:
            event = pygame.event.poll()
            if event.type == pygame.KEYDOWN:
                res = (event.key == pygame.K_1 or event.key == pygame.K_KP1)
                if res:
                    print("Swap rule applied")
                return res
