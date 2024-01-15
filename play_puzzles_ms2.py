import pygame

from hexapp import HexApp
from human_player import HumanPlayer
from puzzle_player import PuzzlePlayer
from level_select import LevelSelect
from random_player import RandomPlayer
from mc_player import MCPlayer

if __name__ == "__main__":
    pygame.init()

    human_player = MCPlayer()
    puzzle_player = PuzzlePlayer()

    #levels = [1, 2, 6, 11, 20, 23, 24, 26, 29, 37, 41, 51, 54, 58, 60]
    levels = [1, 2, 6, 11, 20, 23, 24, 26, 29, 37]
    win = len(levels) * [False]

    # You can also pick certain levels from the list!
    for (i, lvl) in enumerate(levels[0:]):
        # Play all levels and count number of wins and losses
        puzzle_board = puzzle_player.get_board(level_number=lvl, size=12)
        hex_app = HexApp(puzzle_board, human_player, puzzle_player, 1200, 800)
        play = hex_app.execute()
        print(hex_app.hex.get_winner())
        if hex_app.hex.get_winner() == hex_app.hex.players[0].get_id():
            win[i] = True

    # Output of results:
    for i in range(len(levels)):
        print("Level {:02d}: {:s}".format(levels[i],
                                          "Win" if win[i] == True else "Lose"))
    print("\nNumber of Wins: ", sum(win), " / ", len(levels))

    puzzle_player.shut_down()
    pygame.quit()
