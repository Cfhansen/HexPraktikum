import pygame

from hexapp import HexApp
from human_player import HumanPlayer
from puzzle_player import PuzzlePlayer
from level_select import LevelSelect


if __name__ == "__main__":
    pygame.init()

    human_player = HumanPlayer()
    puzzle_player = PuzzlePlayer()

    play = True
    level = 14
    while play:
        # Level select
        ls = LevelSelect()
        level = ls.execute()
        if level <= 0 or level >= 500:
            break

        puzzle_board = puzzle_player.get_board(level_number=level, size=10)
        hex_app = HexApp(puzzle_board, human_player, puzzle_player, 1200, 800)
        play = hex_app.execute()

    puzzle_player.shut_down()
    pygame.quit()
