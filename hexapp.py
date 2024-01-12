import numpy as np
from math import sin, cos, pi
import time

import pygame

from hexgame import HexGame, PuzzleHexGame
from hexglobals import PLAYER_COLORS, BACKGROUND_COLOR, BORDER_COLOR, EMPTY_TILE_COLOR
from puzzle_player import PuzzlePlayer


class HexApp:
    def __init__(self, board, player1, player2, px=800, py=800):
        self._running = None
        self.mouse_x = None
        self.mouse_y = None

        # Display related variables
        self._display_surf = None
        self.disp_size = (px, py)

        self.y_dim, self.x_dim = board.shape

        # State of the game (+ events) related variables
        self._freeze = False
        self._finish = False
        self._mouse_click = False
        self._key_press = False
        self._key = None
        self._replay = False

        use_puzzle_game = isinstance(player2, PuzzlePlayer)
        if use_puzzle_game:
            self.hex = PuzzleHexGame(board, player1, player2)
        else:
            self.hex = HexGame(board, player1, player2)

        # Ensures that no swap is applied if more than 1 tile is occupied
        self.hex.set_turn(1 + (board != 0).sum())

        self.__compute_board(px, py)

    def pygame_init(self):
        try:
            # pygame.init()
            self._display_surf = pygame.display.set_mode(
                self.disp_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("Hex Game ({:d} x {:d})".format(self.x_dim, self.y_dim))
        except:
            # pygame.quit()
            return False

        self._running = True
        self.render()

        return True

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            (self.mouse_x, self.mouse_y) = pygame.mouse.get_pos()
            self._mouse_click = True
        elif event.type == pygame.KEYDOWN:
            self._key_press = True
            self._key = event.key

        return event.type

    def on_loop(self):
        if not self._freeze:
            if self._mouse_click:
                (i, j) = self.__get_tile_index(self.mouse_x, self.mouse_y)
            else:
                (i, j) = (-1, -1)

            if self.hex.wait_for_swap():
                self.hex.swap()
                self.render()

            self.hex.turn(i, j)

            fin = self.hex.check_finish()
            if fin > 0:
                self.render()
                time.sleep(0.5)
                print("Player {:d} wins!".format(fin))
                self._freeze = True
                self._finish = True

        if self._finish:
            if self._key_press:
                res = (self._key == pygame.K_1 or self._key == pygame.K_KP1)
                if res:
                    self._replay = True
                self._running = False

        self._mouse_click = False
        self._key_press = False

    def render(self):
        # Renders new screen (ongoing game or prompt whether to restart)
        w, h = self._display_surf.get_size()
        pygame.draw.rect(self._display_surf, BACKGROUND_COLOR, pygame.Rect(0.0, 0.0, w, h))

        if self._finish:
            fin = self.hex.get_winner()
            font_size = 48
            font = pygame.font.SysFont(None, font_size)
            msg01 = font.render("Player " + str(fin) + " wins!", True, PLAYER_COLORS[fin-1])
            msg02 = font.render("Continue playing? 1: Yes, Otherwise: No", True, PLAYER_COLORS[fin-1])
            msg01_rect = msg01.get_rect(center=(w / 2, h / 2 - font_size / 2))
            msg02_rect = msg02.get_rect(center=(w / 2, h / 2 + font_size / 2))
            self._display_surf.blit(msg01, msg01_rect)
            self._display_surf.blit(msg02, msg02_rect)
        else:
            # Draw current state of board
            for (idx, pos) in enumerate(self.hexagons_center):
                i = idx // self.x_dim
                j = idx % self.x_dim
                k = self.hex.board.get_tile(i, j)
                if k == 0:
                    self.__draw_hexagon(pos[0], pos[1], EMPTY_TILE_COLOR)
                else:
                    self.__draw_hexagon(pos[0], pos[1], PLAYER_COLORS[k-1])

            # Draw colored lines around board to highlight player's direction
            pygame.draw.lines(self._display_surf, PLAYER_COLORS[0], False, self._points_top_border, 5)
            pygame.draw.lines(self._display_surf, PLAYER_COLORS[0], False, self._points_bottom_border, 5)
            pygame.draw.lines(self._display_surf, PLAYER_COLORS[1], False, self._points_left_border, 5)
            pygame.draw.lines(self._display_surf, PLAYER_COLORS[1], False, self._points_right_border, 5)

        pygame.display.flip()

    def cleanup(self):
        pass # main program quits pygame!
        # pygame.quit()

    def execute(self):
        if not self.pygame_init():
            self._running = False

        while self._running:
            event = pygame.event.poll()
            self.on_event(event)
            self.on_loop()
            self.render()

        self.cleanup()

        return self._replay

    def __get_tile_index(self, x_pos, y_pos):
        # Check if mouse position (x_pos, y_pos) corresponds to a tile (i, j)
        # Returns (-1, -1) if not successful.
        for i in range(self.y_dim):
            idx = i * self.y_dim
            for j in range(self.x_dim):
                residual = np.array([x_pos - self.hexagons_center[idx + j][0],
                                     y_pos - self.hexagons_center[idx][1]])
                if np.linalg.norm(residual, ord=2) <= self.hexagon_inner_radius:
                    return i, j

        return -1, -1  # no success

    def __compute_board(self, px, py):
        # Determine all positions for hexagons
        # First determine maximum possible "width" of hexagon
        border_x = int(px / 15)

        inner_radius = int((px - 2 * border_x) / (2 * self.x_dim + (self.y_dim - 1)))
        outer_radius = inner_radius / cos(pi / 6)
        self.hexagon_outer_radius = outer_radius
        self.hexagon_inner_radius = inner_radius

        # We assume that the x direction is length-limiting
        border_y = int((py - (2 * outer_radius + 1.5 * (self.y_dim - 1) * outer_radius)) / 2)

        # Sorted row-wise
        self.hexagons_center = [(border_x + j * inner_radius + (1 + 2 * i) * inner_radius,
                                 border_y + outer_radius + j * (outer_radius + outer_radius / 2))
                                for j in range(self.y_dim) for i in range(self.x_dim)]

        # Also compute and save points for drawing border of board
        top_left = self.hexagons_center[0]
        bottom_right = self.hexagons_center[-1]
        points = [(top_left[0] + outer_radius * cos(-5 * pi / 6), top_left[1] + outer_radius * sin(-5 * pi / 6))]
        for i in range(self.x_dim):
            cntr = self.hexagons_center[i]
            points.append((cntr[0] + outer_radius * cos(-pi / 2), cntr[1] + outer_radius * sin(-pi / 2)))
            points.append((cntr[0] + outer_radius * cos(-pi / 6), cntr[1] + outer_radius * sin(-pi / 6)))
        self._points_top_border = points

        points = [(bottom_right[0] + outer_radius * cos(pi / 6), bottom_right[1] + outer_radius * sin(pi / 6))]
        for i in range(self.x_dim):
            cntr = self.hexagons_center[-1 - i]
            points.append((cntr[0] + outer_radius * cos(pi / 2), cntr[1] + outer_radius * sin(pi / 2)))
            points.append((cntr[0] + outer_radius * cos(5 * pi / 6), cntr[1] + outer_radius * sin(5 * pi / 6)))
        self._points_bottom_border = points

        points = []
        for j in range(self.y_dim):
            idx = j * self.x_dim
            cntr = self.hexagons_center[idx]
            points.append((cntr[0] + outer_radius * cos(-5 * pi / 6), cntr[1] + outer_radius * sin(-5 * pi / 6)))
            points.append((cntr[0] + outer_radius * cos(-7 * pi / 6), cntr[1] + outer_radius * sin(-7 * pi / 6)))
        self._points_left_border = points

        points = []
        for j in range(self.y_dim):
            idx = j * self.x_dim
            cntr = self.hexagons_center[-1 - idx]
            points.append((cntr[0] + outer_radius * cos(1 * pi / 6), cntr[1] + outer_radius * sin(1 * pi / 6)))
            points.append((cntr[0] + outer_radius * cos(-1 * pi / 6), cntr[1] + outer_radius * sin(-1 * pi / 6)))
        self._points_right_border = points

    def __draw_hexagon(self, x_pos, y_pos, color):
        points = [(x_pos + self.hexagon_outer_radius * cos(pi / 6 + pi / 3 * k),
                   y_pos + self.hexagon_outer_radius * sin(pi / 6 + pi / 3 * k)) for k in range(6)]
        pygame.draw.polygon(self._display_surf, color, points)
        pygame.draw.polygon(self._display_surf, BORDER_COLOR, points, 3)
