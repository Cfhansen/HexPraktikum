import numpy as np
from math import sin, cos, pi
import time

import pygame
# from pygame.locals import *

from hexgame import HexGame
from hexglobals import PYGAME_COLORS

class HexApp:
  def __init__(self, board, player1, player2, px=800, py=800):
    # Display related variables
    self._display_surf = None
    self.disp_size = (px, py)

    ydim, xdim = board.shape
    self.xdim = xdim
    self.ydim = ydim

    # State of the game (+ events) related variables
    self._freeze = False
    self._finish = False
    self._mouse_click = False
    self._key_press = False
    self._key = None
    self._replay = False

    self.hex = HexGame(board, player1, player2)

    # Ensures that no swap is applied if more than 1 tile is occupied
    self.hex.set_turn(1 + (board!=0).sum())

    self.__compute_board(xdim, ydim, px, py)

  def pygame_init(self):
    pygame.init()
    self._display_surf = pygame.display.set_mode(
      self.disp_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Hex Game ({:d} x {:d})".format(self.xdim, self.ydim))

    self._running = True
    self.render()


  def on_event(self, event):
    if event.type == pygame.QUIT:
      self._running = False
    if event.type == pygame.MOUSEBUTTONDOWN:
      (xpos, ypos) = pygame.mouse.get_pos()
      self.mouse_x = xpos
      self.mouse_y = ypos
      self._mouse_click = True
    elif event.type == pygame.KEYDOWN:
      self._key_press = True
      self._key = event.key

    return event.type


  def on_loop(self):
    if not self._freeze:
      if self._mouse_click:
        (i,j) = self.__get_tile_index(self.mouse_x, self.mouse_y)
      else:
        (i,j) = (-1,-1)

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
        if res == True:
          self._replay = True
        self._running = False

    self._mouse_click = False
    self._key_press = False

  def render(self):
    # Renders new screen (ongoing game or prompt whether to restart)
    w, h = self._display_surf.get_size()
    pygame.draw.rect(self._display_surf, "black", pygame.Rect(0.0, 0.0, w, h))

    if self._finish:
      fin = self.hex.check_finish()
      fontsz = 48
      font = pygame.font.SysFont(None, fontsz)
      msg01 = font.render("Player " + str(fin) + " wins!", True, PYGAME_COLORS[fin])
      msg02 = font.render("Replay? 1: Yes, Otherwise: No", True, PYGAME_COLORS[fin])
      msg01_rect = msg01.get_rect(center=(w/2, h/2 - fontsz/2))
      msg02_rect = msg02.get_rect(center=(w/2, h/2 + fontsz/2))
      self._display_surf.blit(msg01, msg01_rect)
      self._display_surf.blit(msg02, msg02_rect)
    else:
      # Draw current state of board
      R = self.hexagon_R
      r = self.hexagon_r

      # Draw board
      for (idx, pos) in enumerate(self.hexagons_center):
        i = idx // self.xdim
        j = idx % self.xdim
        self.__draw_hexagon(pos[0], pos[1], R, PYGAME_COLORS[self.hex.board.get_tile(i,j)], 1)
      
      # Draw colored lines around board to highlight player's direction
      pygame.draw.lines(self._display_surf, PYGAME_COLORS[1], False, self._points_top_border, 4)
      pygame.draw.lines(self._display_surf, PYGAME_COLORS[1], False, self._points_bottom_border, 4)
      pygame.draw.lines(self._display_surf, PYGAME_COLORS[2], False, self._points_left_border, 4)
      pygame.draw.lines(self._display_surf, PYGAME_COLORS[2], False, self._points_right_border, 4)

    pygame.display.flip()

  def cleanup(self):
    pygame.quit()

  def execute(self):
    if self.pygame_init() == False:
      self._running = False

    while self._running:
      event = pygame.event.poll()
      self.on_event(event)
      self.on_loop()
      self.render()

    self.cleanup()

    return self._replay

  def __get_tile_index(self, xpos, ypos):
    # Check if mouse position (xpos, ypos) corresponds to a tile (i, j)
    # Returns (-1, -1) if not successful.
    for i in range(self.ydim):
      idx = i * self.ydim
      for j in range(self.xdim):
        residual = np.array([xpos - self.hexagons_center[idx+j][0],
                             ypos - self.hexagons_center[idx][1]])
        if np.linalg.norm(residual, ord=2) <= self.hexagon_r:
          return i, j

    return (-1,-1) # no success
  
  def __compute_board(self, xdim, ydim, px, py):
    # Determine all positions for hexagons
    # First determine maximum possible "width" of hexagon
    border_x = int(px / 15)
    border_y = int(py / 15)
    xlen_max = px - 2 * border_x
    ylen_max = py - 2 * border_y

    # xw is distance to edge, yw is distance to corner (aka radius)
    xw = int(xlen_max / (2*xdim + (ydim-1)))
    yw = int(ylen_max / (2*ydim))

    # R = outer radius, r = inner radius
    r = xw
    R = r / cos(pi/6)
    
    # We assume that the x direction is length-limiting
    border_y = int((py - (2*R + 1.5*(ydim-1)*R)) / 2)

    # Sorted row-wise
    self.hexagons_center = [(border_x + j*r + (1+2*i)*r,
                             border_y + R + j*(R+R/2)) for j in range(ydim) for i in range (xdim)]
    self.hexagon_R = R
    self.hexagon_r = r

    # Also compute and save points for drawing border of board
    top_left = self.hexagons_center[0]
    bottom_right = self.hexagons_center[-1]
    points = [(top_left[0] + R*cos(-5*pi/6), top_left[1] + R*sin(-5*pi/6))]
    for i in range(xdim):
      cntr = self.hexagons_center[i]
      points.append((cntr[0] + R*cos(-pi/2), cntr[1] + R*sin(-pi/2)))
      points.append((cntr[0] + R*cos(-pi/6), cntr[1] + R*sin(-pi/6)))
    self._points_top_border = points

    points = [(bottom_right[0] + R*cos(pi/6), bottom_right[1] + R*sin(pi/6))]
    for i in range(xdim):
      cntr = self.hexagons_center[-1-i]
      points.append((cntr[0] + R*cos(pi/2), cntr[1] + R*sin(pi/2)))
      points.append((cntr[0] + R*cos(5*pi/6), cntr[1] + R*sin(5*pi/6)))
    self._points_bottom_border = points

    points = []
    for j in range(self.ydim):
      idx = j * self.xdim
      cntr = self.hexagons_center[idx]
      points.append((cntr[0] + R*cos(-5*pi/6), cntr[1] + R*sin(-5*pi/6)))
      points.append((cntr[0] + R*cos(-7*pi/6), cntr[1] + R*sin(-7*pi/6)))
    self._points_left_border = points

    points = []
    for j in range(self.ydim):
      idx = j * self.xdim
      cntr = self.hexagons_center[-1-idx]
      points.append((cntr[0] + R*cos(1*pi/6), cntr[1] + R*sin(1*pi/6)))
      points.append((cntr[0] + R*cos(-1*pi/6), cntr[1] + R*sin(-1*pi/6)))
    self._points_right_border = points

  def __draw_hexagon(self, xpos, ypos, R, color, fill):
    points = [(xpos + R * cos(pi/6 + pi/3*k),
               ypos + R * sin(pi/6 + pi/3*k)) for k in range(6)]
    pygame.draw.polygon(self._display_surf, color, points)
    pygame.draw.polygon(self._display_surf, (255, 255, 255), points, 2)