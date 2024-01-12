import itertools
import numpy as np
import random
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.expected_conditions import element_to_be_clickable, presence_of_element_located, \
    invisibility_of_element_located
from selenium.webdriver.support.wait import WebDriverWait as DriverWait

from base_player import BasePlayer


class PuzzlePlayer(BasePlayer):
    """
    Player to do puzzles on 'http://www.mseymour.ca/hex_puzzle/hexpuzzle.html' within our framework.
    Initialize board with 'get_board'-function of this class.
    PuzzlePlayer must be second player and HexApp needs parameter 'use_puzzle_game=True'!
    """

    def __init__(self):
        super().__init__(swap_fun=lambda x: False)
        self.id = 2

        self.level = 0
        self.terminated = 0
        self.reflect_board = False
        self.board = None

        self.driver = webdriver.Chrome()
        self.driver.get('http://www.mseymour.ca/hex_puzzle/hexpuzzle.html')

    def get_board(self, level_number, size=11, pass_if_required=True) -> np.ndarray:
        """
        :param level_number: int
            level to be played
        :param size: int
            size of the game (square of according side length)
            puzzle will be embedded accordingly that logic works and all other fields filled
        :param pass_if_required: Bool
            if puzzle requires to first click the 'Pass'-button because puzzle is already solved,
            do this first before playing
            (logic for passing is not part of the task, thus keep out and still play remaining puzzle)
        :return: numpy array
            board of puzzle displayed on website completed to board of required size
        """
        self.terminated = 0
        self.level = level_number

        self._navigate_driver_to_level(level_number)
        self.reflect_board = (self.driver.find_element(By.ID, 'puzzle-info').text == 'White to move')

        self.board = self._initialize_board(size=size)

        if pass_if_required and level_number >= 7:
            # pass button only exists starting from level 7
            self.driver.find_element(By.ID, 'button-puzzle-pass').click()
            # wait a moment such that the website should display that wrong if passing was wrong
            time.sleep(1)
            if self.driver.find_element(By.ID, 'xmark').is_displayed():
                self.driver.find_element(By.ID, 'button-puzzle-reset').click()

            else:
                # calling once choose_tile updates the board with the additionally occupied field
                self.choose_tile(None)

        DriverWait(self.driver, 10).until(invisibility_of_element_located((By.ID, 'xmark')))
        return self.board

    def get_whether_terminated(self, board):
        """
        overrun own logic and only use whether website determines that game is won or lost
        :param board: numpy array
            current board of game
        :return: int
            0: game is not terminated
            1: other player won (puzzle was solved)
            2: puzzle player won (puzzle was not solved/other player lost)
        """
        self._play_opponent_on_website(board)
        if self.driver.find_element(By.ID, 'puzzle-info').text.__contains__('Correct'):
            self.terminated = 1
        elif self.driver.find_element(By.ID, 'xmark').is_displayed():
            self.terminated = 2

        return self.terminated

    def choose_tile(self, board, *args) -> tuple:
        """
        :param board:
            (not used as other player's move should already be known from determining whether game terminated)
        :param args:
            (not used)
        :return: tuple (int, int)
            coordinates (within whole board used by game) of tile chosen by the website
        """
        assert self.id == 2
        assert self.terminated == 0
        board_section = _extract_puzzle(self._get_puzzle_group())

        for x, y in itertools.product(range(board_section.shape[0]), range(board_section.shape[1])):
            if board_section[x][y] > 0:
                if not self.reflect_board and board_section[x][y] != self.board[x + 1, y + 1]:
                    assert board_section[x][y] == 2
                    self.board[x + 1][y + 1] = 2
                    print('Puzzle made move (', x + 1, ', ', y + 1, ')')
                    return x + 1, y + 1
                elif self.reflect_board and board_section[x][y] != 3 - self.board[y + 1, x + 1]:
                    assert board_section[x][y] == 1
                    self.board[y + 1][x + 1] = 2
                    print('Puzzle made move (', y + 1, ', ', x + 1, ')')
                    return y + 1, x + 1

    def shut_down(self) -> None:
        """
        closes the webdriver used to play the puzzle, should be called after using before stopping the program
        :return: None
        """
        self.driver.close()

    # "Private" methods
    def _navigate_driver_to_level(self, level_number) -> None:
        level_id = 'Problem ' + str(level_number)
        DriverWait(self.driver, 5).until(element_to_be_clickable((By.XPATH, '//*[@id="footer"]/button[2]'))).click()
        DriverWait(self.driver, 5).until(element_to_be_clickable((By.XPATH, "//*[contains(text(), '%s')]" % level_id)))
        level_btn = self.driver.find_element(By.XPATH, "//*[contains(text(), '%s')]" % level_id)
        level_btn.click()

    def _get_puzzle_group(self):
        bs = BeautifulSoup(self.driver.page_source, features="html.parser")
        return bs.find('div', id="puzzle-wrapper").find('g', id="puzzle-group")

    def _play_opponent_on_website(self, new_board) -> None:
        index = np.where((new_board.board - self.board) > 0)
        if len(index[0]) == 0:
            # tried to play every time when checking whether terminated, e.g., also before first move is made
            return
        assert len(index[0]) == 1, 'Only one tile should have changed'
        x = index[0][0]
        y = index[1][0]
        self.board[x][y] = 1

        time.sleep(1)

        # need to shift index back (due to embedding)
        string_index = 's_' + str(x - 1) + '_' + str(y - 1) if not self.reflect_board \
            else 's_' + str(y - 1) + '_' + str(x - 1)
        DriverWait(self.driver, 10).until(presence_of_element_located((By.ID, string_index)))
        for bt in self.driver.find_elements(By.ID, string_index):
            try:
                bt.click()
                print('Played move (', x, ', ', y, ') on website')
            except WebDriverException:
                pass

    def _initialize_board(self, size) -> np.ndarray:

        puzzle_group = self._get_puzzle_group()
        puzzle_section = _extract_puzzle(puzzle_group)

        whole_board = _embed_puzzle_in_full_sized_board(puzzle_section, puzzle_group.find_all(), size)

        if self.reflect_board:
            whole_board = _swap_colours_and_rotate_90_degrees(whole_board)

        return whole_board.astype(int)


# "Private" functions
def _extract_puzzle(puzzle_group):
    children = puzzle_group.find_all(lambda tag: tag.name == "circle")
    max_length = int(len(children) / 2) + 1
    board = np.zeros((max_length, max_length)) - 1
    x_values = []
    y_values = []
    for child in children:
        try:
            _, x, y = child.attrs['id'].split('_')
        except ValueError:
            # raised if id-tag has not required format
            break

        # get board index and label
        x = int(x)
        y = int(y)
        x_values.append(x)
        y_values.append(y)
        player_ = child.attrs['class']

        if player_[0] == 'blackStone':
            board[x][y] = 1
        elif player_[0] == 'whiteStone':
            board[x][y] = 2
        elif player_[0] == 'noStone':
            board[x][y] = 0
    return board[:max(x_values) + 1, :max(y_values) + 1]


def _swap_colours_and_rotate_90_degrees(whole_board):
    ones = (whole_board == 1)
    twos = (whole_board == 2)
    whole_board[ones], whole_board[twos] = 2, 1
    whole_board = whole_board.transpose()
    return whole_board


def _embed_puzzle_in_full_sized_board(board_section, puzzle_tags, size) -> np.ndarray:
    arrows, edges = _get_arrows_and_edges(puzzle_tags)
    board = _surround_fill_puzzle(arrows, board_section, edges)
    whole_board = _embed_in_wished_size(size, board, arrows, board_section.shape)
    return whole_board


def _embed_in_wished_size(size, board, arrows, puzzle_section_shape) -> np.ndarray:
    if size < max(board.shape[0], board.shape[1]):
        raise Exception('Size of board must be at least as large as puzzle.')
    whole_board = np.pad(board,
                         pad_width=((0, size - board.shape[0]), (0, size - board.shape[1])),
                         mode='constant',
                         constant_values=(-1, -1))
    _connect_puzzle_board_to_frame(whole_board, arrows, puzzle_section_shape)
    _colour_empty_fields_outside_puzzle(size, whole_board)
    return whole_board


def _get_arrows_and_edges(puzzle_tags) -> tuple:
    arrows = []
    edges = []
    x = -1
    y = -1
    for tag in puzzle_tags:
        # subsequent tags have the id-tag with the field and the direction of the rotation
        if 'id' in tag.attrs.keys():
            try:
                # position of stones: "id"-tag of fields have format "s_{x-position}_{y-position}"
                _, x, y = tag.attrs['id'].split('_')
                x = int(x)
                y = int(y)
            except ValueError:
                # ValueError if id-tag has not required format
                pass

        elif 'transform' in tag.attrs.keys():
            transform = tag.attrs['transform'].split(' ')
            if len(transform) >= 3 and transform[2] in ['rotate(60)', 'rotate(120)', 'rotate(-60)', 'rotate(-120)']:
                # add one to both coordinates as puzzle surrounded by opponent
                arrows.append((x + 1, y + 1, transform[2]))

        if 'filter' in tag.attrs.keys() and tag.attrs['filter'] == 'textureFilter':
            edges.append(tag)

    return arrows, edges


def _connect_puzzle_board_to_frame(whole_board, arrows, puzzle_section_shape) -> None:
    arrow_down = False
    arrow_up = False
    arrow_to_right = False
    arrow_to_left = False

    for x, y, arrow_direction in arrows:
        if arrow_direction == 'rotate(60)':
            whole_board[x:, y] = np.ones(whole_board.shape[1] - x) * 1
            arrow_down = True
        elif arrow_direction == 'rotate(120)':
            whole_board[x, :y] = np.ones(y) * 2
            arrow_to_left = True
        elif arrow_direction == 'rotate(-60)':
            whole_board[x, y:] = np.ones(whole_board.shape[0] - y) * 2
            arrow_to_right = True
        elif arrow_direction == 'rotate(-120)':
            whole_board[:x, y] = np.ones(x) * 1
            arrow_up = True

    if not arrow_up:
        whole_board[0, 1] = 1
    if not arrow_to_left:
        whole_board[1, 0] = 2
    if not arrow_to_right:
        whole_board[1, puzzle_section_shape[1] + 1:] = np.ones(whole_board.shape[1] - puzzle_section_shape[1] - 1) * 2
    if not arrow_down:
        whole_board[puzzle_section_shape[0] + 1:, 1] = np.ones(whole_board.shape[0] - puzzle_section_shape[0] - 1)


def _colour_empty_fields_outside_puzzle(size, whole_board) -> None:
    number_of_fields_one = len(np.nonzero(whole_board == 1)[0])
    empty_fields = list(zip(*np.where(whole_board == -1)))
    player_1 = random.sample(empty_fields, min(len(empty_fields), max(int(size ** 2 / 2 - number_of_fields_one), 0)))
    for field in empty_fields:
        whole_board[field] = 1 if field in player_1 else 2


def _surround_fill_puzzle(arrows, board_section, edges) -> np.ndarray:
    # surround and occupy fields within range by colour that is not used on arrows
    arrow_colours = {board_section[x - 1, y - 1] for x, y, _ in arrows}
    assert len(arrow_colours) == 1 or len(edges) == 4, \
        'Puzzles should only contain arrows for exactly one colour, or no arrows but four edges'
    colour_to_surround = 3 - arrow_colours.pop() if len(arrow_colours) == 1 else 1

    board = np.zeros((board_section.shape[0] + 2, board_section.shape[1] + 2)) + colour_to_surround
    for x, y in itertools.product(range(board_section.shape[0]), range(board_section.shape[1])):
        board[x + 1][y + 1] = board_section[x][y] if board_section[x][y] >= 0 else colour_to_surround

    _adapt_surrounding_for_edges(board, edges)

    return board


def _adapt_surrounding_for_edges(board, edges) -> None:
    for edge_tag in edges:
        white_or_black = 1 if edge_tag.attrs['class'][1] == 'blackEdge' else 2
        border_rotation = edge_tag.attrs['transform'].split(' ')[0] + ')'
        if border_rotation == 'rotate(0)':
            board[0] = np.ones(board.shape[1]) * white_or_black
        elif border_rotation == 'rotate(60)':
            board[:, -1] = np.ones(board.shape[0]) * white_or_black
        elif border_rotation == 'rotate(180)':
            board[-1] = np.ones(board.shape[1]) * white_or_black
        elif border_rotation == 'rotate(240)':
            board[:, 0] = np.ones(board.shape[0]) * white_or_black
