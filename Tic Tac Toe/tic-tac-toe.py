import pygame

pygame.init()
WIDTH, HEIGHT = 600, 600
ROWS = 3
SCORE_FONT = pygame.font.SysFont("comicsans", 50)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class GameInformation:
    def __init__(self, p1_win, p2_win):
        self.p1_win = p1_win
        self.p2_win = p2_win

class Game:
    def __init__(self, window, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height
        self.p1_win = 0
        self.p2_win = 0
        self.window = window

        row = [0] * ROWS
        board = [row] * ROWS
        self.board = board

    def draw_lines():
        pass

