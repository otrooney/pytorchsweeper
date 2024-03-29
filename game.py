import numpy as np
import random


class MinesweeperGame:
    # Minesweeper game implemented via numpy arrays. The game state consists of two
    # boolean arrays, each the same size.
    # `minefield` is true where there is a mine, false otherwise.
    # `exposedfield` is true for each cell revealed to the player.
    # A third array, `numberfield` stores the number of adjacent cells containing mines,
    # and is calculated when the minefield is initialized.
    
    def __init__(self, width: int, height: int, mines: int, auto_first_guess=False):
        self.width = width
        self.height = height
        self.mines = mines
        self.minefield_initialized = False
        self.minefield = None
        self.exposedfield = np.array([[False]*height]*width, dtype=bool)
        self.numberfield = None
        if auto_first_guess:
            self.guess(random.randrange(self.width), random.randrange(self.height))
    
    def guess(self, x: int, y: int):
        # Returns true if the guess is valid, that is a cell which had not already been
        # revealed and does not contain a mine. In all other cases returns false.
        if x<0 or x>=self.width or y<0 or y>=self.height:
            return False
        if self.exposedfield[x][y]:
            return False
        if not self.minefield_initialized:
            self.generate_minefield(x, y)
            self.generate_numberfield()
            self.minefield_initialized = True
        self.exposedfield[x][y] = True
        # If a cell is chosen with no adjacent mines, then the `guess` method is
        # recursively called on adjacent cells, revealing the region around it which
        # contains no mines.
        if self.numberfield[x][y] == 0:
            for new_x in [x-1, x, x+1]:
                for new_y in [y-1, y, y+1]:
                    self.guess(new_x, new_y)
        return True
    
    def generate_minefield(self, avoid_x: int, avoid_y: int):
        # The minefield is not generated until a guess is made, to prevent players losing
        # to unlucky initial guesses. Mines will not be placed on the cell chosen by the
        # player, or on surrounding cells.
        mine_array = np.array([[False]*self.height]*self.width, dtype=bool)
        while mine_array.sum() < self.mines:
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            if not (avoid_x-1 <= x <= avoid_x+1 and avoid_y-1 <= y <= avoid_y+1):
                mine_array[x][y] = True
        self.minefield = mine_array
        
    def generate_numberfield(self):
        # The numberfield can be generated by summing a series of shifted arrays.
        pf = np.pad(self.minefield, 1).astype(int)
        self.numberfield = (
            pf[:self.width, :self.height]
            + pf[1:self.width+1, :self.height]
            + pf[2:self.width+2, :self.height]
            + pf[:self.width, 1:self.height+1]
            + pf[2:self.width+2, 1:self.height+1]
            + pf[:self.width, 2:self.height+2]
            + pf[1:self.width+1, 2:self.height+2]
            + pf[2:self.width+2, 2:self.height+2]
        )

    @property
    def lost(self):
        # The game has been lost if there exists any cell which both contains a mine
        # and has been exposed.
        if self.minefield_initialized:
            return np.any(np.logical_and(self.minefield, self.exposedfield))
        return False
    
    @property
    def won(self):
        # The game has been won if every cell either contains a mine or has been exposed.
        if self.minefield_initialized:
            return np.all(np.logical_xor(self.minefield, self.exposedfield))
        return False
    
    @property
    def is_over(self):
        return self.lost or self.won
    
    @property
    def visible_gamestate(self):
        # The visible gamestate is what the game presents to the player. It is generated by
        # masking the numberfield by the exposedfield, and filling the remaining values with
        # -10, to represent unknown cells.
        if self.minefield_initialized:
            return np.ma.array(self.numberfield, mask=np.logical_not(self.exposedfield), fill_value=-10).filled()
        return np.full((self.width, self.height), -10)
