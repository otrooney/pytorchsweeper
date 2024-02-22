import torch
import numpy as np

from game import MinesweeperGame
from numpy.random import default_rng
from torch import nn


class MinesweeperAI:
    # Base class for wrapping a variety of Minesweeper AI algorithms

    def __init__(self):
        pass

    def guess(self, game: MinesweeperGame):
        heatmap = self.guess_heatmap(game)
        return np.unravel_index(np.argmax(heatmap), heatmap.shape)

    def guess_heatmap(self, game: MinesweeperGame):
        return self._guess_heatmap(game)


class RandomMinesweeperAI(MinesweeperAI):
    # AI which guesses cells randomly

    def _guess_heatmap(self, game: MinesweeperGame):
        random_guess = default_rng().random((game.width, game.height))
        return np.maximum(random_guess - game.exposedfield.astype(float), 0.0)


class CheatingMinesweeperAI(MinesweeperAI):
    # AI which cheats by avoiding cells with mines

    def _guess_heatmap(self, game: MinesweeperGame):
        random_guess = default_rng().random((game.width, game.height))
        if game.minefield_initialized:
            return np.maximum(random_guess - game.exposedfield.astype(float) - game.minefield.astype(float), 0.0)
        return random_guess


class PytorchsweeperAI(MinesweeperAI):
    # AI which wraps the Pytorchsweeper pytorch model

    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
        
    def _guess_heatmap(self, game: MinesweeperGame):
        input = np.array([game.visible_gamestate.astype('float32')/10.0])
        input_tensor = torch.from_numpy(input).to(self.device)

        with torch.no_grad():
            guess_heatmap = self.model(input_tensor)
        return guess_heatmap.to('cpu').numpy()[0]
