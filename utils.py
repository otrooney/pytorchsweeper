import torch

from ais import MinesweeperAI
from game import MinesweeperGame


def get_device():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def calculate_win_rate(width: int, height: int, mines: int, ai: MinesweeperAI, iterations: int):
    # Runs a number of Minesweeper games using an AI and return the win rate
    wins = 0

    for n in range(iterations):
        game = MinesweeperGame(width, height, mines)
        game_over = False

        while not game_over:
            guess_x, guess_y = ai.guess(game)
            valid_guess = game.guess(guess_x, guess_y)

            if game.is_over or not valid_guess:
                game_over = True

        if game.won:
            wins += 1

    return wins/iterations


def test_win_rates(ai: MinesweeperAI, iterations: int):
    # Test win rates with an AI across the Beginner, Intermediate and Expert game modes
    easy_win_rate = calculate_win_rate(9, 9, 10, ai, iterations)
    print("Win rate on Beginner: {:.2%}".format(easy_win_rate))
    medium_win_rate = calculate_win_rate(16, 16, 40, ai, iterations)
    print("Win rate on Intermediate: {:.2%}".format(medium_win_rate))
    hard_win_rate = calculate_win_rate(30, 16, 99, ai, iterations)
    print("Win rate on Expert: {:.2%}".format(hard_win_rate))
