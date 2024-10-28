import numpy as np
import matplotlib.pyplot as plt

class TicTacToe:
    def __init__(self, size):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.dp = {}

    def is_winner(self, player):
        for i in range(self.size):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        if all([self.board[i, i] == player for i in range(self.size)]) or \
           all([self.board[i, self.size - 1 - i] == player for i in range(self.size)]):
            return True
        return False

    def is_draw(self):
        return np.all(self.board != 0)

    def dp_solve(self, player):
        if self.is_winner(1):
            return 1  # X wins
        if self.is_winner(-1):
            return -1  # O wins
        if self.is_draw():
            return 0  # Draw

        board_tuple = tuple(map(tuple, self.board))
        if board_tuple in self.dp:
            return self.dp[board_tuple]

        best_outcome = -float('inf') if player == 1 else float('inf')
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    self.board[i, j] = player
                    outcome = self.dp_solve(-player)
                    self.board[i, j] = 0
                    if player == 1:
                        best_outcome = max(best_outcome, outcome)
                    else:
                        best_outcome = min(best_outcome, outcome)

        self.dp[board_tuple] = best_outcome
        return best_outcome

def run_experiment(board_size, num_trials=100):
    wins = 0
    for _ in range(num_trials):
        game = TicTacToe(board_size)
        result = game.dp_solve(1)  # X starts first
        if result == 1:
            wins += 1
    return wins / num_trials

def plot_win_rates():
    board_sizes = [4, 5]
    win_rates = [run_experiment(size) for size in board_sizes]

    plt.bar(board_sizes, win_rates)
    plt.xlabel('Board Size')
    plt.ylabel('Win Rate of X')
    plt.title('Win Rates for Tic-Tac-Toe (Dynamic Programming)')
    plt.show()

plot_win_rates()
