# This is a variant of the Connect Four recipe given in the easyAI library
import numpy as np
from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax

class GameController(TwoPlayersGame):
    def __init__(self, players, board = None):
        # Define the players (Human + AI, passed from main)
        self.players = players
        # Board configuration: 6 rows × 7 columns (0 = empty, 1 = Player 1, 2 = Player 2)
        self.board = board if (board != None) else (
            np.array([[0 for i in range(7)] for j in range(6)]))
        # Define who starts the game (Player 1 = Human, per workshop convention)
        self.nplayer = 1
        # Predefine all possible "line directions" to check for 4 consecutive discs
        # (covers horizontal, vertical, and two diagonal directions)
        self.pos_dir = np.array([[[i, 0], [0, 1]] for i in range(6)] +  # Horizontal lines
                   [[[0, i], [1, 0]] for i in range(7)] +  # Vertical lines
                   [[[i, 0], [1, 1]] for i in range(1, 3)] +  # Diagonal (top-left → bottom-right, rows 1-2)
                   [[[0, i], [1, 1]] for i in range(4)] +  # Diagonal (top-left → bottom-right, cols 0-3)
                   [[[i, 6], [1, -1]] for i in range(1, 3)] +  # Diagonal (top-right → bottom-left, rows 1-2)
                   [[[0, i], [1, -1]] for i in range(3, 7)])  # Diagonal (top-right → bottom-left, cols 3-6)
    
    # Define valid moves: columns with at least one empty space (min value = 0)
    def possible_moves(self):
        return [i for i in range(7) if (self.board[:, i].min() == 0)]
    
    # Execute a move: drop disc to the lowest empty row in the selected column
    def make_move(self, column):
        # Find the first empty row (lowest in the column) using np.argmin
        line = np.argmin(self.board[:, column] != 0)
        self.board[line, column] = self.nplayer  # Mark the disc with current player's number
    
    # Show current board status (human-readable format)
    def show(self):
        print('\n' + '\n'.join(
                ['0 1 2 3 4 5 6', 13 * '-'] +  # Column numbers + separator line
                # Print board from bottom (row 5) to top (row 0) to match real Connect Four
                [' '.join([['.', 'O', 'X'][self.board[5 - j][i]] 
                for i in range(7)]) for j in range(6)]))
    
    # Loss condition: check if opponent has 4 consecutive discs (triggers current player's loss)
    def loss_condition(self):
        for pos, direction in self.pos_dir:
            streak = 0  # Count consecutive discs of the opponent
            while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):  # Stay within board bounds
                if self.board[pos[0], pos[1]] == self.nopponent:  # Opponent's disc found
                    streak += 1
                    if streak == 4:  # 4 consecutive = opponent wins
                        return True
                else:
                    streak = 0  # Reset count if non-opponent disc is found
                pos = pos + direction  # Move to next position in the line
        return False
    
    # Check if game is over: board full (draw) or opponent wins
    def is_over(self):
        return (self.board.min() > 0) or self.loss_condition()
    
    # Scoring for AI: -100 if current player loses (opponent has 4 in a row), else 0
    def scoring(self):
        return -100 if self.loss_condition() else 0

if __name__ == '__main__':
    # --------------------------
    # Key Modification: Human vs. Negamax AI (instead of M2M)
    # --------------------------
    # 1. Initialize Negamax algorithm (depth=5, same as original M2M code for consistency)
    # Depth=5 means AI plans 5 moves ahead to select optimal strategies
    negamax_ai = Negamax(5)
    
    # 2. Create game instance: [Human_Player (Player 1), AI_Player (Negamax, Player 2)]
    # Human starts first (matches self.nplayer=1 in __init__)
    game = GameController([Human_Player(), AI_Player(negamax_ai)])
    
    # 3. Start interactive gameplay (human inputs column number, AI responds automatically)
    game.play()
    
    # 4. Print final result after game ends
    if game.loss_condition():
        # Determine winner: if current player lost, opponent is the winner
        winner = "Human (Player 1)" if game.nopponent == 1 else "Negamax AI (Player 2)"
        print(f'\nGame Over! Winner: {winner} (got 4 consecutive discs)')
    else:
        print("\nGame Over! It's a draw (board is full with no winner).")