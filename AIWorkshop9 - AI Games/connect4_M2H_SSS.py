# This is a variant of the Connect Four recipe given in the easyAI library
import numpy as np
from easyAI import TwoPlayersGame, Human_Player, AI_Player, SSS  # Retain SSS*, remove unused Negamax

class GameController(TwoPlayersGame):
    def __init__(self, players, board = None):
        # Define players (Human + SSS* AI, passed from main)
        self.players = players
        # Board configuration: 6 rows × 7 columns (0=empty, 1=Player 1, 2=Player 2)
        self.board = board if (board != None) else (
            np.array([[0 for i in range(7)] for j in range(6)]))
        # Set Human as Player 1 (starts first, aligns with workshop interaction habits)
        self.nplayer = 1
        # Predefine all line directions to check for 4 consecutive discs
        # (covers horizontal, vertical, and two diagonal directions—unchanged from original)
        self.pos_dir = np.array([[[i, 0], [0, 1]] for i in range(6)] +  # Horizontal lines
                   [[[0, i], [1, 0]] for i in range(7)] +  # Vertical lines
                   [[[i, 0], [1, 1]] for i in range(1, 3)] +  # Diagonal (top-left→bottom-right, rows 1-2)
                   [[[0, i], [1, 1]] for i in range(4)] +  # Diagonal (top-left→bottom-right, cols 0-3)
                   [[[i, 6], [1, -1]] for i in range(1, 3)] +  # Diagonal (top-right→bottom-left, rows 1-2)
                   [[[0, i], [1, -1]] for i in range(3, 7)])  # Diagonal (top-right→bottom-left, cols 3-6)
    
    # Valid moves: columns with at least one empty space (min value = 0)
    def possible_moves(self):
        return [i for i in range(7) if (self.board[:, i].min() == 0)]
    
    # Execute move: drop disc to the lowest empty row in the selected column
    def make_move(self, column):
        line = np.argmin(self.board[:, column] != 0)  # Find first empty row (lowest in column)
        self.board[line, column] = self.nplayer  # Mark disc with current player's number
    
    # Show board status (human-readable: column numbers + bottom-to-top display)
    def show(self):
        print('\n' + '\n'.join(
                ['0 1 2 3 4 5 6', 13 * '-'] +  # Column labels + separator line
                [' '.join([['.', 'O', 'X'][self.board[5 - j][i]] 
                for i in range(7)]) for j in range(6)]))  # Print from bottom (row 5) to top (row 0)
    
    # Loss condition: check if opponent has 4 consecutive discs
    def loss_condition(self):
        for pos, direction in self.pos_dir:
            streak = 0  # Count consecutive opponent discs
            while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):  # Stay within board bounds
                if self.board[pos[0], pos[1]] == self.nopponent:  # Opponent's disc found
                    streak += 1
                    if streak == 4:  # 4 in a row = opponent wins
                        return True
                else:
                    streak = 0  # Reset if non-opponent disc is encountered
                pos = pos + direction  # Move to next position in the line
        return False
    
    # Game over check: board full (draw) or opponent wins
    def is_over(self):
        return (self.board.min() > 0) or self.loss_condition()
    
    # Scoring for AI: -100 if current player loses, else 0 (matches original logic)
    def scoring(self):
        return -100 if self.loss_condition() else 0

if __name__ == '__main__':
    # --------------------------
    # Key Modification: Human vs. SSS* AI (replace M2M with H2M)
    # --------------------------
    # 1. Initialize SSS* algorithm (depth=5, same as original M2M code for balanced difficulty)
    # SSS* uses "best-first" search to prioritize high-value moves, ensuring strategic decisions
    sss_ai = SSS(5)
    
    # 2. Create game instance: [Human_Player (Player 1), AI_Player (SSS*, Player 2)]
    # Human starts first (consistent with self.nplayer=1 in __init__)
    game = GameController([Human_Player(), AI_Player(sss_ai)])
    
    # 3. Start interactive gameplay (human inputs column number; AI responds automatically)
    game.play()
    
    # 4. Print clear result (label "Human" or "SSS* AI" for intuition)
    if game.loss_condition():
        winner = "Human (Player 1)" if game.nopponent == 1 else "SSS* AI (Player 2)"
        print(f'\nGame Over! Winner: {winner} (got 4 consecutive discs)')
    else:
        print("\nGame Over! It's a draw (board is full with no winner).")