# This is a variant of the Tic Tac Toe recipe given in the easyAI library
from easyAI import TwoPlayersGame, AI_Player, Negamax
from easyAI.Player import Human_Player

class GameController(TwoPlayersGame):
    def __init__(self, players):
        # Define the players (retained from original)
        self.players = players
        # Define who starts the game (retained: Human as Player 1)
        self.nplayer = 1 
        # MODIFY 1: Expand board to 5x5 (25 positions, 0=empty)
        self.board = [0] * 25  # Changed from [0]*9 to [0]*25

    # MODIFY 2: Update possible moves to match 5x5 (positions 1-25)
    def possible_moves(self):
        # Return 1-25 for empty positions (instead of 1-9)
        return [a + 1 for a, b in enumerate(self.board) if b == 0]
    
    # Make a move (logic unchanged, but adapts to 25-position board)
    def make_move(self, move):
        self.board[int(move) - 1] = self.nplayer

    # MODIFY 3: Update loss condition to "4 consecutive pieces" (5x5 win rule)
    def loss_condition(self):
        opponent = self.nopponent
        # All valid 4-consecutive lines in 5x5 board (rows, columns, diagonals)
        possible_combinations = [
            # Rows (5 rows × 2 sequences each: e.g., Row1: [1-4], [2-5])
            [1,2,3,4], [2,3,4,5],
            [6,7,8,9], [7,8,9,10],
            [11,12,13,14], [12,13,14,15],
            [16,17,18,19], [17,18,19,20],
            [21,22,23,24], [22,23,24,25],
            # Columns (5 columns × 2 sequences each: e.g., Col1: [1,6,11,16], [6,11,16,21])
            [1,6,11,16], [6,11,16,21],
            [2,7,12,17], [7,12,17,22],
            [3,8,13,18], [8,13,18,23],
            [4,9,14,19], [9,14,19,24],
            [5,10,15,20], [10,15,20,25],
            # Diagonals (top-left → bottom-right: 2 sequences)
            [1,7,13,19], [2,8,14,20],
            # Diagonals (top-right → bottom-left: 2 sequences)
            [5,11,17,23], [4,10,16,22]
        ]
        # Check if opponent has any 4-consecutive line (changed from 3-in-a-line)
        return any([all([(self.board[i-1] == opponent) for i in combination]) for combination in possible_combinations]) 
    
    # Check if the game is over (logic unchanged, adapts to 5x5)
    def is_over(self):
        return (self.possible_moves() == []) or self.loss_condition()
    
    # MODIFY 4: Update board display to 5x5 (instead of 3x3)
    def show(self):
        print('\n=== 5x5 Tic-Tac-Toe (Win with 4 Consecutive Pieces) ===')
        print('Position Guide: 1-5 (Row1), 6-10 (Row2), 11-15 (Row3), 16-20 (Row4), 21-25 (Row5)\n')
        # Print 5 rows (each with 5 positions)
        print('\n'.join([
            ' '.join([['.', 'O', 'X'][self.board[5*j + i]] for i in range(5)])  # Changed 3*j to 5*j
            for j in range(5)  # Changed range(3) to range(5)
        ]))
        print('\n' + '-'*20)
        
    # Compute the score (retained, but now ties to 4-consecutive win)
    def scoring(self):
        return -100 if self.loss_condition() else 0
    
if __name__ == "__main__":
    # MODIFY 5: Increase Negamax depth to 9 (handles 5x5 complexity, up from 7)
    algorithm = Negamax(9)  # Changed from Negamax(7)

    # Start the game (retained, but now runs 5x5 variant)
    GameController([Human_Player(), AI_Player(algorithm)]).play()