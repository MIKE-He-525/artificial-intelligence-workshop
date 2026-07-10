# hexapawn_M2M_2.py: Fixed (Add scoring() Method to GameController)
from easyAI import TwoPlayersGame, AI_Player, Negamax, SSS
from easyAI.Player import Human_Player

class GameController(TwoPlayersGame):
    def __init__(self, players, size = (4, 4)):
        self.size = size
        num_pawns, len_board = size
        p = [[(i, j) for j in range(len_board)] for i in [0, num_pawns - 1]]
        
        for i, d, goal, pawns in [(0, 1, num_pawns - 1, p[0]), (1, -1, 0, p[1])]:
            players[i].direction = d
            players[i].goal_line = goal
            players[i].pawns = pawns

        self.players = players
        self.nplayer = 1  # Player1 (Negamax) starts first
        self.alphabets = 'ABCDEFGHIJ'

        self.to_tuple = lambda s: (self.alphabets.index(s[0]), int(s[1:]) - 1)
        self.to_string = lambda move: ' '.join([self.alphabets[move[i][0]] + str(move[i][1] + 1) for i in (0, 1)])
    
    def possible_moves(self):
        moves = []
        opponent_pawns = self.opponent.pawns
        d = self.player.direction

        for i, j in self.player.pawns:
            if (i + d, j) not in opponent_pawns:
                moves.append(((i, j), (i + d, j)))
            if (i + d, j + 1) in opponent_pawns:
                moves.append(((i, j), (i + d, j + 1)))
            if (i + d, j - 1) in opponent_pawns:
                moves.append(((i, j), (i + d, j - 1)))

        return list(map(self.to_string, moves))
    
    def make_move(self, move):
        move = list(map(self.to_tuple, move.split(' ')))
        ind = self.player.pawns.index(move[0])
        self.player.pawns[ind] = move[1]
        if move[1] in self.opponent.pawns:
            self.opponent.pawns.remove(move[1])
    
    def loss_condition(self):
        return (any([i == self.opponent.goal_line for i, j in self.opponent.pawns])
                or (self.possible_moves() == []))
    
    def is_over(self):
        return self.loss_condition()
    
    # --------------------------
    # Fixed: Add scoring() Method (Required by SSS* Algorithm)
    # --------------------------
    def scoring(self):
        # Same logic as the standalone scoring function: -100 if current player loses, else 0
        return -100 if self.loss_condition() else 0
    
    def show(self):
        f = lambda x: '1' if x in self.players[0].pawns else ('2' if x in self.players[1].pawns else '.')
        print("\nCurrent Hexapawn Board (4x4):")
        print("\n".join([" ".join([f((i, j)) for j in range(self.size[1])]) for i in range(self.size[0])]))
        print("-" * (2 * self.size[1] - 1))


if __name__ == '__main__':
    # Standalone scoring function (retained for Negamax compatibility, though SSS* uses the class method)
    def scoring(game):
        return -100 if game.loss_condition() else 0

    # Initialize algorithms (use "scoring=scoring" for Negamax; SSS* uses game.scoring())
    negamax_algorithm = Negamax(depth=12, scoring=scoring)
    sss_algorithm = SSS(depth=12, scoring=scoring)  # SSS* will ignore this and use game.scoring()

    # Start M2M game: Player1=Negamax, Player2=SSS*
    game = GameController([
        AI_Player(negamax_algorithm),
        AI_Player(sss_algorithm)
    ], size=(4, 4))

    game.play()

    # Print result
    winner = game.nopponent
    winner_algorithm = "Negamax" if winner == 1 else "SSS*"
    print(f'\nGame Over! Player {winner} ({winner_algorithm} Algorithm) wins after {game.nmove} turns.')