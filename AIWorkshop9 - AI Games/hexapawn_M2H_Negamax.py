from easyAI import TwoPlayersGame, AI_Player, Negamax
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
        self.nplayer = 1  
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
    
    def scoring(self):
        return -100 if self.loss_condition() else 0
    
    def show(self):
        f = lambda x: '1' if x in self.players[0].pawns else ('2' if x in self.players[1].pawns else '.')
        print("\nCurrent Hexapawn Board (4x4):")
        print("Row 0 (Player2's Goal) → Row 3 (Player1's Goal)")  # 新增：提示目标线
        print("\n".join([f"Row {i}: " + " ".join([f((i, j)) for j in range(self.size[1])]) for i in range(self.size[0])]))
        print("Columns: A B C D (对应 0-3)")  # 新增：提示列坐标
        print("-" * (2 * self.size[1] + 8))


if __name__ == '__main__':
 
    negamax_ai = Negamax(depth=12, scoring=lambda game: -100 if game.loss_condition() else 0)
    
    players = [
        Human_Player(),         
        AI_Player(negamax_ai)   
    ]

    print("=== Hexapawn: Human vs Negamax AI ===")
  
    
    game = GameController(players, size=(4, 4))
    game.play()

  
    winner = game.nopponent  
    winner_type = "Human (Player1)" if winner == 1 else "Negamax AI (Player2)"
    print(f'\n=== Game Over! ===')
    print(f'Winner: {winner_type}')
    print(f'Total Turns: {game.nmove}')