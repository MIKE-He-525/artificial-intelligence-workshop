from easyAI import TwoPlayersGame, AI_Player, SSS 
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
        current_dir = self.player.direction  

        for (i, j) in self.player.pawns:
            forward_pos = (i + current_dir, j)
            if forward_pos not in opponent_pawns:
                moves.append(((i, j), forward_pos))
            attack_right = (i + current_dir, j + 1)
            if attack_right in opponent_pawns:
                moves.append(((i, j), attack_right))
            attack_left = (i + current_dir, j - 1)
            if attack_left in opponent_pawns:
                moves.append(((i, j), attack_left))

        return list(map(self.to_string, moves))
    
    def make_move(self, move):
        start_pos, end_pos = map(self.to_tuple, move.split(' '))
        pawn_idx = self.player.pawns.index(start_pos)
        self.player.pawns[pawn_idx] = end_pos
        if end_pos in self.opponent.pawns:
            self.opponent.pawns.remove(end_pos)
    
    def loss_condition(self):
        opponent_reach_goal = any([pos[0] == self.opponent.goal_line for pos in self.opponent.pawns])
        no_valid_moves = self.possible_moves() == []
        return opponent_reach_goal or no_valid_moves
    
    def is_over(self):
        return self.loss_condition()
    
    def scoring(self):
        return -100 if self.loss_condition() else 0
    
    def show(self):
        get_symbol = lambda pos: '1' if pos in self.players[0].pawns else ('2' if pos in self.players[1].pawns else '.')
        print("\nCurrent Hexapawn Board (4x4):")
        print("Goal Line (SSS* AI): Row 0 | Goal Line (Human): Row 3")  
        print("\n".join([f"Row {i}: " + " ".join([get_symbol((i, j)) for j in range(self.size[1])]) for i in range(self.size[0])]))
        print("Columns: A B C D → Corresponding Index: 0 1 2 3") 
        print("-" * 40) 


if __name__ == '__main__':
    sss_ai = SSS(depth=12, scoring=lambda game: -100 if game.loss_condition() else 0)
    
    players = [
        Human_Player(),         
        AI_Player(sss_ai)        
    ]

  
    print("=== Hexapawn Game: Human vs SSS* AI ===")
   
  
    game = GameController(players, size=(4, 4))
    game.play()

    
    winner = game.nopponent  
    winner_name = "Human (You)" if winner == 1 else "SSS* AI"
    print(f'\n=== Game Over! ===')
    print(f"Final Winner: {winner_name}")
    print(f"Total Turns: {game.nmove}（每回合包含人类和 AI 各一次移动）")