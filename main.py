import chess
import chess.engine
from typing import List

N_UCI_MOVES = 64*63 + 8*2*4 # standard moves + promotions
moves_list: List[chess.Move] # the list of possible UCI moves, to go from integer to a move

class Transformer:
    def __init__(self):
        pass

    def __call__(self):
        pass

engine = chess.engine.SimpleEngine.popen_uci(r"/usr/local/Cellar/stockfish/17.1/bin/stockfish")

board = chess.Board()
while not board.is_game_over():
    result = engine.play(board, chess.engine.Limit(time=0.001))
    if result.move is None:
        print("result is None")
    else:
        print(result.move)
        board.push(result.move)

engine.quit()
