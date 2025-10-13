from itertools import product
import chess
from typing import Dict, List

int_to_move: List[chess.Move] = [] # the list of possible UCI moves, to go from integer to a move
move_to_int: Dict[chess.Move, int] = dict() # a dict that returns the id of each move (the reciproque of int_to_move)

PROMOTION_SYMBOLS = ['', 'q', 'r', 'n', 'b']
NUMBER_UCI_MOVES = 63 * 64 * 5

for from_square, to_square, promotion_piece in product(chess.SQUARE_NAMES, chess.SQUARE_NAMES, PROMOTION_SYMBOLS):
    if from_square != to_square:
        move = chess.Move.from_uci(from_square+to_square+promotion_piece)
        int_to_move.append(move)

for i, move in enumerate(int_to_move):
    move_to_int[move] = i

assert len(int_to_move) == NUMBER_UCI_MOVES, "the moves list doesn't have the right size"
for i in range(len(int_to_move)):
    assert move_to_int[int_to_move[i]] == i, "the two objects should be reciproque one with another"

