import chess
import chess.engine
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Observation:
    move_stack: list[chess.Move]
    legal_moves: list[chess.Move]

class StockfishEnvironment:
    def __init__(self, stockfish_time: float) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(r"/usr/local/Cellar/stockfish/17.1/bin/stockfish")
        self.stockfish_time = chess.engine.Limit(stockfish_time)
        self.board = chess.Board()

    def reset(self, seed: int = 42) -> tuple[Observation, dict[str, Any]]:
        self.board = chess.Board()
        return self._get_observation(), dict()

    def step(self, action: chess.Move) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        # policy action
        self.board.push(action)
        if self.board.is_game_over():
            print("The agent won the game !")
            return self._get_observation(), 1.0, True, False, dict()

        # stockfish action
        stockfish_move = self.engine.play(self.board, self.stockfish_time).move
        if stockfish_move is None:
            raise ValueError("Stockfish move is None")
        self.board.push(stockfish_move)
        if self.board.is_game_over():
            return self._get_observation(), -1.0, True, False, dict()
        return self._get_observation(), 0.0, False, False, dict()

    def close(self) -> None:
        self.engine.quit()

    def render(self) -> chess.Board:
        return self.board

    def _get_observation(self) -> Observation:
        return Observation(self.board.move_stack, list(self.board.legal_moves))
