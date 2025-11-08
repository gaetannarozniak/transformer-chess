import chess
import uuid
import random
import chess.engine
from typing import Optional, List, Tuple
from enum import Enum
from app import config


class GameStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    DRAW = "draw"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    SEVENTYFIVE_MOVE_RULE = "seventy_five_move_rule"
    FIVEFOLD_REPETITION = "fivefold_repetition"


class Game:
    def __init__(self, fen: Optional[str] = None):
        self.game_id = str(uuid.uuid4())
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_history: List[str] = []
        self.captured_white: List[str] = []
        self.captured_black: List[str] = []

        if config.use_stockfish:
            try:
                self.stockfish = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
            except FileNotFoundError:
                raise RuntimeError(
                    f"Stockfish binary not found at '{config.stockfish_path}'. "
                    "Please check config.stockfish_path."
                )

            self.stockfish_elo = max(1350, min(2850, int(config.stockfish_elo)))
            self.stockfish.configure({
                    "UCI_LimitStrength": True,
                    "UCI_Elo": self.stockfish_elo
                })
        else:
            self.stockfish = None

    def make_move(self, move: str, promotion: Optional[str] = None) -> Tuple[bool, bool]:
        if len(move) == 4 and not promotion:
            try:
                from_square = chess.parse_square(move[:2])
                to_square = chess.parse_square(move[2:])
                piece = self.board.piece_at(from_square)
                if piece and piece.piece_type == chess.PAWN:
                    to_rank = chess.square_rank(to_square)
                    if to_rank in [0, 7]:
                        for legal_move in self.board.legal_moves:
                            if (legal_move.from_square == from_square and
                                legal_move.to_square == to_square and
                                legal_move.promotion):
                                return False, True
            except Exception:
                pass

        move_str = move
        if promotion and len(move) == 4:
            move_str = move + promotion.lower()

        try:
            if len(move_str) == 4 or (len(move_str) == 5 and move_str[4] in 'qrbn'):
                move_obj = self.board.parse_uci(move_str)
            else:
                move_obj = self.board.parse_san(move_str)
        except ValueError:
            return False, False

        if move_obj not in self.board.legal_moves:
            return False, False

        captured_piece = self.board.piece_at(move_obj.to_square)
        if captured_piece:
            piece_symbol = chess.piece_symbol(captured_piece.piece_type)
            if captured_piece.color == chess.WHITE:
                self.captured_white.append(piece_symbol.upper())
            else:
                self.captured_black.append(piece_symbol)

        self.board.push(move_obj)
        self.move_history.append(move_obj.uci())
        return True, False

    def get_status(self) -> GameStatus:
        if self.board.is_checkmate():
            return GameStatus.CHECKMATE
        elif self.board.is_stalemate():
            return GameStatus.STALEMATE
        elif self.board.is_insufficient_material():
            return GameStatus.INSUFFICIENT_MATERIAL
        elif self.board.is_seventyfive_moves():
            return GameStatus.SEVENTYFIVE_MOVE_RULE
        elif self.board.is_fivefold_repetition():
            return GameStatus.FIVEFOLD_REPETITION
        elif self.board.is_game_over():
            return GameStatus.DRAW
        else:
            return GameStatus.IN_PROGRESS

    def get_legal_moves(self) -> List[str]:
        return [move.uci() for move in self.board.legal_moves]

    def get_legal_moves_san(self) -> List[str]:
        return [self.board.san(move) for move in self.board.legal_moves]

    def get_fen(self) -> str:
        return self.board.fen()

    def get_turn(self) -> str:
        return "white" if self.board.turn == chess.WHITE else "black"

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def get_random_move(self) -> Optional[str]:
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves).uci() if legal_moves else None

    def get_stockfish_move(self) -> Optional[str]:
        if not self.stockfish:
            return self.get_random_move()
        try:
            best_move = self.stockfish.play(
                self.board, limit=chess.engine.Limit(time=config.stockfish_time_limit)
            )
            return best_move.move.uci() if best_move and best_move.move else self.get_random_move()
        except Exception:
            return self.get_random_move()

    def get_captured_pieces(self) -> Tuple[List[str], List[str]]:
        return self.captured_white, self.captured_black

    def quit_stockfish(self):
        if self.stockfish:
            self.stockfish.quit()
            self.stockfish = None
