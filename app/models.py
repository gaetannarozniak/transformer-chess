from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GameCreateResponse:
    game_id: str
    board_state: str
    turn: str
    status: str
    legal_moves: List[str]
    captured_white: List[str]
    captured_black: List[str]


@dataclass
class MoveRequest:
    move: str
    promotion: Optional[str] = None


@dataclass
class MoveResponse:
    success: bool
    message: str
    requires_promotion: bool = False
    board_state: Optional[str] = None
    turn: Optional[str] = None
    status: Optional[str] = None
    legal_moves: Optional[List[str]] = None
    captured_white: Optional[List[str]] = None
    captured_black: Optional[List[str]] = None


@dataclass
class BoardStateResponse:
    game_id: str
    board_state: str
    turn: str
    status: str
    legal_moves: List[str]
    move_count: int
    captured_white: List[str]
    captured_black: List[str]


@dataclass
class AIMoveResponse:
    move: str
    board_state: str
    turn: str
    status: str
    legal_moves: List[str]
    captured_white: List[str]
    captured_black: List[str]


@dataclass
class LegalMovesResponse:
    game_id: str
    legal_moves_uci: List[str]
    legal_moves_san: List[str]
    count: int
