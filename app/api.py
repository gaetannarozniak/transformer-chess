from fastapi import APIRouter, HTTPException, status
from typing import Dict
from app.game import Game
from app import config
from app.models import (
    GameCreateResponse,
    MoveRequest,
    MoveResponse,
    BoardStateResponse,
    AIMoveResponse,
    LegalMovesResponse
)

router = APIRouter()
games: Dict[str, Game] = {}


@router.get("/config")
async def get_config():
    return {
        "use_stockfish": config.use_stockfish,
        "stockfish_elo": config.stockfish_elo
    }


@router.post("/games", status_code=status.HTTP_201_CREATED)
async def create_game():
    try:
        game = Game()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Engine initialization failed: {e}"
        )
    games[game.game_id] = game
    captured_white, captured_black = game.get_captured_pieces()
    return GameCreateResponse(
        game_id=game.game_id,
        board_state=game.get_fen(),
        turn=game.get_turn(),
        status=game.get_status().value,
        legal_moves=game.get_legal_moves(),
        captured_white=captured_white,
        captured_black=captured_black
    )


@router.get("/games/{game_id}")
async def get_game(game_id: str):
    if game_id not in games:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game {game_id} not found"
        )
    game = games[game_id]
    captured_white, captured_black = game.get_captured_pieces()
    return BoardStateResponse(
        game_id=game.game_id,
        board_state=game.get_fen(),
        turn=game.get_turn(),
        status=game.get_status().value,
        legal_moves=game.get_legal_moves(),
        move_count=len(game.move_history),
        captured_white=captured_white,
        captured_black=captured_black
    )


@router.post("/games/{game_id}/move")
async def make_move(game_id: str, move_request: MoveRequest):
    if game_id not in games:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game {game_id} not found"
        )

    if move_request.promotion and move_request.promotion.lower() not in {"q", "r", "b", "n"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid promotion piece. Use one of q, r, b, n."
        )

    game = games[game_id]
    if game.is_game_over():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Game finished. Status: {game.get_status().value}"
        )

    success, requires_promotion = game.make_move(
        move_request.move, promotion=move_request.promotion
    )

    if requires_promotion:
        return MoveResponse(
            success=False,
            message="Promotion required",
            requires_promotion=True
        )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid move: {move_request.move}"
        )

    if game.is_game_over():
        game.quit_stockfish()

    captured_white, captured_black = game.get_captured_pieces()
    return MoveResponse(
        success=True,
        message=f"Move {move_request.move} executed",
        requires_promotion=False,
        board_state=game.get_fen(),
        turn=game.get_turn(),
        status=game.get_status().value,
        legal_moves=game.get_legal_moves(),
        captured_white=captured_white,
        captured_black=captured_black
    )


@router.post("/games/{game_id}/ai-move")
async def get_ai_move(game_id: str):
    if game_id not in games:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game {game_id} not found"
        )
    game = games[game_id]
    if game.is_game_over():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Game finished. Status: {game.get_status().value}"
        )

    ai_move = game.get_stockfish_move() if config.use_stockfish else game.get_random_move()
    if ai_move is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No legal moves available"
        )

    ok, needs_promo = game.make_move(ai_move)
    if not ok or needs_promo:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI produced an invalid or incomplete move"
        )

    if game.is_game_over():
        game.quit_stockfish()

    captured_white, captured_black = game.get_captured_pieces()
    return AIMoveResponse(
        move=ai_move,
        board_state=game.get_fen(),
        turn=game.get_turn(),
        status=game.get_status().value,
        legal_moves=game.get_legal_moves(),
        captured_white=captured_white,
        captured_black=captured_black
    )


@router.get("/games/{game_id}/legal-moves")
async def get_legal_moves(game_id: str):
    if game_id not in games:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game {game_id} not found"
        )
    game = games[game_id]
    legal_moves_uci = game.get_legal_moves()
    legal_moves_san = game.get_legal_moves_san()
    return LegalMovesResponse(
        game_id=game.game_id,
        legal_moves_uci=legal_moves_uci,
        legal_moves_san=legal_moves_san,
        count=len(legal_moves_uci)
    )
