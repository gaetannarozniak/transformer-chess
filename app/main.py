import sys
import os
from pathlib import Path

# Add parent directory to path so we can import app modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from app import config
import argparse
import uvicorn


app = FastAPI(
    title="Chess API",
    description="REST API for playing chess with LLM integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api", tags=["chess"])

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    frontend_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Chess API is running. Frontend not found."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess API Server")
    parser.add_argument(
        "--stockfish-elo",
        type=int,
        help=f"Stockfish ELO rating (default: {config.stockfish_elo})"
    )
    parser.add_argument(
        "--no-stockfish",
        action="store_true",
        help="Disable Stockfish and use random moves instead"
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        help=f"Path to Stockfish binary (default: {config.stockfish_path})"
    )

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    args = parser.parse_args()

    if args.stockfish_elo is not None:
        config.stockfish_elo = max(1350, min(2850, args.stockfish_elo))
    if args.stockfish_path:
        config.stockfish_path = args.stockfish_path
    if args.no_stockfish:
        config.use_stockfish = False

    print("Starting Chess API server...")
    if config.use_stockfish:
        print(f"Stockfish enabled with ELO: {config.stockfish_elo}")
    else:
        print("Stockfish disabled — using random moves.")

    uvicorn.run(app, host=args.host, port=args.port)
