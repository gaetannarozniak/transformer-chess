# Web app


```bash
cd app
uv sync
```

Need Stockfish installed
```bash
brew install stockfish
```

## Paramètres
- `--stockfish-elo` : ELO de Stockfish entre 1350 et 2850 (défaut: valeur dans config)
- `--no-stockfish` : Désactive Stockfish et utilise des coups aléatoires
- `--stockfish-path` : Chemin vers le binaire Stockfish (défaut: valeur dans config)
- `--host` : Adresse du serveur (défaut: `0.0.0.0`) 
- `--port` : Port du serveur (défaut: `8000`)


Run:
```bash
python main.py --stockfish-path $(which stockfish) --stockfish-elo 1350
```