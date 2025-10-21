import chess
import torch.nn as nn
import random
from rl_environment import Observation, StockfishEnvironment


class RandomPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs: Observation) -> chess.Move:
        return random.choice(obs.legal_moves)


def run_episode(agent: RandomPolicy, env: StockfishEnvironment) -> int:
    obs, _ = env.reset()
    while True:
        action = agent(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return len(env.board.move_stack)

def main():
    agent = RandomPolicy()
    env = StockfishEnvironment(stockfish_time=0.01)
    for _ in range(10):
        l = run_episode(agent, env)
        print(l)
    env.close()

if __name__ == "__main__":
    main()




