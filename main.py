import numpy as np
import matplotlib.pyplot as plt
import tqdm
import utils
import chess
import torch
import torch.nn as nn
from torch.optim import AdamW
import random
from rl_environment import Observation, StockfishEnvironment


class RandomPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs: Observation) -> chess.Move:
        return random.choice(obs.legal_moves)

class TabularPolicy(nn.Module):
    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.logits = nn.Parameter(torch.randn(utils.NUMBER_UCI_MOVES, generator=generator, requires_grad=True))

    def forward(self, obs: Observation) -> tuple[chess.Move, torch.Tensor]:
        legal_ids: list[int] = [utils.move_to_int[move] for move in obs.legal_moves]
        mask = torch.zeros(size=(utils.NUMBER_UCI_MOVES,), dtype=torch.float32)
        mask[legal_ids] = 1.0
        masked_logits = self.logits.masked_fill(mask==0, float('-inf'))
        probs = nn.functional.softmax(masked_logits, dim=0)
        sample = torch.multinomial(probs, num_samples=1)
        logprob = torch.log(probs[sample])
        return utils.int_to_move[sample], logprob

    def compute_loss(self, gamma: float, rewards: list[float], logprobs: list[torch.Tensor]) -> torch.Tensor:
        loss = torch.zeros((1,))
        g = 0.0 
        for reward, logprob in zip(reversed(rewards), reversed(logprobs)):
            g = reward + gamma * g 
            loss += -g*logprob
        return loss


def run_episode(agent: nn.Module, env: StockfishEnvironment):
    obs, _ = env.reset()
    logprobs, rewards = [], []
    while True:
        action, logprob = agent(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        logprobs.append(logprob)
        rewards.append(reward)
        if terminated or truncated:
            break
    return logprobs, rewards, len(env.board.move_stack)

def main():
    agent = TabularPolicy()
    optimizer = AdamW(agent.parameters(), lr=0.1)
    env = StockfishEnvironment(stockfish_time=0.0001)
    lengths = []
    first_moves = chess.Board().legal_moves
    for _ in tqdm.tqdm(range(10000)):
        logprobs, rewards, episode_length = run_episode(agent, env)
        loss = agent.compute_loss(0.97, rewards, logprobs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lengths.append(episode_length)
        print([agent.logits[utils.move_to_int[move]].item() for move in first_moves])
    env.close()
    print(lengths)
    lengths = np.array(lengths)
    lengths = np.convolve(lengths, np.ones(30))
    plt.plot(lengths)
    plt.show()

if __name__ == "__main__":
    main()




