from multiprocessing import Pool
import multiprocessing
import chess
import time
from rl_environment import Observation, StockfishEnvironment
from main import RandomPolicy, run_episode
from tqdm import tqdm
import torch.nn as nn

def run_episode_timer(agent: nn.Module, env: StockfishEnvironment):
    obs, _ = env.reset()
    t_agent, t_env = 0, 0
    while True:
        t0_agent = time.perf_counter()
        action = agent(obs)
        t1_agent = time.perf_counter()
        t0_env = time.perf_counter()
        obs, reward, terminated, truncated, _ = env.step(action)
        t1_env = time.perf_counter()
        t_agent += t1_agent - t0_agent
        t_env += t1_env - t0_env
        if terminated or truncated:
            break
    return t_agent, t_env 

def run_n_episodes(n: int):
    agent = RandomPolicy()
    env = StockfishEnvironment(stockfish_time=0.001)
    t_agent, t_env = 0, 0
    start = time.perf_counter()
    for _ in range(n):
        t_agent_episode, t_env_episode = run_episode_timer(agent, env)
        t_agent += t_agent_episode
        t_env += t_env_episode
    env.close()
    end = time.perf_counter()
    print(f"time: {end - start:.2f}, agent: {t_agent}, env: {t_env}")
    return t_env

def main():
    start_total = time.perf_counter()
    with Pool() as pool:
        returns = pool.imap_unordered(run_n_episodes, [10]*100, chunksize=10)
        for t_env in returns:
            print(t_env)

    end_total = time.perf_counter()
    print(f"total time: {end_total-start_total:.2f}")

def main2():
    start_total = time.perf_counter()
    time_return = run_n_episodes(1000)

    end_total = time.perf_counter()
    print(f"total time: {end_total-start_total:.2f}")


if __name__ == "__main__":
    main()

# n_episodes = 1000, stockfish_time = 0.001

# base technique: 28.1
# multiprocessing (10 processes, 100 episodes per process): 10.5












