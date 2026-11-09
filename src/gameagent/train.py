"""Training loop with a difficulty curriculum + evaluation.

Curriculum: difficulty ramps 0 -> 1 across training, so the bomb rate
rises from 16% to 36% as the agent gets better. The agent is evaluated
greedy (epsilon = 0) on fresh seeds, and compared with a random policy
on the same seeds — the headline number for the README.
"""
from __future__ import annotations

import numpy as np

from .dqn import DQNAgent
from .env import MAX_STEPS, NeonCatchEnv


def evaluate(agent: DQNAgent, episodes: int = 20, seed: int = 999,
             difficulty: float = 0.5) -> dict:
    """Greedy evaluation. Returns mean reward / score / orbs / bombs."""
    rewards, scores, orbs, bombs = [], [], [], []
    for i in range(episodes):
        env = NeonCatchEnv(seed=seed + i, difficulty=difficulty)
        s = env.reset()
        total = 0.0
        done = False
        while not done:
            s, r, done, info = env.step(agent.act(s, epsilon=0.0))
            total += r
        rewards.append(total)
        scores.append(info["score"])
        orbs.append(info["orbs"])
        bombs.append(info["bombs"])
    return {"mean_reward": float(np.mean(rewards)),
            "mean_score": float(np.mean(scores)),
            "mean_orbs": float(np.mean(orbs)),
            "mean_bombs": float(np.mean(bombs)),
            "episodes": episodes}


def evaluate_random(episodes: int = 20, seed: int = 999,
                    difficulty: float = 0.5) -> dict:
    rng = np.random.default_rng(seed)
    rewards, scores, orbs, bombs = [], [], [], []
    for i in range(episodes):
        env = NeonCatchEnv(seed=seed + i, difficulty=difficulty)
        env.reset()
        total = 0.0
        done = False
        while not done:
            _, r, done, info = env.step(int(rng.integers(3)))
            total += r
        rewards.append(total)
        scores.append(info["score"])
        orbs.append(info["orbs"])
        bombs.append(info["bombs"])
    return {"mean_reward": float(np.mean(rewards)),
            "mean_score": float(np.mean(scores)),
            "mean_orbs": float(np.mean(orbs)),
            "mean_bombs": float(np.mean(bombs)),
            "episodes": episodes}


def train_agent(episodes: int = 800, seed: int = 0,
                target_sync: int = 25, log_every: int = 100) -> tuple[DQNAgent, dict]:
    agent = DQNAgent(seed=seed)
    env = NeonCatchEnv(seed=seed)
    rewards: list[float] = []
    losses: list[float] = []
    for ep in range(episodes):
        difficulty = ep / max(1, episodes - 1)  # curriculum 0 -> 1
        env.difficulty = difficulty
        agent.set_epsilon(ep, episodes)
        s = env.reset(seed=seed * 1000 + ep)
        total = 0.0
        done = False
        while not done:
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.buffer.add(s, a, r, s2, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            s = s2
            total += r
        rewards.append(total)
        if (ep + 1) % target_sync == 0:
            agent.sync_target()
        if log_every and (ep + 1) % log_every == 0:
            recent = float(np.mean(rewards[-log_every:]))
            print(f"episode {ep + 1:4d}/{episodes}  "
                  f"mean reward (last {log_every}) = {recent:6.2f}  "
                  f"epsilon = {agent.epsilon:.2f}  difficulty = {difficulty:.2f}")
    history = {"episodes": episodes, "rewards": rewards,
               "mean_loss": float(np.mean(losses)) if losses else 0.0,
               "final_100_mean_reward": float(np.mean(rewards[-100:])),
               "first_100_mean_reward": float(np.mean(rewards[:100])),
               "train_steps": agent.train_steps,
               "n_params": agent.online.n_params}
    return agent, history


def greedy_episode_frames(agent: DQNAgent, seed: int = 1234,
                          difficulty: float = 0.5) -> list[list[list[str]]]:
    """Play one greedy episode, returning the grid of every step."""
    env = NeonCatchEnv(seed=seed, difficulty=difficulty)
    s = env.reset()
    frames = [env.grid()]
    done = False
    while not done:
        s, _, done, _ = env.step(agent.act(s, epsilon=0.0))
        frames.append(env.grid())
    return frames
