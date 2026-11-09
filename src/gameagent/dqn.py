"""Double DQN agent with experience replay, from scratch in NumPy.

Double DQN: the online net picks the next action, the target net values
it — this cuts the over-optimism of vanilla DQN. Target net syncs every
`target_sync` episodes. Epsilon decays linearly from 1.0 to 0.05.
"""
from __future__ import annotations

import numpy as np

from .env import N_ACTIONS, STATE_SIZE
from .nn import MLP


class ReplayBuffer:
    def __init__(self, capacity: int = 5000, seed: int = 0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.buf: list[tuple] = []

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, s, a, r, s2, done) -> None:
        if len(self.buf) >= self.capacity:
            self.buf.pop(0)
        self.buf.append((np.asarray(s, dtype=np.float64), int(a), float(r),
                         np.asarray(s2, dtype=np.float64), bool(done)))

    def sample(self, batch: int) -> tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, len(self.buf), size=batch)
        s = np.stack([self.buf[i][0] for i in idx])
        a = np.array([self.buf[i][1] for i in idx])
        r = np.array([self.buf[i][2] for i in idx])
        s2 = np.stack([self.buf[i][3] for i in idx])
        d = np.array([self.buf[i][4] for i in idx], dtype=np.float64)
        return s, a, r, s2, d


class DQNAgent:
    def __init__(self, seed: int = 0, hidden: int = 64, lr: float = 1e-3,
                 gamma: float = 0.99, batch: int = 64):
        self.rng = np.random.default_rng(seed)
        self.online = MLP([STATE_SIZE, hidden, hidden, N_ACTIONS], seed=seed)
        self.target = MLP([STATE_SIZE, hidden, hidden, N_ACTIONS], seed=seed + 1)
        self.target.copy_from(self.online)
        self.buffer = ReplayBuffer(seed=seed)
        self.lr = lr
        self.gamma = gamma
        self.batch = batch
        self.epsilon = 1.0
        self.train_steps = 0

    def act(self, state: np.ndarray, epsilon: float | None = None) -> int:
        eps = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < eps:
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmax(self.online.forward(state)))

    def learn(self) -> float | None:
        """One Double-DQN gradient step on a replay batch. Returns MSE loss."""
        if len(self.buffer) < self.batch:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch)
        # Double DQN target: online net picks, target net evaluates.
        # (Compute targets first: forward() overwrites the backprop cache,
        # so the online forward on `s` must run last, just before backward.)
        next_a = np.argmax(self.online.forward(s2), axis=1)
        q_next = self.target.forward(s2)[np.arange(self.batch), next_a]
        q = self.online.forward(s)
        target = q.copy()
        target[np.arange(self.batch), a] = r + self.gamma * q_next * (1 - d)
        # MSE loss, gradient wrt the online net's output.
        err = q - target
        loss = float(np.mean(err ** 2))
        self.online.backward(2 * err / err.size)
        self.online.adam_step(lr=self.lr)
        self.train_steps += 1
        return loss

    def sync_target(self) -> None:
        self.target.copy_from(self.online)

    def set_epsilon(self, episode: int, total_episodes: int,
                    eps_end: float = 0.05) -> None:
        frac = min(1.0, episode / max(1, int(0.7 * total_episodes)))
        self.epsilon = 1.0 + frac * (eps_end - 1.0)
