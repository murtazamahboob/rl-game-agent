"""Neon Catch — a small original arcade game for reinforcement learning.

A catcher slides along the bottom row of a 10 x 8 grid while orbs (good)
and bombs (bad) fall from the top. Catch orbs (+1), dodge bombs (-1),
and don't let orbs hit the floor (-0.2). Nothing here is a Gym/CartPole
clone: the falling-object mix, the moving hazards and the curriculum
(bomb rate rises as the agent improves) are designed for this project.

State (12 numbers, all in [-1, 1] roughly):
  [agent_x, orbs_caught, bombs_hit, step_fraction,
   dx, dy, kind, dx, dy, kind, dx, dy, kind]  for the 3 objects
   nearest the floor (kind = +1 orb, -1 bomb; 0 = no object).
"""
from __future__ import annotations

import numpy as np

WIDTH = 10
HEIGHT = 8
MAX_STEPS = 120
N_ACTIONS = 3  # 0 = left, 1 = stay, 2 = right
STATE_SIZE = 12

ORB_REWARD = 1.0
BOMB_PENALTY = -1.0
MISS_PENALTY = -0.2
STEP_PENALTY = -0.01  # tiny nudge against standing still under a bomb


class NeonCatchEnv:
    """Deterministic-when-seeded falling-objects game."""

    def __init__(self, seed: int = 0, difficulty: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.agent_x = WIDTH // 2
        self.objects: list[dict] = []  # {"x": int, "y": int, "kind": +1|-1}
        self.step_count = 0
        self.orbs_caught = 0
        self.bombs_hit = 0
        self.orbs_missed = 0

    # -- spawning / difficulty -------------------------------------------------
    @property
    def orb_rate(self) -> float:
        return 0.42 - 0.07 * self.difficulty

    @property
    def bomb_rate(self) -> float:
        return 0.16 + 0.20 * self.difficulty

    def _spawn(self) -> None:
        r = self.rng.random()
        if r < self.orb_rate:
            self.objects.append({"x": int(self.rng.integers(WIDTH)), "y": 0, "kind": 1})
        elif r < self.orb_rate + self.bomb_rate:
            self.objects.append({"x": int(self.rng.integers(WIDTH)), "y": 0, "kind": -1})

    # -- core loop --------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_x = WIDTH // 2
        self.objects = []
        self.step_count = 0
        self.orbs_caught = 0
        self.bombs_hit = 0
        self.orbs_missed = 0
        return self.state()

    def state(self) -> np.ndarray:
        near = sorted(self.objects, key=lambda o: -o["y"])[:3]
        s = [self.agent_x / (WIDTH - 1) * 2 - 1,
             self.orbs_caught / 10.0, self.bombs_hit / 10.0,
             self.step_count / MAX_STEPS]
        for o in near:
            s += [(o["x"] - self.agent_x) / WIDTH,
                  (HEIGHT - 1 - o["y"]) / HEIGHT,
                  float(o["kind"])]
        while len(s) < STATE_SIZE:
            s += [0.0, 0.0, 0.0]
        return np.array(s[:STATE_SIZE], dtype=np.float64)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action < N_ACTIONS, f"action {action} out of range"
        self.agent_x = int(np.clip(self.agent_x + action - 1, 0, WIDTH - 1))
        self.step_count += 1
        reward = STEP_PENALTY

        # Objects fall one row; resolve anything reaching the floor.
        landed = []
        for o in self.objects:
            o["y"] += 1
            if o["y"] >= HEIGHT - 1:
                landed.append(o)
        for o in landed:
            self.objects.remove(o)
            if o["x"] == self.agent_x:
                if o["kind"] == 1:
                    reward += ORB_REWARD
                    self.orbs_caught += 1
                else:
                    reward += BOMB_PENALTY
                    self.bombs_hit += 1
            elif o["kind"] == 1:
                reward += MISS_PENALTY
                self.orbs_missed += 1

        self._spawn()
        done = self.step_count >= MAX_STEPS
        info = {"orbs": self.orbs_caught, "bombs": self.bombs_hit,
                "missed": self.orbs_missed, "score": self.orbs_caught - self.bombs_hit}
        return self.state(), reward, done, info

    # -- rendering ---------------------------------------------------------------
    def grid(self) -> list[list[str]]:
        g = [["." for _ in range(WIDTH)] for _ in range(HEIGHT)]
        for o in self.objects:
            g[o["y"]][o["x"]] = "O" if o["kind"] == 1 else "X"
        g[HEIGHT - 1][self.agent_x] = "A"
        return g

    def render_ascii(self) -> str:
        return "\n".join(" ".join(row) for row in self.grid())
