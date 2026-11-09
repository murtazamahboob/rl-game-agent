"""Visuals: training-curve PNG and an animated GIF of a greedy episode."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Grid symbols -> RGB colours (neon palette on a dark board).
COLOURS = {".": (16, 18, 32), "O": (0, 230, 170), "X": (255, 70, 90), "A": (255, 210, 60)}


def training_curve(rewards: list[float], path: str, window: int = 25) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    r = np.asarray(rewards, dtype=float)
    ma = np.convolve(r, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r, color="#7aa2ff", alpha=0.35, linewidth=0.7, label="episode reward")
    ax.plot(np.arange(window - 1, len(r)), ma, color="#00e6aa", linewidth=2,
            label=f"{window}-episode mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("DQN learns Neon Catch (Double DQN, curriculum difficulty)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def episode_gif(frames: list[list[list[str]]], path: str,
                cell: int = 28, fps: int = 6) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h, w = len(frames[0]), len(frames[0][0])
    imgs = []
    for grid in frames:
        arr = np.zeros((h * cell, w * cell, 3), dtype=np.uint8)
        for y, row in enumerate(grid):
            for x, ch in enumerate(row):
                arr[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = COLOURS[ch]
        imgs.append(Image.fromarray(arr))
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
