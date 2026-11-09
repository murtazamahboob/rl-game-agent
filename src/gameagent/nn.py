"""A small MLP with backprop + Adam, written from scratch in NumPy.

No PyTorch, no TensorFlow: two hidden layers, ReLU, He init. Used as the
DQN's Q-network. Kept deliberately tiny (12 -> 64 -> 64 -> 3) so a full
training run finishes in about a minute on a laptop CPU.
"""
from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(rng.standard_normal((fan_in, fan_out)) * scale)
            self.biases.append(np.zeros(fan_out))
        # Adam state
        self._m_w = [np.zeros_like(w) for w in self.weights]
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        self._t = 0
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []

    @property
    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 1
        h = x[None, :] if single else x
        self._cache = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            pre = h @ w + b
            self._cache.append((h, pre))
            h = relu(pre) if i < len(self.weights) - 1 else pre
        return h[0] if single else h

    def backward(self, dout: np.ndarray) -> None:
        """Backprop `dout` (d loss / d output) into .grad_w / .grad_b."""
        grad = dout
        self.grad_w = [None] * len(self.weights)
        self.grad_b = [None] * len(self.biases)
        for i in reversed(range(len(self.weights))):
            h_in, pre = self._cache[i]
            if i < len(self.weights) - 1:
                grad = grad * (pre > 0)
            self.grad_w[i] = h_in.T @ grad
            self.grad_b[i] = grad.sum(axis=0)
            grad = grad @ self.weights[i].T

    def adam_step(self, lr: float = 1e-3, beta1: float = 0.9,
                  beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._t += 1
        for i in range(len(self.weights)):
            for param, grad, m, v in (
                (self.weights[i], self.grad_w[i], self._m_w[i], self._v_w[i]),
                (self.biases[i], self.grad_b[i], self._m_b[i], self._v_b[i]),
            ):
                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * grad ** 2
                m_hat = m / (1 - beta1 ** self._t)
                v_hat = v / (1 - beta2 ** self._t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_from(self, other: "MLP") -> None:
        for w, ow in zip(self.weights, other.weights):
            w[:] = ow
        for b, ob in zip(self.biases, other.biases):
            b[:] = ob

    def save(self, path: str) -> None:
        np.savez(path, **{f"w{i}": w for i, w in enumerate(self.weights)},
                 **{f"b{i}": b for i, b in enumerate(self.biases)},
                 sizes=np.array(self.sizes))

    @classmethod
    def load(cls, path: str) -> "MLP":
        d = np.load(path)
        net = cls(list(d["sizes"]), seed=0)
        for i in range(len(net.weights)):
            net.weights[i][:] = d[f"w{i}"]
            net.biases[i][:] = d[f"b{i}"]
        return net
