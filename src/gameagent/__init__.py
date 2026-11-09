"""A Double DQN agent that learns to play the original Neon Catch arcade game — NumPy only."""
from .dqn import DQNAgent
from .env import NeonCatchEnv
from .nn import MLP
from .train import evaluate, evaluate_random, train_agent

__all__ = ["DQNAgent", "NeonCatchEnv", "MLP", "evaluate", "evaluate_random", "train_agent"]
__version__ = "1.0.0"
