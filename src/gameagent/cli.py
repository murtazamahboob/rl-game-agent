"""CLI: `gameagent train|demo|play|eval`.

- train: train a DQN and save it to outputs/model.npz
- demo: train + evaluate vs random + write report, curve PNG and GIF
- play: play Neon Catch yourself (a = left, d = right, Enter = stay)
- eval: load a saved model and report greedy vs random scores
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .dqn import DQNAgent
from .env import NeonCatchEnv
from .nn import MLP
from .render import episode_gif, training_curve
from .train import evaluate, evaluate_random, greedy_episode_frames, train_agent


def cmd_train(args) -> int:
    agent, history = train_agent(episodes=args.episodes, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agent.online.save(args.out)
    print(f"saved {args.out}  final-100 mean reward={history['final_100_mean_reward']:.2f}")
    return 0


def cmd_demo(args) -> int:
    agent, history = train_agent(episodes=args.episodes, seed=args.seed)
    trained = evaluate(agent, episodes=20, seed=999)
    random = evaluate_random(episodes=20, seed=999)
    os.makedirs("outputs", exist_ok=True)
    agent.online.save("outputs/model.npz")
    with open("outputs/history.json", "w") as f:
        json.dump({**history, "trained_eval": trained, "random_eval": random},
                  f, indent=2)
    training_curve(history["rewards"], "outputs/training_curve.png")
    frames = greedy_episode_frames(agent, seed=1234)
    episode_gif(frames, "outputs/greedy_episode.gif")
    report = (
        "# Demo report — RL Game Agent (DQN plays Neon Catch)\n\n"
        f"- Q-network: {history['n_params']:,} parameters, "
        f"{history['train_steps']:,} gradient steps over {history['episodes']} episodes\n"
        f"- First-100 mean reward: {history['first_100_mean_reward']:.2f}\n"
        f"- Final-100 mean reward: {history['final_100_mean_reward']:.2f}\n"
        f"- Mean TD loss: {history['mean_loss']:.4f}\n\n"
        "## Greedy agent vs random policy (20 episodes each, same seeds, difficulty 0.5)\n\n"
        f"| policy | mean reward | mean score (orbs − bombs) | orbs/ep | bombs/ep |\n"
        f"|---|---|---|---:|---:|\n"
        f"| trained DQN | {trained['mean_reward']:.2f} | {trained['mean_score']:.2f} "
        f"| {trained['mean_orbs']:.1f} | {trained['mean_bombs']:.1f} |\n"
        f"| random | {random['mean_reward']:.2f} | {random['mean_score']:.2f} "
        f"| {random['mean_orbs']:.1f} | {random['mean_bombs']:.1f} |\n\n"
        "![Training curve](training_curve.png)\n\n"
        "![Greedy episode](greedy_episode.gif)\n"
    )
    with open("outputs/demo_report.md", "w") as f:
        f.write(report)
    print(f"demo done: trained {trained['mean_reward']:.2f} vs random "
          f"{random['mean_reward']:.2f}  report=outputs/demo_report.md")
    return 0


def cmd_play(args) -> int:
    env = NeonCatchEnv(seed=args.seed, difficulty=0.3)
    env.reset()
    total = 0.0
    print("Neon Catch — you are A. a = left, d = right, Enter = stay. q quits.")
    print(env.render_ascii())
    while True:
        try:
            key = input("move [a/d/Enter/q]: ").strip().lower()
        except EOFError:
            break
        if key == "q":
            break
        action = {"a": 0, "d": 2}.get(key, 1)
        _, r, done, info = env.step(action)
        total += r
        print(env.render_ascii())
        print(f"reward {r:+.2f}  total {total:.2f}  orbs {info['orbs']}  "
              f"bombs {info['bombs']}  score {info['score']}")
        if done:
            print(f"game over — final score {info['score']} "
                  f"({info['orbs']} orbs, {info['bombs']} bombs)")
            break
    return 0


def cmd_eval(args) -> int:
    agent = DQNAgent(seed=0)
    agent.online = MLP.load(args.model)
    agent.target.copy_from(agent.online)
    trained = evaluate(agent, episodes=20, seed=999)
    random = evaluate_random(episodes=20, seed=999)
    print(f"trained: mean reward {trained['mean_reward']:.2f}, "
          f"score {trained['mean_score']:.2f}")
    print(f"random:  mean reward {random['mean_reward']:.2f}, "
          f"score {random['mean_score']:.2f}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gameagent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/model.npz")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("demo")
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_demo)
    p = sub.add_parser("play")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_play)
    p = sub.add_parser("eval")
    p.add_argument("--model", default="outputs/model.npz")
    p.set_defaults(func=cmd_eval)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
