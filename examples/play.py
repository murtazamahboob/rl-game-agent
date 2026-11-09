"""Play Neon Catch yourself, before the agent takes over.

Run from the project root:  python examples/play.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gameagent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["play"]))
