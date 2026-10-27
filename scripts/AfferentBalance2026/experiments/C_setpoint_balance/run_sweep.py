"""Replay C grids or compare explicit joint/staged training methods and GA settings."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.run_experiment import main


if __name__ == "__main__":
    main(grid=True)
