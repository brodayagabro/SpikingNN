"""Replay the fixed C controller over targets and initial conditions, without training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.run_experiment import main


if __name__ == "__main__":
    main(grid=True)
