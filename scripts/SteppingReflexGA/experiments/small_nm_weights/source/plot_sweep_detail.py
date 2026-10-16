"""Experiment entry point; implementation is shared in experiments/core."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.SteppingReflexGA.experiments.core.sweep_detail import main


if __name__ == "__main__":
    main()
