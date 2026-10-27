"""Compare C seeds, command filter times and two trajectory fine-tuning modes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.AfferentBalance2026.experiments.D_trajectory.run_experiment import main


if __name__ == '__main__':
    main()
