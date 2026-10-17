"""3. Заданный угол через постоянные моторные токи. Calculation and search are imported from experiments/core."""
from pathlib import Path
import sys

# Direct script execution from any working directory and python -m both work.
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.SteppingReflexGA.experiments.core.run import main

if __name__ == "__main__":
    main("angle_setpoint")
