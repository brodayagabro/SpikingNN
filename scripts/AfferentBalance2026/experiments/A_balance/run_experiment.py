"""Run one GA training for A_balance; results stay in the experiment directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.AfferentBalance2026.experiments.core.run import balance_main


if __name__ == "__main__":
    balance_main(Path(__file__).with_name("genome_cfg.json"), sweep_mode=False)
