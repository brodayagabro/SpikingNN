"""Compatibility launcher for finetuning; source domains remain explicit."""
from pathlib import Path
import sys
for root in Path(__file__).resolve().parents:
    if (root/'src/SpikingNN').is_dir():
        sys.path.insert(0, str(root))
        break
from scripts.AfferentBalance2026.run import legacy_main

if __name__ == '__main__':
    raise SystemExit(legacy_main(warm=True))
