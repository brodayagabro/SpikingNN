"""Compatibility name for explicit solution-pointer publication.

The former dated report-driven publisher modified archives and notebook outputs.
Use --solution and --pointer to publish exactly one checked solution. Existing
scientific files and population archives are never annotated or overwritten.
"""
from pathlib import Path
import sys

for root in Path(__file__).resolve().parents:
    if (root/'src/SpikingNN').is_dir():
        sys.path.insert(0, str(root))
        break
from scripts.AfferentBalance2026.run import main

if __name__ == '__main__':
    raise SystemExit(main(['publish', *sys.argv[1:]]))
