"""Balance with restricted muscle gains, 0.02 <= w < 0.1.

Examples (from topubsrc):
    python B_balance/experiment.py check
    python B_balance/experiment.py train --smoke --output results/B_balance_smoke
    python B_balance/experiment.py simulate --genome B_balance/genomes/slow.json
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolution import cli
from model import balance_banks

HERE = Path(__file__).resolve().parent


def make_banks(config, kind='none'):
    """Fix independent initial-state banks; all command currents remain zero."""
    if kind != 'none':
        raise ValueError('Balance experiments do not have command genes')
    return balance_banks(config['simulation'])


if __name__ == '__main__':
    cli('B_balance', HERE, make_banks)
