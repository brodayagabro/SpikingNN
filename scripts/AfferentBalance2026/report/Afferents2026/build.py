#!/usr/bin/env python3
"""Build genome appendices and compile the manuscript; --figures rebuilds plots."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', action='store_true', help='Replay prescribed diagnostic cases and rebuild five figures; no training')
    args = parser.parse_args()
    if args.figures:
        subprocess.run([sys.executable, str(HERE/'prepare_fresh_results.py')], cwd=HERE, check=True)
        subprocess.run([sys.executable, str(HERE/'build_figures.py')], cwd=HERE, check=True)
    subprocess.run([sys.executable, str(HERE/'build_appendices.py')], cwd=HERE, check=True)
    (HERE/'build').mkdir(exist_ok=True)
    latex = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
             '-file-line-error', '-output-directory=build', '-jobname=Afferents2026', 'main.tex']
    commands = [latex, ['bibtex', 'build/Afferents2026'], latex, latex]
    for index, command in enumerate(commands, 1):
        result = subprocess.run(command, cwd=HERE, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (HERE/'build'/f'pass_{index}.log').write_text(result.stdout)
        if result.returncode:
            print(result.stdout[-9000:])
            raise SystemExit(result.returncode)
    shutil.copy2(HERE/'build/Afferents2026.pdf', HERE/'Afferents2026.pdf')
    print(HERE/'Afferents2026.pdf')


if __name__ == '__main__':
    main()
