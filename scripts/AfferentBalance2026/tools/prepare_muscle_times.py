"""Prepare reproducible warm starts under revised muscle-time bounds.

The destination configuration is explicit: saved genomes retain their original
bounds and are never silently reinterpreted. Four seeds cover direct clipping,
configured reference times, and approximate steady-gain compensation for each.
"""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.AfferentBalance2026.experiments.core.genome import GenomeConfig, file_hash, load_genome, write_json
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import JointGenome, load_joint_genome
from scripts.AfferentBalance2026.experiments.D_trajectory.controller import TrajectoryGenome, load_solution
from scripts.AfferentBalance2026.experiments.simulation import MODEL


def prepare(source, configuration, output):
    """Write four admissible seeds and preserve the source genome unchanged.

    Parameters
    ----------
    source : pathlib.Path
        Self-contained A/B/C/D genome from a previous study.
    configuration : pathlib.Path
        Destination genome configuration, including new time bounds.
    output : pathlib.Path
        New directory; existing artifacts are never overwritten.

    Returns
    -------
    dict
        Mapping from initialization names to their saved genome paths.
    """
    cfg = GenomeConfig(configuration)
    kind = cfg.data['experiment'][0]
    if kind in 'AB':
        _, old = load_genome(source)
        codec, coefficients, tau = cfg, None, 0.
    elif kind == 'C':
        previous, unit = load_joint_genome(source)
        old, coefficients = previous.decode(unit)
        codec, tau = JointGenome(cfg, previous.kind, previous.degree), 0.
    else:
        _, old, coefficients, tau, _ = load_solution(source)
        codec = TrajectoryGenome(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, output/'previous_best.json')
    write_json(output/'destination_config.json', cfg.data)
    variants = {}
    for reference in (False, True):
        for compensate in (False, True):
            physical = old.copy()
            for muscle, start in enumerate((16, 23)):
                for index in (start+3, start+4):
                    field = cfg.data['parameters'][index]
                    physical[index] = (field['value'] if reference else
                                       np.clip(old[index], field['minimum'], field['maximum']))
                if compensate:
                    # Match steady calcium drive; P carries any correction that
                    # the experiment's neuromuscular weight bounds cannot admit.
                    weight = old[start]*old[start+3]/physical[start+3]
                    field = cfg.data['parameters'][start]
                    physical[start] = np.clip(weight, field['minimum'], field['maximum'])
                    physical[48+2*muscle:50+2*muscle] *= weight/physical[start]
                    field = cfg.data['parameters'][start+1]
                    physical[start+1] = np.clip(old[start+1]*old[start+4]/physical[start+4],
                                              field['minimum'], field['maximum'])
            name = ('reference' if reference else 'clipped')+('_gain_matched' if compensate else '')
            path = output/f'{name}.json'
            metadata = dict(model=MODEL, initialization=name, source_sha256=file_hash(source),
                            note='Steady-gain matching does not preserve transient dynamics.',
                            changed_fields=[dict(index=int(i), before=float(old[i]), after=float(physical[i]))
                                            for i in np.flatnonzero(physical != old)])
            if kind in 'AB':
                codec.save(path, physical, **metadata)
            elif kind == 'C':
                codec.save(path, codec.encode(physical, coefficients), **metadata)
            else:
                codec.save(path, codec.encode(physical, coefficients), current_tau_s=tau, **metadata)
            variants[name] = str(path)
    write_json(output/'preparation.json', dict(source_sha256=file_hash(source),
               configuration_sha256=cfg.signature, variants=variants))
    return variants


def main():
    """Parse explicit source, destination configuration and new output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--genome', type=Path, required=True, help='Previous self-contained genome.')
    parser.add_argument('--config', type=Path, required=True, help='Destination bounds and reference times.')
    parser.add_argument('--output', type=Path, required=True, help='New directory for the four seeds.')
    args = parser.parse_args()
    for name, path in prepare(args.genome, args.config, args.output).items():
        print(f'{name}: {path}')


if __name__ == '__main__':
    main()
