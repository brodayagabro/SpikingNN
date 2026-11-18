"""Prepare an explicitly mirrored warm start without modifying its source genome."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.AfferentBalance2026.experiments.core.genome import (
    GenomeConfig, SCHEMA, file_hash, load_genome, project_mirror, write_json,
)
from scripts.AfferentBalance2026.experiments.C_setpoint_balance.joint_study import JointGenome
from scripts.AfferentBalance2026.experiments.D_trajectory.controller import TrajectoryGenome, load_solution


def prepare(source, config, output, degree=3):
    """Project paired physical genes and polynomial parity into a new domain.

    Parameters
    ----------
    source : Path
        A self-contained controller, C polynomial, or D trajectory genome.
    config : Path
        Destination C/D bounds and episode protocol. Source mechanics must match.
    output : Path
        New directory for the projected genome, configuration and provenance.
    degree : {2, 3}
        C command degree. D always retains four ascending-power coefficients.
    """
    source, config, output = Path(source), Path(config), Path(output)
    data = json.loads(config.read_text())
    data['mirror_symmetry'] = True
    initial = project_mirror([field['value'] for field in data['parameters']])
    for field, value in zip(data['parameters'], initial):
        field['value'] = float(value)
    cfg = GenomeConfig(data)
    raw = json.loads(source.read_text())
    if raw['schema'] == SCHEMA:
        old, physical = load_genome(source)
        coefficients, tau = np.zeros((2, 4)), 0.
    else:
        old, physical, coefficients, tau, _ = load_solution(source)
    if not np.array_equal(old.mechanics, cfg.mechanics):
        raise ValueError('Source and destination mechanics differ')
    projected = project_mirror(physical)
    cfg.encode(projected)
    first = .5*(coefficients[0] + coefficients[1]*[1, -1, 1, -1])
    command = np.array([first, first*[1, -1, 1, -1]])
    if data['experiment'] == 'C_setpoint_balance':
        if degree == 2 and np.any(command[:, 3] != 0):
            raise ValueError('Cannot discard nonzero cubic coefficients; choose degree 3')
        codec = JointGenome(cfg, degree=degree)
        command = command[:, :degree+1]
    else:
        codec = TrajectoryGenome(cfg)
    unit = codec.encode(projected, command)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'genome_cfg.json', cfg.data)
    metadata = dict(selection='explicit reflection projection; not performance preserving',
                    source=str(source.resolve()), source_sha256=file_hash(source))
    if isinstance(codec, TrajectoryGenome):
        codec.save(output/'best_genome.json', unit, current_tau_s=tau, **metadata)
    else:
        codec.save(output/'best_genome.json', unit, **metadata)
    write_json(output/'projection.json', dict(**metadata, independent_coordinates=codec.size,
        controller_coordinates=cfg.size, physical_fields=len(codec.names),
        physical_change=(projected-physical).tolist(),
        polynomial_change=(np.pad(command, ((0, 0), (0, 4-command.shape[1])))-coefficients).tolist(),
        command_rule='I1(theta) = I0(-theta)', averaging='arithmetic in physical units',
        dt_ms=.1, current_tau_s=tau,
        muscle_time_bounds_ms={field['name']: [field['minimum'], field['maximum']]
                              for field in cfg.data['parameters'] if field['index'] in (19, 20, 26, 27)}))
    return output/'best_genome.json'


def main():
    """Create a reproducible symmetric seed for the normal warm-start runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--genome', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--degree', choices=[2, 3], default=3, type=int)
    args = parser.parse_args()
    print(prepare(args.genome, args.config, args.output, args.degree))


if __name__ == '__main__':
    main()
