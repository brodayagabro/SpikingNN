"""Check notebook genome selection and recovery from cached pre-mirror codecs."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.AfferentBalance2026.experiments.notebook_runtime import (
    GENOME_VARIANTS, resolve_genome_path,
)

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('variant', GENOME_VARIANTS)
def test_named_variant_is_independent_of_working_directory(tmp_path, monkeypatch, variant):
    """Use the requested experiment's file even if cwd has a file of the same name."""
    experiment = tmp_path / 'D_trajectory'
    expected = experiment / GENOME_VARIANTS[variant]
    expected.parent.mkdir(parents=True)
    expected.write_text('{}')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    (elsewhere / expected.name).write_text('{"wrong": true}')
    monkeypatch.chdir(elsewhere)
    assert resolve_genome_path(experiment, variant=variant) == expected
    assert resolve_genome_path(experiment, expected) == expected
    assert resolve_genome_path(experiment, GENOME_VARIANTS[variant]) == expected


def test_missing_mirror_does_not_silently_load_another_genome(tmp_path):
    """Keep variant selection explicit rather than presenting an old winner as mirrored."""
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/best_genome.json').write_text('{}')
    with pytest.raises(FileNotFoundError, match='best_genome_slow_muscle_mirrow.json'):
        resolve_genome_path(tmp_path)
    with pytest.raises(ValueError, match='Unknown genome variant'):
        resolve_genome_path(tmp_path, variant='typo')


def test_notebook_refresh_recovers_cached_decoder_without_changing_genomes(tmp_path):
    """Reproduce obsolete coordinate counts in a separate interpreter, then reload.

    Isolation matters: refreshing class definitions in pytest's own interpreter
    would invalidate the deliberately cached imports of unrelated test modules.
    The fixtures are generated locally; archived training results are not needed.
    """
    script = r'''
from copy import deepcopy
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
from scripts.AfferentBalance2026.experiments.core import genome
from scripts.AfferentBalance2026.experiments.C_setpoint_balance import joint_study
from scripts.AfferentBalance2026.experiments.D_trajectory import controller

experiments = Path('scripts/AfferentBalance2026/experiments').resolve()
out = Path(sys.argv[1])
files = []
for name, codec_type in [('C_setpoint_balance', joint_study.JointGenome),
                         ('D_trajectory', controller.TrajectoryGenome)]:
    for mirror in (False, True):
        data = deepcopy(genome.GenomeConfig(experiments/name/'genome_cfg.json').data)
        data['mirror_symmetry'] = mirror
        if mirror:
            physical = genome.project_mirror([p['value'] for p in data['parameters']])
            for field, value in zip(data['parameters'], physical):
                field['value'] = float(value)
        codec = codec_type(genome.GenomeConfig(data))
        path = out/f'{name}_{mirror}.json'
        codec.save(path, np.full(codec.size, .5))
        files.append(path)
hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]

# Emulate the old in-memory codecs that ignore the mirror flag.
genome.GenomeConfig.size = property(lambda self: len(self.mutable))
controller.TrajectoryGenome.size = property(lambda self: self.controller.size + 8)
try:
    controller.load_solution(files[-1])
except ValueError as error:
    assert 'Expected 56 finite coordinates' in str(error), str(error)
else:
    raise AssertionError('The stale decoder must reproduce the reported failure')

# A genuinely old module has no new symbols. Executing archived source over a
# current module dictionary alone would incorrectly retain MIRROR_INDICES.
del genome.MIRROR_INDICES
sys.modules.pop('scripts.AfferentBalance2026.experiments.visualization', None)
try:
    importlib.import_module('scripts.AfferentBalance2026.experiments.visualization')
except ImportError as error:
    assert 'MIRROR_INDICES' in str(error), str(error)
else:
    raise AssertionError('Fresh visualization must fail against the obsolete genome module')

from scripts.AfferentBalance2026.experiments import notebook_runtime
importlib.reload(notebook_runtime)
notebook_runtime.refresh_model_modules()
for name in ('core.run', 'C_setpoint_balance.polynomial_study',
             'C_setpoint_balance.joint_study', 'C_setpoint_balance.run_experiment',
             'D_trajectory.controller', 'visualization'):
    module = importlib.import_module('scripts.AfferentBalance2026.experiments.' + name)
    assert module.GenomeConfig is genome.GenomeConfig, name
package = importlib.import_module('scripts.AfferentBalance2026.experiments.D_trajectory')
assert package.AngleController is controller.AngleController

for path in files:
    cfg, physical, coefficients, tau, _ = controller.load_solution(path)
    assert cfg.mirror_symmetry == ('True' in path.name)
    if cfg.mirror_symmetry:
        np.testing.assert_array_equal(physical, physical[genome.MIRROR_INDICES])
        np.testing.assert_array_equal(coefficients[1], coefficients[0]*[1, -1, 1, -1])
    state = controller.AngleController(path).advance(0., .002)
    assert state['time_s'] == .002 and state['failure'] == 'none'

identities = (genome.GenomeConfig, joint_study.JointGenome, controller.AngleController)
importlib.reload(notebook_runtime)
notebook_runtime.refresh_model_modules()
assert identities == (genome.GenomeConfig, joint_study.JointGenome, controller.AngleController)
assert hashes == [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
