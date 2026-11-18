"""Check that the diagnostic rescaling preserves physics and respects B bounds."""
from pathlib import Path
import json

import numpy as np
import pytest

from scripts.AfferentBalance2026.experiments.core.genome import GenomeConfig
from scripts.AfferentBalance2026.experiments.simulation import case_grid, simulate
from scripts.AfferentBalance2026.tools.diagnose_balance_b import rescale_muscle_drive


def diagnosis_config():
    """Build the specified 0.02 domain independently of an edited live B config."""
    root = Path(__file__).resolve().parents[1]/'experiments'
    data = json.loads((root/'A_balance/genome_cfg.json').read_text())
    data['experiment'] = 'B_balance'
    for i in (16,23):
        data['parameters'][i].update(value=.01, maximum=.02)
    return GenomeConfig(data)


def test_rescaled_weights_preserve_noncalcium_dynamics():
    """Compare every state channel after an exactly representable calcium scaling."""
    config = diagnosis_config()
    original = config.initial_physical.copy()
    original[[16, 23]] = .08
    original[[22, 29]] = .8
    transformed = rescale_muscle_drive(original, config)
    config.encode(transformed)
    unchanged = np.setdiff1d(np.arange(52), [16,22,23,29])
    np.testing.assert_array_equal(original[unchanged], transformed[unchanged])
    case = case_grid([20.], [.3])[0]
    _, before = simulate(original, case, config.mechanics, duration_s=.15)
    _, after = simulate(transformed, case, config.mechanics, duration_s=.15)
    keep = np.setdiff1d(np.arange(26), [9,10])
    np.testing.assert_allclose(before[:, keep], after[:, keep], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(before[:, 9:11]*.25, after[:, 9:11], rtol=1e-10, atol=1e-10)


def test_rescaling_rejects_threshold_outside_target_domain():
    """Do not label a mapped genome B-admissible when its required k is too small."""
    config = diagnosis_config()
    original = config.initial_physical.copy()
    original[16], original[22] = .8, .1
    with pytest.raises(ValueError, match='bounds'):
        rescale_muscle_drive(original, config)
