"""Check prescribed article banks and truthful rendering of truncated trajectories."""
import numpy as np
import pytest

from scripts.AfferentBalance2026.report.article_figure_data import (
    SIGNAL_INDEX, case_banks, display_indices,
)


def test_phase_grid_and_zero_speed_bank_have_distinct_physical_conditions():
    """Cover the full 10×10 domain and ten extra zero-speed starts without impulses."""
    phase, zero = case_banks(22.5)
    assert phase.shape == (100, 5) and zero.shape == (10, 5)
    np.testing.assert_allclose(np.rad2deg(np.unique(phase[:, 1])), np.linspace(-50, 50, 10))
    np.testing.assert_allclose(np.unique(phase[:, 2]), np.linspace(-1, 1, 10))
    assert len(np.unique(phase, axis=0)) == 100
    assert np.all(phase[:, 2] != 0) and np.all(zero[:, 2] == 0)
    assert np.all(phase[:, 3] == -1) and np.all(phase[:, 4] == 0)
    np.testing.assert_array_equal(phase[:, 0], np.full(100, np.deg2rad(22.5)))
    # Target commands do not shift the actual initial-angle range.
    np.testing.assert_array_equal(zero[:, 1], case_banks(-45)[1][:, 1])
    np.testing.assert_allclose(np.rad2deg(zero[SIGNAL_INDEX, 1]), -27.77777777777778)


@pytest.mark.parametrize('length', [1, 2, 10, 11, 3918, 100001])
def test_display_decimation_keeps_terminal_fall_or_episode_sample(length):
    """Never visually extend or drop the terminal state of a short/falling trace."""
    indices = display_indices(length)
    assert indices[0] == 0 and indices[-1] == length - 1
    assert np.all(np.diff(indices) > 0)
    assert np.all(indices < length)


@pytest.mark.parametrize('target', [float('nan'), -90, 90])
def test_invalid_target_is_rejected(target):
    """Reject targets outside the finite upright-angle domain before integration."""
    with pytest.raises(ValueError):
        case_banks(target)
