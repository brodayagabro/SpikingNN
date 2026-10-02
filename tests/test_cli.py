"""Exercise real CLI parsing and exports with short simulations."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from SpikingNN import create_simulation, run_simulation
from SpikingNN.cli import main


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "system_type": "fullsys",
        "network": {
            "neuron_count": 2, "input_size": 2, "output_size": 2, "afferent_size": 6,
            "weights": [[0, 0], [0, 0]], "Q_app": [[1, 0], [0, 1]],
            "Q_aff": np.zeros((2, 6)).tolist(), "P": [[1, 0], [0, 1]],
        },
        "limbs": [{"name": "leg", "mechanics": {"q0": 1.2}}],
        "simulation": {
            "dt": 0.1, "duration": 2,
            "input_current": {"type": "sine", "amplitude": 5, "frequency": 10, "neurons": [1]},
        },
    }))
    return path


def test_cli_phase_override_preserves_config_signal_and_exports_limbs(config_path, tmp_path, monkeypatch):
    csv_path, npz_path = tmp_path / "data.csv", tmp_path / "data.npz"
    monkeypatch.setattr(sys, "argv", ["spikingnn", "sim", str(config_path), "--phase", "0.25",
                                     "-o", str(csv_path), str(npz_path)])
    main()
    expected = run_simulation(create_simulation(config_path), {
        "type": "sine", "amplitude": 5, "frequency": 10, "neurons": [1], "phase": 0.25,
    }).get_data()
    frame = pd.read_csv(csv_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["V"], expected["V"])
        for field, values in expected["limbs"]["leg"].items():
            prefix = f"limb_leg_{field}"
            np.testing.assert_allclose(archive[prefix], values)
            if values.ndim == 1:
                np.testing.assert_allclose(frame[prefix], values)
            else:
                columns = [f"{prefix}_{i}" for i in range(6)]
                np.testing.assert_allclose(frame[columns], values)
    np.testing.assert_allclose(frame[["V_0", "V_1"]], expected["V"])


def test_cli_without_overrides_uses_json_current(config_path, tmp_path, monkeypatch):
    output = tmp_path / "data.npz"
    monkeypatch.setattr(sys, "argv", ["spikingnn", "sim", str(config_path), "-o", str(output)])
    main()
    expected = run_simulation(create_simulation(config_path)).get_data()
    with np.load(output) as archive:
        np.testing.assert_allclose(archive["V"], expected["V"])


def test_cli_reports_export_failure(config_path, tmp_path, monkeypatch):
    output = tmp_path / "missing" / "data.npz"
    monkeypatch.setattr(sys, "argv", ["spikingnn", "sim", str(config_path), "-o", str(output)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
