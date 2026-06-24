import json
import tempfile
import pytest
from SpikingNN.api import create_simulation, run_simulation, get_results


def test_create_simulation():
    config = {
        "system_type": "onlynet",
        "network": {"neuron_count": 5, "neuron_types": ["RS"] * 5},
        "simulation": {"dt": 0.1, "duration": 100.0}
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
    sim = create_simulation(f.name)
    assert sim is not None


def test_create_simulation_invalid_config():
    config = {"system_type": "onlynet", "network": {"neuron_count": 5}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
    with pytest.raises(Exception):
        create_simulation(f.name)


def test_run_simulation():
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_count": 5, 
            "neuron_types": ["RS"] * 5,
            "input_size": 5,
            "output_size": 5,
            "afferent_size": 0
        },
        "simulation": {"dt": 0.1, "duration": 10.0}
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
    sim = create_simulation(f.name)
    results = run_simulation(sim, {})
    assert results is not None
    data = get_results(results)
    assert "time" in data
    assert "V" in data
    assert "U" in data
    assert len(data["time"]) > 0
