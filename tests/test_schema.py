import pytest
from SpikingNN.schema import validate_config, ConfigurationError

def test_valid_config():
    config = {
        "network": {
            "neuron_count": 10,
            "neuron_types": ["RS", "FS"],
            "connectivity": [[0, 1, 1.0]],
            "weights": [[0.0, 1.0], [0.5, 0.0]],
            "tau_syn": [[10.0, 10.0], [10.0, 10.0]]
        },
        "simulation": {
            "dt": 0.1,
            "duration": 1000.0
        }
    }
    validate_config(config)  # Should not raise

def test_invalid_config():
    config = {"invalid": "config"}
    with pytest.raises(ConfigurationError):
        validate_config(config)
