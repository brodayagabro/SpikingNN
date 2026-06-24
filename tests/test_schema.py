"""
Tests for SpikingNN schema validation.
"""

import pytest
from SpikingNN.schema import validate_config, ConfigurationError


def test_valid_config():
    """Test that a valid configuration passes validation."""
    config = {
        "system_type": "onlynet",
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


def test_valid_config_with_input_current():
    """Test that a valid configuration with input_current passes validation."""
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_count": 5,
            "neuron_types": ["RS"] * 5
        },
        "simulation": {
            "dt": 0.1,
            "duration": 100.0,
            "input_current": {
                "type": "constant",
                "amplitude": 10.0,
                "neurons": [0, 1]
            }
        }
    }
    validate_config(config)  # Should not raise


def test_invalid_config_missing_network():
    """Test that missing network section raises error."""
    config = {
        "system_type": "onlynet",
        "simulation": {
            "dt": 0.1,
            "duration": 100.0
        }
    }
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "'network' is a required property" in str(exc_info.value)


def test_invalid_config_missing_simulation():
    """Test that missing simulation section raises error."""
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_count": 5
        }
    }
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "'simulation' is a required property" in str(exc_info.value)


def test_invalid_config_missing_neuron_count():
    """Test that missing neuron_count raises error."""
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_types": ["RS"]
        },
        "simulation": {
            "dt": 0.1,
            "duration": 100.0
        }
    }
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "'neuron_count' is a required property" in str(exc_info.value)


def test_invalid_config_wrong_type_neuron_count():
    """Test that wrong type for neuron_count raises error."""
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_count": "ten"
        },
        "simulation": {
            "dt": 0.1,
            "duration": 100.0
        }
    }
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "'ten' is not of type 'integer'" in str(exc_info.value)


def test_invalid_config_zero_neuron_count():
    """Test that zero neuron_count raises error."""
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_count": 0
        },
        "simulation": {
            "dt": 0.1,
            "duration": 100.0
        }
    }
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "0 is less than the minimum of 1" in str(exc_info.value)


def test_invalid_config_negative_dt():
    """Test that negative dt raises error."""
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_count": 5
        },
        "simulation": {
            "dt": -0.1,
            "duration": 100.0
        }
    }
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "-0.1 is less than the minimum of 0.001" in str(exc_info.value)


def test_invalid_config_zero_duration():
    """Test that zero duration raises error."""
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_count": 5
        },
        "simulation": {
            "dt": 0.1,
            "duration": 0
        }
    }
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "0 is less than or equal to the minimum of 0" in str(exc_info.value)


def test_invalid_config_unknown_key():
    """Test that unknown keys raise error."""
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_count": 5,
            "TYPO_KEY": True
        },
        "simulation": {
            "dt": 0.1,
            "duration": 100.0
        }
    }
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "Additional properties are not allowed" in str(exc_info.value)


def test_invalid_config_wrong_array_shape():
    """Test that wrong array shape raises error."""
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_count": 5,
            "connectivity": [1, 2, 3]  # Should be array of arrays
        },
        "simulation": {
            "dt": 0.1,
            "duration": 100.0
        }
    }
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "3 is not of type 'array'" in str(exc_info.value)


def test_invalid_config_empty():
    """Test that empty config raises error."""
    config = {}
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "'system_type' is a required property" in str(exc_info.value)