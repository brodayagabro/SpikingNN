import numpy as np
from SpikingNN.signals import generate_signal

def test_constant_signal():
    params = {"type": "constant", "amplitude": 10.0}
    signal = generate_signal(params, duration=100.0, dt=0.1)
    assert signal.shape == (1000,)
    assert np.all(signal == 10.0)

def test_sine_signal():
    params = {"type": "sine", "amplitude": 5.0, "frequency": 1.0}
    signal = generate_signal(params, duration=100.0, dt=0.01)
    assert signal.shape == (10000,)
    assert np.isclose(np.max(signal), 5.0, atol=0.01)

def test_square_signal():
    params = {"type": "square", "amplitude": 3.0, "frequency": 1.0}
    signal = generate_signal(params, duration=10.0, dt=0.1)
    assert signal.shape == (100,)
    assert set(np.unique(signal)).issubset({-3.0, 0.0, 3.0})

def test_ramp_signal():
    params = {"type": "ramp", "amplitude": 10.0}
    signal = generate_signal(params, duration=10.0, dt=1.0)
    assert signal.shape == (10,)
    assert signal[0] == 0.0
    assert np.isclose(signal[-1], 10.0)

def test_noise_signal():
    params = {"type": "noise", "amplitude": 1.0}
    signal = generate_signal(params, duration=10.0, dt=0.1)
    assert signal.shape == (100,)
    assert np.std(signal) < 2.0

def test_unknown_type_raises():
    params = {"type": "invalid"}
    try:
        generate_signal(params, duration=1.0, dt=0.1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown signal type" in str(e)
