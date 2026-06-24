import numpy as np
from SpikingNN.signals import generate_signal

def test_constant_signal():
    params = {"type": "constant", "amplitude": 10.0}
    signal = generate_signal(params, duration=100.0, dt=0.1)
    assert signal.shape == (1000,)
    assert np.all(signal == 10.0)

def test_sine_signal():
    params = {"type": "sine", "amplitude": 5.0, "frequency": 1.0}
    signal = generate_signal(params, duration=2000.0, dt=0.01)
    assert signal.shape == (200000,)
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
    np.random.seed(42)  # Seed for reproducibility
    params = {"type": "noise", "amplitude": 1.0}
    signal = generate_signal(params, duration=10.0, dt=0.1)
    assert signal.shape == (100,)
    assert np.std(signal) < 2.0

def test_unknown_type_raises():
    params = {"type": "invalid", "amplitude": 1.0}
    try:
        generate_signal(params, duration=1.0, dt=0.1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown signal type" in str(e)


def test_missing_type_raises():
    params = {"amplitude": 1.0}
    try:
        generate_signal(params, duration=1.0, dt=0.1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Signal type is required" in str(e)


def test_missing_amplitude_raises():
    params = {"type": "constant"}
    try:
        generate_signal(params, duration=1.0, dt=0.1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Signal amplitude is required" in str(e)


def test_sine_frequency_correctness():
    """Test that sine wave has correct frequency (10 Hz = 0.1s period)."""
    params = {"type": "sine", "amplitude": 1.0, "frequency": 10.0}
    signal = generate_signal(params, duration=1.0, dt=0.001)
    # Find zero crossings
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    # Period should be 0.1 seconds = 100 samples at dt=0.001
    if len(zero_crossings) >= 2:
        period_samples = zero_crossings[1] - zero_crossings[0]
        period_seconds = period_samples * 0.001
        assert np.isclose(period_seconds, 0.1, atol=0.01)
