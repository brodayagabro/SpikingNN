import numpy as np


def generate_signal(params: dict, duration: float, dt: float) -> np.ndarray:
    t = np.arange(0, duration, dt)
    signal_type = params.get("type", "constant")
    amplitude = params.get("amplitude", 0.0)
    frequency = params.get("frequency", 1.0)

    if signal_type == "constant":
        return np.full_like(t, amplitude)
    elif signal_type == "sine":
        return amplitude * np.sin(2 * np.pi * frequency * t)
    elif signal_type == "square":
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif signal_type == "ramp":
        return np.linspace(0, amplitude, len(t))
    elif signal_type == "noise":
        return np.random.normal(0, amplitude, len(t))
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
