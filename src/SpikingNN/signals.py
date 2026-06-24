import numpy as np


def generate_signal(params: dict, duration: float, dt: float) -> np.ndarray:
    """
    Generate a signal based on parameters.
    
    Args:
        params: Dictionary with signal parameters:
            - type: Signal type (required): "constant", "sine", "square", "ramp", "noise"
            - amplitude: Signal amplitude in nA (required)
            - frequency: Signal frequency in Hz (for periodic signals)
        duration: Signal duration in ms
        dt: Time step in ms
        
    Returns:
        NumPy array with signal values
        
    Raises:
        ValueError: If required parameters are missing or signal type is unknown
    """
    # Validate required parameters
    if "type" not in params:
        raise ValueError("Signal type is required")
    if "amplitude" not in params:
        raise ValueError("Signal amplitude is required")
    
    t = np.arange(0, duration, dt)
    signal_type = params["type"]
    amplitude = params["amplitude"]
    frequency = params.get("frequency", 1.0)

    if signal_type == "constant":
        return np.full_like(t, amplitude)
    elif signal_type == "sine":
        # Convert t from ms to seconds for frequency in Hz
        t_seconds = t / 1000.0
        return amplitude * np.sin(2 * np.pi * frequency * t_seconds)
    elif signal_type == "square":
        # Convert t from ms to seconds for frequency in Hz
        t_seconds = t / 1000.0
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t_seconds))
    elif signal_type == "ramp":
        return np.linspace(0, amplitude, len(t))
    elif signal_type == "noise":
        return np.random.normal(0, amplitude, len(t))
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
