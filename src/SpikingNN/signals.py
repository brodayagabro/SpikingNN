"""
Signal Generation Module

This module provides functions for generating various signal types
for use as input currents in spiking neural network simulations.

Supported signal types:
- constant: Constant current
- sine: Sinusoidal current with phase control
- square: Square wave with phase control
- ramp: Linear ramp from 0 to amplitude
- noise: Gaussian noise

Example:
    >>> from SpikingNN.signals import generate_signal
    >>> params = {"type": "sine", "amplitude": 10.0, "frequency": 1.0, "phase": 0.25}
    >>> signal = generate_signal(params, duration=1000.0, dt=0.1)
"""

import numpy as np


def generate_signal(params: dict, duration: float, dt: float) -> np.ndarray:
    """
    Generate a signal based on parameters.
    
    Args:
        params: Dictionary with signal parameters:
            - type: Signal type (required): "constant", "sine", "square", "ramp", "noise"
            - amplitude: Signal amplitude in nA (required)
            - frequency: Signal frequency in Hz (for periodic signals)
            - phase: Phase shift in radians (for periodic signals, default: 0)
                     Or phase as a fraction of period (0 to 1)
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
    phase = params.get("phase", 0.0)
    
    # Convert phase from fraction (0-1) to radians if needed
    # If phase is between 0 and 1, treat it as fraction of period
    if isinstance(phase, (int, float)) and 0 <= phase <= 1:
        phase_rad = phase * 2 * np.pi
    else:
        phase_rad = phase

    if signal_type == "constant":
        return np.full_like(t, amplitude)
    elif signal_type == "sine":
        # Convert t from ms to seconds for frequency in Hz
        t_seconds = t / 1000.0
        return amplitude * np.sin(2 * np.pi * frequency * t_seconds + phase_rad)
    elif signal_type == "square":
        # Convert t from ms to seconds for frequency in Hz
        t_seconds = t / 1000.0
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t_seconds + phase_rad))
    elif signal_type == "ramp":
        return np.linspace(0, amplitude, len(t))
    elif signal_type == "noise":
        return np.random.normal(0, amplitude, len(t))
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
