"""
SpikingNN API Module

This module provides a function-based API for configuring and running
spiking neural network simulations. It supports:

- JSON configuration files with validation
- Multiple signal types (constant, sine, square, ramp, noise)
- Phase control for periodic signals
- Network-only (onlynet) and full neuromechanical (fullsys) systems
- Synchronous and asynchronous execution

Example:
    >>> from SpikingNN import create_simulation, run_simulation, get_results
    >>> sim = create_simulation("config.json")
    >>> signals = {"type": "sine", "amplitude": 10.0, "frequency": 1.0, "phase": 0.25}
    >>> results = run_simulation(sim, signals)
    >>> data = get_results(results)
"""

import asyncio
from copy import deepcopy
import json
from numbers import Integral, Real
import numpy as np
from tqdm.auto import tqdm
from .schema import validate_config, ConfigurationError, ValidationError, SimulationError
from .signals import generate_signal
from .core.Izh_net import Izhikevich_IO_Network, types2params
from .core.multi_limb import MultiLimbSystem
from .core.factories import AfferentedLimbFactory


class Simulation:
    """
    Represents a configured simulation.

    This class encapsulates the network and optional limb components
    for running spiking neural network simulations.

    Attributes:
        config (dict): Original configuration dictionary.
        system_type (str): Type of system - "onlynet" or "fullsys".
        network (Izhikevich_IO_Network): Neural network object.
        limbs (list): List of Afferented_Limb objects (for fullsys).
        system (MultiLimbSystem): Multi-limb system (for fullsys).

    Example:
        >>> sim = Simulation(config)
        >>> print(f"System type: {sim.system_type}")
        >>> print(f"Neuron count: {sim.network.N}")
    """

    def __init__(self, config: dict):
        """
        Initialize Simulation from configuration dictionary.

        Args:
            config (dict): Configuration dictionary with network and simulation parameters.

        Raises:
            ConfigurationError: If configuration is invalid or missing required fields.
        """
        validate_config(config)
        self.config = deepcopy(config)
        self.system_type = config.get("system_type", "onlynet")
        self.network = None
        self.limbs = []
        self.system = None
        self._setup_system()

    def _setup_system(self):
        net_config = self.config["network"]
        N = net_config["neuron_count"]
        
        # Build network parameters
        net_kwargs = {
            'N': N,
            'input_size': net_config.get("input_size", N),
            'output_size': net_config.get("output_size", N),
            'afferent_size': net_config.get("afferent_size", 0)
        }
        
        # Set neuron types if provided
        neuron_types = net_config.get("neuron_types")
        if neuron_types:
            # Convert neuron types to parameters
            a, b, c, d = types2params(neuron_types)
            net_kwargs['a'] = a
            net_kwargs['b'] = b
            net_kwargs['c'] = c
            net_kwargs['d'] = d

        # Explicit per-neuron values override the selected presets.
        for name, values in net_config.get("neuron_params", {}).items():
            net_kwargs[name] = np.asarray(values, dtype=float)
        
        # Set connectivity mask if provided
        connectivity = net_config.get("connectivity")
        if connectivity:
            M = np.zeros((N, N))
            for conn in connectivity:
                if len(conn) >= 3:
                    i, j, coef = conn[0], conn[1], conn[2]
                    if 0 <= i < N and 0 <= j < N:
                        M[j, i] = np.sign(coef)
            net_kwargs['M'] = M
        
        # Set weights if provided
        weights = net_config.get("weights")
        if weights:
            W = np.zeros((N, N))
            for i, row in enumerate(weights):
                for j, val in enumerate(row):
                    if 0 <= i < N and 0 <= j < N:
                        W[i, j] = val
            net_kwargs['W'] = W
        
        # Set synaptic time constants if provided
        tau_syn = net_config.get("tau_syn")
        if tau_syn:
            tau = np.zeros((N, N))
            for i, row in enumerate(tau_syn):
                for j, val in enumerate(row):
                    if 0 <= i < N and 0 <= j < N:
                        tau[i, j] = val
            net_kwargs['tau_syn'] = tau
        
        # Set Q_app if provided
        Q_app = net_config.get("Q_app")
        if Q_app:
            net_kwargs['Q_app'] = np.array(Q_app)
        
        # Set Q_aff if provided
        Q_aff = net_config.get("Q_aff")
        if Q_aff:
            net_kwargs['Q_aff'] = np.array(Q_aff)
        
        # Set P if provided
        P = net_config.get("P")
        if P:
            net_kwargs['P'] = np.array(P)
        
        # Create network
        self.network = Izhikevich_IO_Network(**net_kwargs)
        
        # Create limbs if fullsys
        if self.system_type == "fullsys":
            limbs_config = self.config.get("limbs", [])
            for limb_config in limbs_config:
                limb = AfferentedLimbFactory.create_from_dict(limb_config)
                self.limbs.append(limb)
            
            # Create MultiLimbSystem
            if self.limbs:
                limb_names = [limb_config.get("name", f"limb_{i}") 
                             for i, limb_config in enumerate(limbs_config)]
                self.system = MultiLimbSystem(
                    network=self.network,
                    limbs=self.limbs,
                    names=limb_names
                )


class Results:
    """
    Contains simulation results.

    This class wraps the simulation output data and provides
    methods to access the results.

    Attributes:
        data (dict): Dictionary containing simulation data with keys:
            - "time": NumPy array of time points (ms)
            - "V": NumPy array of membrane potentials
            - "U": NumPy array of recovery variables
            - "limbs": Named histories of mechanics and afferents (fullsys only)

    Example:
        >>> results = run_simulation(sim, signals)
        >>> data = get_results(results)
        >>> print(f"V shape: {data['V'].shape}")
    """

    def __init__(self, data: dict):
        """
        Initialize Results with simulation data.

        Args:
            data (dict): Dictionary with time, V, and U arrays.
        """
        self.data = data

    def get_data(self) -> dict:
        """
        Get the simulation data.

        Returns:
            dict: Dictionary with time, V, U and optional named limb histories.
        """
        return self.data


def create_simulation(config_path: str) -> Simulation:
    """
    Create a simulation from a JSON configuration file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Simulation object ready for execution
        
    Raises:
        FileNotFoundError: If configuration file does not exist
        ConfigurationError: If configuration is invalid
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
    
    return Simulation(config)


def _prepare_signal(signals: dict, input_size: int, duration: float, dt: float):
    """Validate input before stepping and generate one complete time series."""
    if not isinstance(signals, dict):
        raise ValidationError("Signals must be a dictionary", field="signals")
    if not signals:
        return None, []

    signal_type = signals.get("type")
    if signal_type not in ("constant", "sine", "square", "ramp", "noise"):
        raise ValidationError("A supported signal type is required", field="type")

    for field, default in (("amplitude", None), ("frequency", 1.0), ("phase", 0.0)):
        value = signals.get(field, default)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValidationError(f"{field} must be a finite number", field=field)
    if signals.get("frequency", 1.0) < 0:
        raise ValidationError("frequency must be non-negative", field="frequency")
    if signal_type == "noise" and signals["amplitude"] < 0:
        raise ValidationError("Noise amplitude must be non-negative", field="amplitude")

    indices = signals.get("neurons", list(range(input_size)))
    if isinstance(indices, Integral) and not isinstance(indices, bool):
        indices = [indices]
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise ValidationError("neurons must contain input channel indices", field="neurons")
    if isinstance(indices, np.ndarray) and indices.ndim != 1:
        raise ValidationError("neurons must be one-dimensional", field="neurons")
    for idx in indices:
        if isinstance(idx, bool) or not isinstance(idx, Integral) or not 0 <= idx < input_size:
            raise ValidationError(f"Input channel index {idx} is out of range [0, {input_size - 1}]", field="neurons")

    for field in ("amplitude_range", "frequency_range", "phase_range"):
        if field not in signals:
            continue
        bounds = signals[field]
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(isinstance(value, bool) or not isinstance(value, Real)
                       or not np.isfinite(value) for value in bounds)
                or bounds[0] > bounds[1]):
            raise ValidationError(f"{field} must be finite [min, max] with min <= max", field=field)

    return generate_signal(signals, duration=duration, dt=dt), list(indices)


def run_simulation(simulation: Simulation, signals: dict = None, progress: bool = False) -> Results:
    """Run from the current state, recording each state before its time step.

    With signals=None, use simulation.input_current from the configuration.
    An explicit empty dictionary selects zero external current. The historical
    'neurons' field selects input channels, which Q_app maps to actual neurons.
    Repeated calls continue the current state with a local time axis starting
    at zero; create a new Simulation to repeat the initial conditions.
    Set progress=True to show a tqdm progress bar counting completed steps.

    Returns time, V, U and, for fullsys, a 'limbs' dictionary of named histories
    (angle_q, angular_velocity_w, force_flex, force_ext and afferents).
    """
    sim_config = simulation.config["simulation"]
    dt = sim_config["dt"]
    duration = sim_config["duration"]
    t = np.arange(0, duration, dt)
    net = simulation.network
    signal_params = sim_config.get("input_current", {}) if signals is None else signals
    signal, indices = _prepare_signal(signal_params, net.input_size, duration, dt)

    results = {
        "time": t,
        "V": np.empty((len(t), net.N)),
        "U": np.empty((len(t), net.N)),
    }
    system = simulation.system
    if system is not None:
        results["limbs"] = {
            name: {
                "angle_q": np.empty(len(t)),
                "angular_velocity_w": np.empty(len(t)),
                "force_flex": np.empty(len(t)),
                "force_ext": np.empty(len(t)),
                "afferents": np.empty((len(t), 6)),
            }
            for name in system.limb_names
        }

    Iapp = np.zeros(net.input_size)
    Iaff = np.zeros(net.afferent_size)
    with tqdm(range(len(t)), disable=not progress, desc="Simulation", unit="step") as steps:
        for i in steps:
            try:
                results["V"][i] = net.V_prev
                results["U"][i] = net.U_prev
                if system is not None:
                    for name, limb in zip(system.limb_names, system.limbs):
                        history = results["limbs"][name]
                        history["angle_q"][i] = limb.q
                        history["angular_velocity_w"][i] = limb.w
                        history["force_flex"][i] = limb.F_flex
                        history["force_ext"][i] = limb.F_ext
                        history["afferents"][i] = limb.output

                if signal is not None:
                    Iapp[indices] = signal[i]
                if system is not None:
                    system.step(dt=dt, Iapp=Iapp)
                else:
                    net.step(dt=dt, Iapp=Iapp, Iaff=Iaff)
            except Exception as exc:
                raise SimulationError(f"Simulation failed at timestep {i}: {exc}", timestep=i) from exc

    return Results(results)


async def run_simulation_async(simulation: Simulation, signals: dict = None, progress: bool = False) -> Results:
    # run_in_executor keeps compatibility with the declared Python 3.8 minimum.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_simulation, simulation, signals, progress)


def get_results(results: Results) -> dict:
    return results.get_data()
