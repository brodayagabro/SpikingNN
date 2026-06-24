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
import json
import numpy as np
from .schema import validate_config, ConfigurationError, ValidationError, SimulationError
from .signals import generate_signal
from .core.Izh_net import Izhikevich_IO_Network, types2params, Afferented_Limb
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
            ValueError: If configuration is invalid or missing required fields.
        """
        self.config = config
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
            dict: Dictionary with keys "time", "V", "U".
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
    
    validate_config(config)
    return Simulation(config)


def run_simulation(simulation: Simulation, signals: dict) -> Results:
    """
    Run a simulation with the given signals.
    
    Args:
        simulation: Simulation object created by create_simulation()
        signals: Dictionary of signal parameters with the following format:
            {
                "type": "constant",  # Signal type (required)
                "amplitude": 10.0,   # Signal amplitude in nA (required)
                "frequency": 1.0,    # Frequency in Hz (for periodic signals)
                "neurons": [0, 1],   # Neuron indices to apply signal (optional)
                "amplitude_range": [min, max],  # Range for parameter sweep (optional)
                "frequency_range": [min, max]   # Range for parameter sweep (optional)
            }
            
    Returns:
        Results object containing simulation data
        
    Raises:
        ValidationError: If signal parameters are invalid
        SimulationError: If simulation encounters a runtime error
    """
    sim_config = simulation.config["simulation"]
    dt = sim_config["dt"]
    duration = sim_config["duration"]
    t = np.arange(0, duration, dt)

    results = {"time": t, "V": [], "U": []}

    N = simulation.network.N
    input_size = simulation.network.input_size
    
    # Validate signal parameters if provided
    if signals:
        signal_type = signals.get("type")
        amplitude = signals.get("amplitude")
        
        if signal_type and amplitude is None:
            raise ValidationError("Signal amplitude is required when signal type is specified", field="amplitude")
        if amplitude is not None and signal_type is None:
            raise ValidationError("Signal type is required when amplitude is specified", field="type")
        
        if signal_type and amplitude is not None:
            # Validate neuron indices
            neuron_indices = signals.get("neurons", list(range(input_size)))
            if isinstance(neuron_indices, int):
                neuron_indices = [neuron_indices]
            
            for idx in neuron_indices:
                if idx < 0 or idx >= input_size:
                    raise ValidationError(f"Neuron index {idx} is out of range [0, {input_size-1}]", field="neurons")
            
            # Validate parameter ranges
            amplitude_range = signals.get("amplitude_range")
            if amplitude_range:
                if len(amplitude_range) != 2 or amplitude_range[0] > amplitude_range[1]:
                    raise ValidationError("amplitude_range must be [min, max] with min <= max", field="amplitude_range")
            
            frequency_range = signals.get("frequency_range")
            if frequency_range:
                if len(frequency_range) != 2 or frequency_range[0] > frequency_range[1]:
                    raise ValidationError("frequency_range must be [min, max] with min <= max", field="frequency_range")

    try:
        for i in range(len(t)):
            # Store current state
            if simulation.system_type == "fullsys" and simulation.system:
                # For fullsys, store limb states
                state = simulation.system.get_state()
                v = state["neurons_V"]
                u = state["neurons_U"]
            else:
                # For onlynet, store network states
                v = simulation.network.V_prev.copy()
                u = simulation.network.U_prev.copy()
            
            # Ensure V_prev and U_prev are 1D arrays of length N
            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    v = np.full(N, v.item())
                elif v.shape != (N,):
                    v = v.flatten()[:N]
            else:
                v = np.array(v)
            
            if isinstance(u, np.ndarray):
                if u.ndim == 0:
                    u = np.full(N, u.item())
                elif u.shape != (N,):
                    u = u.flatten()[:N]
            else:
                u = np.array(u)
            
            results["V"].append(v)
            results["U"].append(u)
            
            # For IO_Network, Iapp must be a vector of length input_size
            Iapp = np.zeros(input_size)
            
            # Apply signal to specified neurons at current timestep
            if signals:
                signal_type = signals.get("type")
                amplitude = signals.get("amplitude")
                
                if signal_type and amplitude is not None:
                    neuron_indices = signals.get("neurons", list(range(input_size)))
                    if isinstance(neuron_indices, int):
                        neuron_indices = [neuron_indices]
                    
                    # Get signal value at current time
                    signal = generate_signal(signals, duration=duration, dt=dt)
                    signal_value = signal[i] if i < len(signal) else 0.0
                    
                    for neuron_idx in neuron_indices:
                        if neuron_idx < input_size:
                            Iapp[neuron_idx] = signal_value
            
                # Step the system
                if simulation.system_type == "fullsys" and simulation.system:
                    simulation.system.step(dt=dt, Iapp=Iapp)
                else:
                    # For IO_Network, Iaff must be a vector of length afferent_size
                    afferent_size = simulation.network.afferent_size
                    Iaff = np.zeros(afferent_size)
                    simulation.network.step(dt=dt, Iapp=Iapp, Iaff=Iaff)
    except Exception as e:
        raise SimulationError(f"Simulation failed at timestep {i}: {e}", timestep=i)

    results["V"] = np.array(results["V"])
    results["U"] = np.array(results["U"])

    return Results(results)


async def run_simulation_async(simulation: Simulation, signals: dict) -> Results:
    return await asyncio.to_thread(run_simulation, simulation, signals)


def get_results(results: Results) -> dict:
    return results.get_data()
