# SpikingNN API Documentation

This document provides detailed documentation for the SpikingNN Python API.

## Overview

The SpikingNN API provides a function-based interface for configuring and running spiking neural network simulations. It supports:

- JSON configuration files
- Multiple signal types with phase control
- Network-only (`onlynet`) and full neuromechanical (`fullsys`) systems
- Synchronous and asynchronous execution

## Quick Start

```python
from SpikingNN import create_simulation, run_simulation, get_results

# Create simulation from config
sim = create_simulation("config.json")

# Run with signals
signals = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 0.25
}

results = run_simulation(sim, signals)
data = get_results(results)
```

## Functions

### `create_simulation(config_path: str) -> Simulation`

Creates a simulation object from a JSON configuration file.

**Parameters:**
- `config_path` (str): Path to JSON configuration file

**Returns:**
- `Simulation`: Simulation object ready for execution

**Raises:**
- `FileNotFoundError`: If configuration file does not exist
- `ConfigurationError`: If configuration is invalid

**Example:**
```python
sim = create_simulation("my_config.json")
print(f"System type: {sim.system_type}")
print(f"Neuron count: {sim.network.N}")
```

### `run_simulation(simulation: Simulation, signals: dict) -> Results`

Runs a simulation with the given signals.

**Parameters:**
- `simulation` (Simulation): Simulation object created by `create_simulation()`
- `signals` (dict): Signal parameters dictionary

**Returns:**
- `Results`: Object containing simulation data

**Raises:**
- `ValidationError`: If signal parameters are invalid
- `SimulationError`: If simulation encounters a runtime error

**Example:**
```python
signals = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 0.5,
    "neurons": [0, 1]
}

results = run_simulation(sim, signals)
```

### `run_simulation_async(simulation: Simulation, signals: dict) -> Results`

Async version of `run_simulation()` for non-blocking execution.

**Parameters:**
- `simulation` (Simulation): Simulation object
- `signals` (dict): Signal parameters

**Returns:**
- `Results`: Coroutine that resolves to Results object

**Example:**
```python
import asyncio
from SpikingNN import create_simulation, run_simulation_async, get_results

async def main():
    sim = create_simulation("config.json")
    results = await run_simulation_async(sim, signals)
    data = get_results(results)
    return data

data = asyncio.run(main())
```

### `get_results(results: Results) -> dict`

Extracts raw data from Results object.

**Parameters:**
- `results` (Results): Results object from simulation

**Returns:**
- `dict`: Dictionary with keys:
  - `"time"`: NumPy array of time points (ms)
  - `"V"`: NumPy array of membrane potentials (shape: `[time_steps, neurons]`)
  - `"U"`: NumPy array of recovery variables (shape: `[time_steps, neurons]`)

**Example:**
```python
data = get_results(results)
print(f"Time range: {data['time'][0]} - {data['time'][-1]} ms")
print(f"V shape: {data['V'].shape}")
```

## Classes

### `Simulation`

Represents a configured simulation.

**Attributes:**
- `config` (dict): Original configuration dictionary
- `system_type` (str): `"onlynet"` or `"fullsys"`
- `network` (Izhikevich_IO_Network): Neural network object
- `limbs` (list): List of Afferented_Limb objects (for fullsys)
- `system` (MultiLimbSystem): Multi-limb system (for fullsys)

### `Results`

Contains simulation results.

**Methods:**
- `get_data() -> dict`: Returns dictionary with simulation data

## Signal Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | str | Signal type: `"constant"`, `"sine"`, `"square"`, `"ramp"`, `"noise"` |
| `amplitude` | float | Signal amplitude in nA |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frequency` | float | 1.0 | Frequency in Hz (for periodic signals) |
| `phase` | float | 0.0 | Phase shift (radians or fraction 0-1) |
| `neurons` | list | All input neurons | Neuron indices to apply signal |

### Parameter Ranges

For parameter sweeps, you can specify ranges:

| Parameter | Type | Description |
|-----------|------|-------------|
| `amplitude_range` | [min, max] | Range for amplitude sweep |
| `frequency_range` | [min, max] | Range for frequency sweep |
| `phase_range` | [min, max] | Range for phase sweep |

## Signal Types

### Constant

```python
signals = {
    "type": "constant",
    "amplitude": 10.0,
    "neurons": [0, 1]
}
```

### Sine

```python
signals = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,  # Hz
    "phase": 0.25,     # 25% of period
    "neurons": [0]
}
```

### Square

```python
signals = {
    "type": "square",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 0.0,
    "neurons": [0]
}
```

### Ramp

```python
signals = {
    "type": "ramp",
    "amplitude": 10.0,
    "neurons": [0]
}
```

### Noise

```python
signals = {
    "type": "noise",
    "amplitude": 1.0,  # Standard deviation
    "neurons": [0]
}
```

## Phase Control

Phase can be specified in two ways:

1. **As fraction of period (0 to 1):**
   ```python
   "phase": 0.25  # 25% of period = 90 degrees
   ```

2. **In radians:**
   ```python
   "phase": 1.5708  # pi/2 radians = 90 degrees
   ```

### Multi-Channel Phase Examples

For locomotor CPGs with flexor/extensor alternation:

```python
# Flexor channel (phase 0)
signals_flex = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 0.0,
    "neurons": [0]
}

# Extensor channel (phase 0.5 = 180 degrees)
signals_ext = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 0.5,
    "neurons": [1]
}
```

## Error Handling

### ConfigurationError

Raised when configuration file is invalid.

```python
from SpikingNN import ConfigurationError

try:
    sim = create_simulation("invalid.json")
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Path: {e.path}")
```

### ValidationError

Raised when signal parameters are invalid.

```python
from SpikingNN import ValidationError

try:
    results = run_simulation(sim, invalid_signals)
except ValidationError as e:
    print(f"Validation error: {e.message}")
    print(f"Field: {e.field}")
```

### SimulationError

Raised when simulation encounters a runtime error.

```python
from SpikingNN import SimulationError

try:
    results = run_simulation(sim, signals)
except SimulationError as e:
    print(f"Simulation error: {e.message}")
    print(f"Timestep: {e.timestep}")
```

## Complete Example

```python
import json
import tempfile
from SpikingNN import create_simulation, run_simulation, get_results

# Create configuration
config = {
    "system_type": "onlynet",
    "network": {
        "neuron_count": 5,
        "neuron_types": ["RS", "FS", "IB", "CH", "RS"],
        "input_size": 2,
        "output_size": 5,
        "afferent_size": 0
    },
    "simulation": {
        "dt": 0.1,
        "duration": 1000.0
    }
}

# Save to temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(config, f)
    config_path = f.name

# Create and run simulation
sim = create_simulation(config_path)

signals = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 0.25,
    "neurons": [0, 1]
}

results = run_simulation(sim, signals)
data = get_results(results)

# Analyze results
print(f"Simulation completed!")
print(f"Time steps: {len(data['time'])}")
print(f"V range: [{data['V'].min():.2f}, {data['V'].max():.2f}]")
print(f"U range: [{data['U'].min():.2f}, {data['U'].max():.2f}]")
```