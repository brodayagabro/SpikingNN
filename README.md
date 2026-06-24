# SpikingNN

This project provides a comprehensive computational framework for modeling and simulating Spiking Neural Networks (SNNs) based on the Izhikevich neuron model. The core library implements biologically plausible neuron dynamics with support for multiple neuron types (RS, FS, IB, CH, etc.), configurable synaptic connectivity, and synaptic current relaxation dynamics. Beyond pure neural simulation, the framework includes neuromechanical components such as muscle models, afferent feedback pathways, and limb dynamics, enabling research into closed-loop locomotor control and Central Pattern Generators (CPGs).

## Installation

Clone repository to your local computer:
```bash
git clone https://github.com/brodayagabro/SpikingNN
```

Cd to dir with package:
```bash
cd SpikingNN
```

Install package using pip:

```bash
pip install .
```

With devmode:
```bash
pip install -e .
```

## Command Line Interface (CLI)

SpikingNN provides a command-line interface for running simulations and launching the GUI.

### GUI Mode

Launch the interactive web interface:

```bash
# Launch GUI on default port (8501)
spikingnn gui

# Launch GUI on custom port
spikingnn gui --port 8502

# Launch GUI accessible from network
spikingnn gui --host 0.0.0.0

# Launch in debug mode
spikingnn gui --debug
```

### Simulation Mode

Run simulations from JSON configuration files:

```bash
# Run simulation with config file
spikingnn sim config.json

# Run simulation and save results to CSV
spikingnn sim config.json -o results.csv

# Run simulation with signal parameters
spikingnn sim config.json --signal-type sine --amplitude 10 --frequency 1

# Run simulation with phase shift
spikingnn sim config.json --signal-type sine --amplitude 10 --frequency 1 --phase 0.5

# Run simulation targeting specific neurons
spikingnn sim config.json --signal-type constant --amplitude 5 --neurons 0,1,2
```

### Short Command

Use `spknn` as a shortcut for `spikingnn`:

```bash
spknn gui
spknn sim config.json
```

## Python API

SpikingNN provides a function-based API for programmatic access to simulations.

### Basic Usage

```python
from SpikingNN import create_simulation, run_simulation, get_results

# Create simulation from JSON config
sim = create_simulation("config.json")

# Run simulation with signals
signals = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 0.25,  # Quarter period phase shift
    "neurons": [0, 1]
}

results = run_simulation(sim, signals)
data = get_results(results)

# Access results
print(f"Time steps: {len(data['time'])}")
print(f"Membrane potentials shape: {data['V'].shape}")
print(f"Recovery variables shape: {data['U'].shape}")
```

### Async API

For non-blocking execution:

```python
import asyncio
from SpikingNN import create_simulation, run_simulation_async, get_results

async def run_async():
    sim = create_simulation("config.json")
    results = await run_simulation_async(sim, signals)
    return get_results(results)

data = asyncio.run(run_async())
```

### Signal Types

The API supports various signal types for input currents:

| Signal Type | Description | Parameters |
|-------------|-------------|------------|
| `constant` | Constant current | `amplitude` |
| `sine` | Sinusoidal current | `amplitude`, `frequency`, `phase` |
| `square` | Square wave | `amplitude`, `frequency`, `phase` |
| `ramp` | Linear ramp | `amplitude` |
| `noise` | Gaussian noise | `amplitude` (std dev) |

### Phase Parameter

For periodic signals (sine, square), you can specify phase shift:

```python
# Phase as fraction of period (0 to 1)
signals = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 0.25  # 25% of period = 90 degrees
}

# Phase in radians
signals = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 1.5708  # pi/2 radians = 90 degrees
}
```

### Multi-Channel Signals

For simulations with multiple input channels:

```python
# Create simulation with 2 input channels
config = {
    "system_type": "onlynet",
    "network": {
        "neuron_count": 5,
        "input_size": 2,
        "output_size": 5,
        "afferent_size": 0
    },
    "simulation": {
        "dt": 0.1,
        "duration": 1000.0
    }
}

# Channel 0: sine with phase 0
signals_ch0 = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 0.0,
    "neurons": [0]
}

# Channel 1: sine with phase 0.5 (180 degrees)
signals_ch1 = {
    "type": "sine",
    "amplitude": 10.0,
    "frequency": 1.0,
    "phase": 0.5,
    "neurons": [1]
}
```

## Configuration File Format

### Network-Only System (`onlynet`)

```json
{
    "system_type": "onlynet",
    "network": {
        "neuron_count": 5,
        "neuron_types": ["RS", "FS", "IB", "CH", "RS"],
        "input_size": 2,
        "output_size": 5,
        "afferent_size": 0,
        "connectivity": [
            [0, 1, 1.0],
            [1, 2, -0.5]
        ],
        "weights": [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        "tau_syn": [
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0]
        ]
    },
    "simulation": {
        "dt": 0.1,
        "duration": 1000.0,
        "input_current": {
            "type": "sine",
            "amplitude": 10.0,
            "frequency": 1.0,
            "phase": 0.25,
            "neurons": [0, 1]
        }
    }
}
```

### Full System with Limbs (`fullsys`)

```json
{
    "system_type": "fullsys",
    "network": {
        "neuron_count": 12,
        "neuron_types": ["CH", "CH", "FS", "FS", "RS", "FS", "CH", "CH", "FS", "FS", "RS", "FS"],
        "input_size": 2,
        "output_size": 4,
        "afferent_size": 12
    },
    "simulation": {
        "dt": 0.1,
        "duration": 1000.0
    },
    "limbs": [
        {
            "name": "Front_Left",
            "mechanics": {
                "mass": 0.3,
                "length": 0.3,
                "viscosity": 0.01,
                "q0": 1.5708,
                "w0": 0.0,
                "tendon_a1": 0.06,
                "tendon_a2": 0.007
            },
            "flexor": {
                "w": 0.5,
                "N": 50,
                "A": 0.0074,
                "tau_c": 39,
                "tau_1": 21
            },
            "extensor": {
                "w": 0.5,
                "N": 50,
                "A": 0.0074,
                "tau_c": 39,
                "tau_1": 21
            }
        }
    ]
}
```

## GUI Interface

Streamlit GUI Features:
The interactive web interface allows users to design, simulate, and analyze neural networks without writing code. Key functionalities include:

- **Network Design**: Create networks with customizable neuron counts and types, and edit connectivity matrices interactively.
- **Parameter Tuning**: Adjust synaptic weights (excitatory/inhibitory), relaxation constants (τ), and individual input current vectors for each neuron.
- **Real-time Visualization**: View membrane potentials, spike rasters, network graph structures with directed edges, and weight heatmaps.
- **Simulation Control**: Start, stop, and reset simulations with configurable time steps and duration.
- **Data Management**: Save and load network configurations via JSON, and export simulation results (NPZ/CSV).

## Neuron Types

The library supports the following Izhikevich neuron types:

| Type | Name | Description |
|------|------|-------------|
| RS | Regular Spiking | Regular spiking neurons |
| FS | Fast Spiking | Fast spiking inhibitory neurons |
| IB | Intrinsically Bursting | Intrinsically bursting neurons |
| CH | Chattering | Chattering neurons |
| TC | Thalamo-Cortical | Thalamo-cortical neurons |
| RZ | Resonator | Resonator neurons |
| LTS | Low-Threshold Spiking | Low-threshold spiking neurons |

## Examples

See the `scripts/` directory for complete examples:
- `scripts/TZ1_1_1/` - Rybak 2002 network simulations with various stimulation patterns
- `test_api_demo.py` - API demonstration script

## License

MIT License