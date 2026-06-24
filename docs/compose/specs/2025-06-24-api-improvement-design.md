# [S1] Problem

The SpikingNN library currently requires users to write Python code to configure and run simulations. External researchers need a way to submit simulation configurations via JSON files, specify input signal parameters, and receive raw data results programmatically.

## Current Limitations
- No standardized configuration format
- No programmatic interface for remote execution
- No support for signal parameterization
- Limited error reporting for configuration issues

# [S2] Solution Overview

Implement a function-based API that accepts JSON configuration files, signal parameters, and returns raw simulation data asynchronously. The API will:
1. Validate configurations against a schema
2. Convert signal parameters to time series
3. Execute simulations in batch mode
4. Return results as NumPy arrays

# [S3] Configuration Schema

## Network Configuration
```json
{
  "network": {
    "neuron_count": 10,
    "neuron_types": ["RS", "FS", "IB"],
    "connectivity": [[0, 1, 1.0], [1, 2, -0.5]],
    "weights": [[0.0, 1.0, 0.0], [0.5, 0.0, 0.8]],
    "tau_syn": [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
  }
}
```

## Simulation Configuration
```json
{
  "simulation": {
    "dt": 0.1,
    "duration": 1000.0,
    "input_current": {
      "type": "constant",
      "amplitude": 10.0,
      "neurons": [0, 1]
    }
  }
}
```

## Signal Parameters
```json
{
  "type": "constant",
  "amplitude": 10.0,
  "frequency": 1.0,
  "duration": 1000.0,
  "neurons": [0, 1]
}
```

### Signal Types
- `constant`: Constant current `I(t) = amplitude`
- `sine`: Sinusoidal current `I(t) = amplitude * sin(2π * frequency * t)`
- `square`: Square wave current with given frequency
- `ramp`: Linear ramp from 0 to amplitude
- `noise`: Gaussian noise with given amplitude as standard deviation

### Parameters
- `type`: Signal type (required)
- `amplitude`: Signal amplitude in nA (required)
- `frequency`: Signal frequency in Hz (for periodic signals)
- `duration`: Signal duration in ms (default: simulation duration)
- `neurons`: List of neuron indices to apply signal (default: all neurons)

# [S4] API Functions

## create_simulation(config_path: str) -> Simulation
- Loads and validates JSON configuration
- Returns Simulation object for execution

## run_simulation(simulation: Simulation, signals: dict) -> Results
- Executes simulation with specified signals
- Returns Results object with raw data

## get_results(results: Results) -> dict
- Extracts raw data from Results object
- Returns dictionary of NumPy arrays

# [S5] Error Handling

- ConfigurationError: Invalid JSON or schema violations
- SimulationError: Runtime errors during execution
- ValidationError: Invalid signal parameters
- FileNotFoundError: Missing configuration files

# [S6] Data Flow

1. User provides JSON config file path
2. API validates configuration
3. User specifies signal parameters
4. API converts parameters to time series
5. Simulation executes asynchronously
6. Results returned as NumPy arrays

# [S7] Implementation Notes

- Use jsonschema for configuration validation
- Implement async/await for non-blocking execution
- Support batch processing of multiple simulations
- Maintain backward compatibility with existing classes
