#!/usr/bin/env python3
"""
Demo script for the SpikingNN API.

This script demonstrates how to use the new function-based API
to configure and run a spiking neural network simulation.
"""

import json
import tempfile
import asyncio
from SpikingNN import create_simulation, run_simulation, run_simulation_async, get_results


def create_demo_config():
    """Create a demo configuration file."""
    config = {
        "system_type": "onlynet",
        "network": {
            "neuron_count": 5,
            "neuron_types": ["RS", "FS", "IB", "CH", "RS"],
            "connectivity": [
                [0, 1, 1.0],  # Neuron 0 -> Neuron 1 (excitatory)
                [1, 1, -1],  # Neuron 1 -> Neuron 2 (inhibitory)
                [1, 1, 1],   # Neuron 2 -> Neuron 3 (excitatory)
                [1, 1, -1]   # Neuron 3 -> Neuron 4 (inhibitory)
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
                "type": "constant",
                "amplitude": 10.0,
                "neurons": [0, 1]
            }
        }
    }
    return config


def test_sync_api():
    """Test the synchronous API."""
    print("Testing synchronous API...")
    
    # Create configuration file
    config = create_demo_config()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    # Create simulation
    sim = create_simulation(config_path)
    print(f"Created simulation with {sim.network.N} neurons")
    
    # Define signals
    signals = {
        "type": "constant",
        "amplitude": 10.0,
        "neurons": [0, 1]
    }
    
    # Run simulation
    results = run_simulation(sim, signals)
    data = get_results(results)
    
    print(f"Simulation completed!")
    print(f"  Time steps: {len(data['time'])}")
    print(f"  V shape: {data['V'].shape}")
    print(f"  U shape: {data['U'].shape}")
    print(f"  V range: [{data['V'].min():.2f}, {data['V'].max():.2f}]")
    print(f"  U range: [{data['U'].min():.2f}, {data['U'].max():.2f}]")
    
    return data


async def test_async_api():
    """Test the asynchronous API."""
    print("\nTesting asynchronous API...")
    
    # Create configuration file
    config = create_demo_config()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    # Create simulation
    sim = create_simulation(config_path)
    print(f"Created simulation with {sim.network.N} neurons")
    
    # Define signals
    signals = {
        "type": "sine",
        "amplitude": 5.0,
        "frequency": 1.0,
        "neurons": [0]
    }
    
    # Run simulation asynchronously
    results = await run_simulation_async(sim, signals)
    data = get_results(results)
    
    print(f"Async simulation completed!")
    print(f"  Time steps: {len(data['time'])}")
    print(f"  V shape: {data['V'].shape}")
    print(f"  U shape: {data['U'].shape}")
    
    return data


def test_different_signal_types():
    """Test different signal types."""
    print("\nTesting different signal types...")
    
    config = create_demo_config()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    signal_types = ["constant", "sine", "square", "ramp", "noise"]
    
    for signal_type in signal_types:
        print(f"\n  Testing {signal_type} signal...")
        
        sim = create_simulation(config_path)
        
        signals = {
            "type": signal_type,
            "amplitude": 10.0,
            "neurons": [0]
        }
        
        if signal_type in ["sine", "square"]:
            signals["frequency"] = 1.0
        
        results = run_simulation(sim, signals)
        data = get_results(results)
        
        print(f"    V range: [{data['V'].min():.2f}, {data['V'].max():.2f}]")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SpikingNN API Demo")
    print("=" * 60)
    
    # Test synchronous API
    test_sync_api()
    
    # Test asynchronous API
    asyncio.run(test_async_api())
    
    # Test different signal types
    test_different_signal_types()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
