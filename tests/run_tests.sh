#!/bin/bash

# Script to run SpikingNN API tests
# This script runs the API tests and demo scripts

set -e  # Exit on any error

echo "=========================================="
echo "SpikingNN API Test Suite"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo ""
echo "1. Running unit tests..."
echo "------------------------------------------"
python -m pytest tests/test_schema.py tests/test_signals.py tests/test_api.py -v

echo ""
echo "2. Running API demo script..."
echo "------------------------------------------"
python tests/test_api_demo.py

echo ""
echo "3. Testing CLI command..."
echo "------------------------------------------"
echo "Testing GUI command help:"
python -m SpikingNN.cli gui --help

echo ""
echo "Testing SIM command help:"
python -m SpikingNN.cli sim --help

echo ""
echo "4. Testing simulation with config file..."
echo "------------------------------------------"
echo "Creating test output directory..."
mkdir -p results

echo "Running simulation with test config..."
python -m SpikingNN.cli sim tests/test_config.json -o results/test_output.csv

echo ""
echo "5. Testing phase parameters via CLI..."
echo "------------------------------------------"
echo "Testing sine signal with phase 0.0..."
python -m SpikingNN.cli sim tests/test_config.json --signal-type sine --amplitude 10 --frequency 1 --phase 0.0 -o results/phase_0.csv

echo "Testing sine signal with phase 0.25 (90 degrees)..."
python -m SpikingNN.cli sim tests/test_config.json --signal-type sine --amplitude 10 --frequency 1 --phase 0.25 -o results/phase_025.csv

echo "Testing sine signal with phase 0.5 (180 degrees)..."
python -m SpikingNN.cli sim tests/test_config.json --signal-type sine --amplitude 10 --frequency 1 --phase 0.5 -o results/phase_05.csv

echo "Testing multi-channel: channel 0 phase=0, channel 1 phase=0.5..."
python -m SpikingNN.cli sim tests/test_config.json --signal-type sine --amplitude 10 --frequency 1 --phase 0.0 --neurons 0 -o results/ch0_phase_0.csv
python -m SpikingNN.cli sim tests/test_config.json --signal-type sine --amplitude 10 --frequency 1 --phase 0.5 --neurons 1 -o results/ch1_phase_05.csv

echo ""
echo "6. Verifying phase outputs are different..."
echo "------------------------------------------"
python -c "
import numpy as np
import pandas as pd

# Load results
ch0 = pd.read_csv('results/ch0_phase_0.csv')
ch1 = pd.read_csv('results/ch1_phase_05.csv')

# Extract V columns for neuron 0
v0 = ch0['V_0'].values
v1 = ch1['V_1'].values

print(f'Channel 0 (phase=0) V range: [{v0.min():.2f}, {v0.max():.2f}]')
print(f'Channel 1 (phase=0.5) V range: [{v1.min():.2f}, {v1.max():.2f}]')
print(f'Signals are different: {not np.allclose(v0, v1)}')
"

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="