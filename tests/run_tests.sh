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
python test_api_demo.py

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
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="