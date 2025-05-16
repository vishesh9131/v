#!/bin/bash

# Set up trap to handle interruptions
trap 'echo "Script interrupted. Cleaning up..."; exit 1' INT TERM

# Install required dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the simplified benchmark
echo "Running simplified benchmark..."
python simple_benchmark.py

echo "Simplified benchmark completed! Check the 'benchmark_results/results' directory for results." 