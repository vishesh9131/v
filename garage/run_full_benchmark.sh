#!/bin/bash

# Set up trap to handle interruptions
trap 'echo "Script interrupted. Cleaning up..."; exit 1' INT TERM

# Install required dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Ensure results directory exists
mkdir -p benchmark_results/results

# Set environment variables to limit memory usage
echo "Setting memory limits..."
export PYTHONMALLOC=malloc
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Run the full benchmark
echo "Running full benchmark comparison with all models..."
python full_benchmark.py

echo "==================================================="
echo "Full benchmark completed!"
echo "Check benchmark_results/results directory for results and benchmark.log for detailed logs."
echo "===================================================" 