#!/bin/bash

# Set up trap to handle interruptions
trap 'echo "Script interrupted. Cleaning up..."; exit 1' INT TERM

# Install required dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Clean up any previous temporary files
echo "Cleaning up previous runs..."
rm -f /tmp/benchmark_*.log

# Limit memory usage (MacOS)
echo "Setting memory limits..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run the benchmark with smaller footprint
echo "Running benchmark with optimized settings..."
python benchmark.py

echo "Benchmark completed! Check the 'benchmark_results' directory for results." 
# (base) visheshyadav@Vishesh-Yadav v % chmod +x run_benchmark.sh