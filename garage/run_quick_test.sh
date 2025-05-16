#!/bin/bash

# This script runs a quick test of the benchmarks with smaller sample sizes
# Especially useful for testing if Scoreformer works on different datasets

# Set smaller sample size for quicker testing
SAMPLES=10000

# Create directories if they don't exist
mkdir -p benchmark_results/data
mkdir -p benchmark_results/models
mkdir -p benchmark_results/results

# Test Scoreformer with MovieLens-1M dataset
echo "==============================================================================="
echo "Testing Scoreformer on ml-1m dataset with $SAMPLES samples"
echo "==============================================================================="
python full_benchmark.py --dataset ml-1m --models Scoreformer --samples $SAMPLES

# Test Scoreformer with Amazon Books dataset (if available)
if [ -f "benchmark_results/data/amazon-books/ratings.csv" ]; then
    echo "==============================================================================="
    echo "Testing Scoreformer on amazon-books dataset with $SAMPLES samples"
    echo "==============================================================================="
    python full_benchmark.py --dataset amazon-books --models Scoreformer --samples $SAMPLES
else
    echo "Amazon Books dataset not found. Run 'python dataset_downloader.py --dataset amazon-books' to download it."
fi

# Test Scoreformer with Yelp dataset (if available)
if [ -f "benchmark_results/data/yelp/ratings.csv" ]; then
    echo "==============================================================================="
    echo "Testing Scoreformer on yelp dataset with $SAMPLES samples"
    echo "==============================================================================="
    python full_benchmark.py --dataset yelp --models Scoreformer --samples $SAMPLES
else
    echo "Yelp dataset not found. Run 'python dataset_downloader.py --dataset yelp' to download it."
fi

echo "==============================================================================="
echo "Quick test completed!"
echo "Check benchmark_results/results/ for the output files."
echo "===============================================================================" 