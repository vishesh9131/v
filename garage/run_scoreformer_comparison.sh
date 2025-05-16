#!/bin/bash

# This script compares Scoreformer with NCF (a baseline model) across multiple datasets
# It's designed to specifically evaluate Scoreformer's performance

# Default settings
SAMPLES=25000
BASELINE_MODEL="NCF"  # Neural Collaborative Filtering as baseline

# Parse command-line arguments
while [ "$1" != "" ]; do
    case $1 in
        --baseline )  shift
                     BASELINE_MODEL=$1
                     ;;
        --samples )   shift
                     SAMPLES=$1
                     ;;
        * )          echo "Unknown parameter: $1"
                     exit 1
                     ;;
    esac
    shift
done

echo "Starting Scoreformer comparison with $BASELINE_MODEL across multiple datasets"
echo "Using $SAMPLES samples per dataset"
echo

# Create required directories
mkdir -p benchmark_results/data
mkdir -p benchmark_results/models
mkdir -p benchmark_results/results

# Define datasets to test
DATASETS=("ml-1m" "amazon-books" "yelp" "netflix")

# Ensure we have all datasets
echo "Checking datasets..."
python dataset_downloader.py --dataset all

# Run comparison for each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "==============================================================================="
    echo "Comparing Scoreformer vs $BASELINE_MODEL on $dataset dataset"
    echo "==============================================================================="
    echo ""
    
    # Run the benchmark with just Scoreformer and the baseline model
    python full_benchmark.py --dataset "$dataset" --models "Scoreformer,$BASELINE_MODEL" --samples $SAMPLES
    
    echo "Completed comparison on $dataset dataset"
    echo ""
done

echo "==============================================================================="
echo "All comparisons completed!"
echo "Results are saved in benchmark_results/results/"
echo "==============================================================================="

# Optional: Display a summary of the results
echo "Summary of Results:"
echo "-------------------------------------------------------------------------------"
for dataset in "${DATASETS[@]}"; do
    # Find the most recent result file for this dataset
    result_file=$(ls -t benchmark_results/results/comparison_table_${dataset}_*.csv 2>/dev/null | head -1)
    if [ -n "$result_file" ]; then
        echo "Results for $dataset:"
        cat "$result_file"
        echo ""
    else
        echo "No results found for $dataset"
    fi
done 