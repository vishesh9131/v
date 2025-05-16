#!/bin/bash

# Default values
DATASET="ml-1m"
MODELS="all"
SAMPLES=50000

# Function to display script usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -d, --dataset DATASET   Specify dataset to use (default: ml-1m)"
    echo "                         Available datasets: ml-1m, ml-1m-csv, amazon-books, yelp, lastfm, gowalla, netflix"
    echo "  -m, --models MODELS    Specify models to benchmark (comma-separated, default: all)"
    echo "                         Available models: BPR, NCF, NGCF, LightGCN, PinSage, SASRec, BERT4Rec, GTN, DualGNN, Scoreformer"
    echo "  -s, --samples SAMPLES  Maximum number of samples to use (default: 50000)"
    echo "  -a, --all-datasets     Run benchmarks on all available datasets"
    echo "  --download-datasets    Download and prepare all datasets first"
    echo "  -h, --help             Display this help message"
    exit 1
}

# Parse command line arguments
while [ "$1" != "" ]; do
    case $1 in
        -d | --dataset )          shift
                                  DATASET=$1
                                  ;;
        -m | --models )           shift
                                  MODELS=$1
                                  ;;
        -s | --samples )          shift
                                  SAMPLES=$1
                                  ;;
        -a | --all-datasets )     ALL_DATASETS=true
                                  ;;
        --download-datasets )     DOWNLOAD_DATASETS=true
                                  ;;
        -h | --help )             usage
                                  ;;
        * )                       echo "Unknown parameter: $1"
                                  usage
                                  ;;
    esac
    shift
done

# Create required directories
mkdir -p benchmark_results/data
mkdir -p benchmark_results/models
mkdir -p benchmark_results/results

# Download datasets if requested
if [ "$DOWNLOAD_DATASETS" = true ]; then
    echo "Downloading and preparing all datasets..."
    python dataset_downloader.py --dataset all
fi

# Convert models to python list format
if [ "$MODELS" != "all" ]; then
    # Replace commas with spaces and put in quotes
    MODELS_ARRAY=$(echo "$MODELS" | tr ',' ' ')
    MODELS_PARAM="--models $MODELS_ARRAY"
else
    MODELS_PARAM=""
fi

# Run benchmarks
if [ "$ALL_DATASETS" = true ]; then
    DATASETS=("ml-1m" "ml-1m-csv" "amazon-books" "yelp" "lastfm" "gowalla" "netflix")
    
    # Download all datasets first if they don't exist
    echo "Checking and downloading datasets if needed..."
    python dataset_downloader.py --dataset all
    
    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo "==============================================================================="
        echo "Starting benchmark for dataset: $dataset"
        echo "==============================================================================="
        echo ""
        
        if [ "$MODELS" = "all" ]; then
            python full_benchmark.py --dataset "$dataset" --samples $SAMPLES
        else
            python full_benchmark.py --dataset "$dataset" $MODELS_PARAM --samples $SAMPLES
        fi
    done
else
    echo ""
    echo "==============================================================================="
    echo "Starting benchmark for dataset: $DATASET"
    echo "==============================================================================="
    echo ""
    
    # Check if the dataset exists, if not download it
    dataset_file=""
    case $DATASET in
        ml-1m) dataset_file="benchmark_results/data/ml-1m/ratings.dat" ;;
        ml-1m-csv) dataset_file="benchmark_results/data/ml-1m/ratings.csv" ;;
        amazon-books) dataset_file="benchmark_results/data/amazon-books/ratings.csv" ;;
        yelp) dataset_file="benchmark_results/data/yelp/ratings.csv" ;;
        lastfm) dataset_file="benchmark_results/data/lastfm/ratings.csv" ;;
        gowalla) dataset_file="benchmark_results/data/gowalla/ratings.csv" ;;
        netflix) dataset_file="benchmark_results/data/netflix/ratings.csv" ;;
    esac
    
    if [ ! -f "$dataset_file" ]; then
        echo "Dataset file '$dataset_file' not found. Downloading now..."
        python dataset_downloader.py --dataset "$DATASET"
    fi
    
    if [ "$MODELS" = "all" ]; then
        python full_benchmark.py --dataset "$DATASET" --samples $SAMPLES
    else
        python full_benchmark.py --dataset "$DATASET" $MODELS_PARAM --samples $SAMPLES
    fi
fi

echo ""
echo "Benchmarks completed successfully!"
echo "Results are saved in benchmark_results/results/" 