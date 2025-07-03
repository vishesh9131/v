#!/bin/bash

# Scoreformer Benchmark Runner
# Comprehensive benchmarking script for comparing Scoreformer against competitive models

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATASET="ml-1m"
MODELS=("scoreformer" "ncf" "mfbias" "autorec")
SAMPLE_SIZE=""
DOWNLOAD_DATASETS=false
ALL_DATASETS=false
QUICK_TEST=false

# Available datasets
AVAILABLE_DATASETS=("ml-100k" "ml-1m" "ml-10m" "lastfm" "amazon-books" "yelp" "gowalla" "netflix")

# Available models
AVAILABLE_MODELS=("scoreformer" "ncf" "ngcf" "autorec" "dmf" "mfbias" "cfuica" "stgcn" "graphsage")

usage() {
    echo -e "${BLUE}Scoreformer Benchmark Runner${NC}"
    echo -e "${BLUE}=============================${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset DATASET       Dataset to use (default: ml-1m)"
    echo "  -m, --models MODEL1,MODEL2  Comma-separated list of models to benchmark"
    echo "  -s, --sample SIZE           Sample size for quick testing"
    echo "  --download-datasets         Download datasets before benchmarking"
    echo "  --all-datasets              Run benchmark on all available datasets"
    echo "  --quick                     Quick test with small sample (10,000 interactions)"
    echo "  --list-datasets             List available datasets"
    echo "  --list-models               List available models"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Available Datasets:"
    for dataset in "${AVAILABLE_DATASETS[@]}"; do
        echo "  - $dataset"
    done
    echo ""
    echo "Available Models:"
    for model in "${AVAILABLE_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Examples:"
    echo "  $0 -d ml-1m -m scoreformer,ncf,mfbias"
    echo "  $0 --all-datasets -m scoreformer,ncf"
    echo "  $0 --quick -d ml-100k"
    echo "  $0 --download-datasets -d ml-1m"
    echo ""
}

list_datasets() {
    echo -e "${BLUE}Available Datasets:${NC}"
    echo "==================="
    for dataset in "${AVAILABLE_DATASETS[@]}"; do
        echo -e "${GREEN}✓${NC} $dataset"
    done
}

list_models() {
    echo -e "${BLUE}Available Models:${NC}"
    echo "================="
    for model in "${AVAILABLE_MODELS[@]}"; do
        echo -e "${GREEN}✓${NC} $model"
    done
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_python_deps() {
    log_info "Checking Python dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if required packages are installed
    python3 -c "
import sys
missing_packages = []

try:
    import torch
except ImportError:
    missing_packages.append('torch')

try:
    import pandas
except ImportError:
    missing_packages.append('pandas')

try:
    import numpy
except ImportError:
    missing_packages.append('numpy')

try:
    import sklearn
except ImportError:
    missing_packages.append('scikit-learn')

try:
    import tqdm
except ImportError:
    missing_packages.append('tqdm')

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    print('Please install with: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
" || {
        log_error "Missing Python dependencies. Please install requirements.txt"
        exit 1
    }
    
    log_success "All Python dependencies are available"
}

download_datasets() {
    log_info "Downloading datasets..."
    
    if [ "$ALL_DATASETS" = true ]; then
        python3 dataset_downloader.py --all
    else
        python3 dataset_downloader.py --dataset "$DATASET"
    fi
    
    if [ $? -eq 0 ]; then
        log_success "Datasets downloaded successfully"
    else
        log_error "Failed to download datasets"
        exit 1
    fi
}

run_single_benchmark() {
    local dataset=$1
    local models_str=$2
    local sample_arg=$3
    
    log_info "Running benchmark on $dataset with models: $models_str"
    
    # Build Python command with properly separated models
    local cmd="python3 full_benchmark.py --dataset $dataset --models"
    
    # Convert comma-separated models to space-separated for Python
    IFS=',' read -ra MODEL_ARRAY <<< "$models_str"
    for model in "${MODEL_ARRAY[@]}"; do
        # Trim whitespace and add to command
        model=$(echo "$model" | xargs)
        cmd="$cmd $model"
    done
    
    if [ -n "$sample_arg" ]; then
        cmd="$cmd --sample $sample_arg"
    fi
    
    log_info "Executing: $cmd"
    
    # Run the benchmark
    if eval "$cmd"; then
        log_success "Benchmark completed for $dataset"
    else
        log_error "Benchmark failed for $dataset"
        return 1
    fi
}

run_all_datasets_benchmark() {
    local models_str=$1
    local sample_arg=$2
    
    log_info "Running benchmark on all datasets..."
    
    local failed_datasets=()
    
    for dataset in "${AVAILABLE_DATASETS[@]}"; do
        echo ""
        echo "================================================================"
        echo "BENCHMARKING DATASET: $dataset"
        echo "================================================================"
        
        if ! run_single_benchmark "$dataset" "$models_str" "$sample_arg"; then
            failed_datasets+=("$dataset")
        fi
    done
    
    # Summary
    echo ""
    echo "================================================================"
    echo "BENCHMARK SUMMARY"
    echo "================================================================"
    
    if [ ${#failed_datasets[@]} -eq 0 ]; then
        log_success "All datasets completed successfully!"
    else
        log_warning "Some datasets failed:"
        for dataset in "${failed_datasets[@]}"; do
            echo "  - $dataset"
        done
    fi
}

parse_models() {
    local models_input=$1
    IFS=',' read -ra ADDR <<< "$models_input"
    MODELS=()
    for model in "${ADDR[@]}"; do
        # Trim whitespace
        model=$(echo "$model" | xargs)
        # Check if model is valid
        if [[ " ${AVAILABLE_MODELS[@]} " =~ " ${model} " ]]; then
            MODELS+=("$model")
        else
            log_warning "Unknown model: $model (skipping)"
        fi
    done
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dataset)
                DATASET="$2"
                shift 2
                ;;
            -m|--models)
                parse_models "$2"
                shift 2
                ;;
            -s|--sample)
                SAMPLE_SIZE="$2"
                shift 2
                ;;
            --download-datasets)
                DOWNLOAD_DATASETS=true
                shift
                ;;
            --all-datasets)
                ALL_DATASETS=true
                shift
                ;;
            --quick)
                QUICK_TEST=true
                SAMPLE_SIZE="10000"
                shift
                ;;
            --list-datasets)
                list_datasets
                exit 0
                ;;
            --list-models)
                list_models
                exit 0
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Validate dataset
    if [ "$ALL_DATASETS" = false ] && [[ ! " ${AVAILABLE_DATASETS[@]} " =~ " ${DATASET} " ]]; then
        log_error "Invalid dataset: $DATASET"
        list_datasets
        exit 1
    fi
    
    # Set default models if none specified
    if [ ${#MODELS[@]} -eq 0 ]; then
        MODELS=("scoreformer" "ncf" "mfbias" "autorec")
    fi
    
    # Create models string
    models_str=$(IFS=,; echo "${MODELS[*]}")
    
    # Print configuration
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}SCOREFORMER BENCHMARK CONFIGURATION${NC}"
    echo -e "${BLUE}================================================================${NC}"
    if [ "$ALL_DATASETS" = true ]; then
        echo "Datasets: ALL (${AVAILABLE_DATASETS[*]})"
    else
        echo "Dataset: $DATASET"
    fi
    echo "Models: ${MODELS[*]}"
    if [ -n "$SAMPLE_SIZE" ]; then
        echo "Sample Size: $SAMPLE_SIZE"
    fi
    if [ "$QUICK_TEST" = true ]; then
        echo "Mode: Quick Test"
    fi
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    
    # Check dependencies
    check_python_deps
    
    # Download datasets if requested
    if [ "$DOWNLOAD_DATASETS" = true ]; then
        download_datasets
    fi
    
    # Run benchmarks
    if [ "$ALL_DATASETS" = true ]; then
        run_all_datasets_benchmark "$models_str" "$SAMPLE_SIZE"
    else
        run_single_benchmark "$DATASET" "$models_str" "$SAMPLE_SIZE"
    fi
    
    log_success "Benchmark run completed!"
    echo ""
    echo "Results are saved in the 'results/' directory"
    echo "Logs are saved in the current directory with timestamp"
}

# Run main function
main "$@" 