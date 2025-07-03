#!/bin/bash

# Scoreformer Comparison Script
# Focused comparison of Scoreformer against key baseline models

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}================================================================${NC}"
echo -e "${PURPLE}SCOREFORMER vs BASELINES COMPARISON${NC}"
echo -e "${PURPLE}================================================================${NC}"
echo ""

# Key baseline models for comparison
BASELINE_MODELS="ncf,mfbias,autorec,ngcf"
ALL_MODELS="scoreformer,$BASELINE_MODELS"

# Default configuration
DATASETS=("ml-100k" "ml-1m" "lastfm")
SAMPLE_SIZE=""
QUICK_MODE=false

usage() {
    echo "Scoreformer Comparison Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick              Quick mode with small sample (5000 interactions)"
    echo "  --full               Full comparison on all datasets"
    echo "  --dataset DATASET    Single dataset comparison"
    echo "  --sample SIZE        Custom sample size"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Default datasets: ${DATASETS[*]}"
    echo "Baseline models: $BASELINE_MODELS"
    echo ""
    echo "Examples:"
    echo "  $0 --quick"
    echo "  $0 --dataset ml-1m"
    echo "  $0 --full"
    exit 0
}

log_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

run_comparison() {
    local dataset=$1
    local sample_arg=$2
    
    log_header "COMPARING SCOREFORMER ON $dataset"
    
    # Download dataset if needed
    if [ ! -d "data/$dataset" ]; then
        log_info "Downloading $dataset..."
        python3 dataset_downloader.py --dataset "$dataset"
    fi
    
    # Build command with properly separated models
    local cmd="python3 full_benchmark.py --dataset $dataset --models"
    
    # Convert comma-separated models to space-separated for Python
    IFS=',' read -ra MODEL_ARRAY <<< "$ALL_MODELS"
    for model in "${MODEL_ARRAY[@]}"; do
        cmd="$cmd $model"
    done
    
    if [ -n "$sample_arg" ]; then
        cmd="$cmd --sample $sample_arg"
    fi
    
    log_info "Running: $cmd"
    
    # Execute benchmark
    if eval "$cmd"; then
        log_success "Comparison completed for $dataset"
        
        # Show results summary
        echo ""
        echo -e "${YELLOW}Results for $dataset:${NC}"
        echo "Check results/ directory for detailed metrics"
        echo ""
    else
        echo -e "${RED}[ERROR] Comparison failed for $dataset${NC}"
        return 1
    fi
}

create_summary_report() {
    log_header "GENERATING COMPARISON SUMMARY"
    
    # Find the latest result files
    latest_results=$(find results/ -name "benchmark_*.json" -type f -exec ls -t {} + 2>/dev/null | head -3)
    
    if [ -z "$latest_results" ]; then
        echo "No recent benchmark results found"
        return
    fi
    
    echo -e "${YELLOW}Recent benchmark results:${NC}"
    for file in $latest_results; do
        echo "  - $file"
    done
    
    # Create a simple comparison table using Python
    python3 -c "
import json
import glob
import os
from datetime import datetime

print('\n' + '='*80)
print('SCOREFORMER PERFORMANCE SUMMARY')
print('='*80)
print(f\"{'Dataset':<12} {'Scoreformer HR@10':<18} {'Best Baseline':<15} {'Improvement':<12}\")
print('-'*80)

# Find recent result files
result_files = sorted(glob.glob('results/benchmark_*.json'), key=os.path.getmtime, reverse=True)[:5]

for file in result_files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        
        # Extract dataset name from filename
        dataset = file.split('_')[1] if '_' in file else 'unknown'
        
        # Get Scoreformer performance
        scoreformer_hr = data.get('scoreformer', {}).get('HR@10', 0)
        
        # Find best baseline
        baselines = ['ncf', 'mfbias', 'autorec', 'ngcf', 'dmf']
        best_baseline = 0
        best_baseline_name = 'None'
        
        for model in baselines:
            if model in data and 'HR@10' in data[model]:
                hr = data[model]['HR@10']
                if hr > best_baseline:
                    best_baseline = hr
                    best_baseline_name = model
        
        # Calculate improvement
        if best_baseline > 0:
            improvement = ((scoreformer_hr - best_baseline) / best_baseline) * 100
            improvement_str = f'{improvement:+.1f}%'
        else:
            improvement_str = 'N/A'
        
        print(f'{dataset:<12} {scoreformer_hr:<18.4f} {best_baseline_name}({best_baseline:.4f}) {improvement_str:<12}')
        
    except Exception as e:
        print(f'Error processing {file}: {e}')

print('='*80)
"
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                QUICK_MODE=true
                SAMPLE_SIZE="5000"
                shift
                ;;
            --full)
                DATASETS=("ml-100k" "ml-1m" "ml-10m" "lastfm" "amazon-books")
                shift
                ;;
            --dataset)
                DATASETS=("$2")
                shift 2
                ;;
            --sample)
                SAMPLE_SIZE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Show configuration
    echo -e "${YELLOW}Comparison Configuration:${NC}"
    echo "Datasets: ${DATASETS[*]}"
    echo "Models: $ALL_MODELS"
    if [ -n "$SAMPLE_SIZE" ]; then
        echo "Sample Size: $SAMPLE_SIZE"
    fi
    if [ "$QUICK_MODE" = true ]; then
        echo "Mode: Quick"
    fi
    echo ""
    
    # Run comparisons
    failed_datasets=()
    for dataset in "${DATASETS[@]}"; do
        if ! run_comparison "$dataset" "$SAMPLE_SIZE"; then
            failed_datasets+=("$dataset")
        fi
    done
    
    # Show summary
    log_header "COMPARISON SUMMARY"
    
    if [ ${#failed_datasets[@]} -eq 0 ]; then
        log_success "All comparisons completed successfully!"
    else
        echo -e "${RED}Failed datasets: ${failed_datasets[*]}${NC}"
    fi
    
    # Generate comparison report
    create_summary_report
    
    echo ""
    echo -e "${GREEN}Scoreformer comparison completed!${NC}"
    echo "Detailed results are available in the results/ directory"
}

# Run main function
main "$@" 