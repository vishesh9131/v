#!/bin/bash

# Quick Test Script for Scoreformer
# Runs a fast benchmark with small sample size for rapid testing

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}SCOREFORMER QUICK TEST${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Default configuration for quick testing
DATASET="ml-100k"
# DATASET="REES46"
SAMPLE_SIZE="100000"
MODELS="scoreformer,ncf,mfbias"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -s|--sample)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Quick Test Script for Scoreformer"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --dataset DATASET    Dataset to use (default: ml-100k)"
            echo "  -s, --sample SIZE        Sample size (default: 100000)"
            echo "  -m, --models MODELS      Comma-separated models (default: scoreformer,ncf,mfbias)"
            echo "  -h, --help               Show this help"
            echo ""
            echo "Example: $0 -d ml-1m -s 5000 -m scoreformer,ncf"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Quick Test Configuration:${NC}"
echo "Dataset: $DATASET"
echo "Sample Size: $SAMPLE_SIZE"
echo "Models: $MODELS"
echo ""

# Check if dataset exists, if not download it
if [ ! -d "data/$DATASET" ]; then
    echo -e "${YELLOW}Dataset not found, downloading...${NC}"
    python3 dataset_downloader.py --dataset "$DATASET"
fi

# Run the quick benchmark
echo -e "${BLUE}Running quick benchmark...${NC}"
python3 full_benchmark.py \
    --dataset "$DATASET" \
    --models ${MODELS//,/ } \
    --sample "$SAMPLE_SIZE"

echo ""
echo -e "${GREEN}Quick test completed!${NC}"
echo "Check the results/ directory for detailed results." 