# Scoreformer Benchmark Framework

A comprehensive benchmarking framework for recommendation systems, comparing the novel Scoreformer model against state-of-the-art algorithms.

## Overview

This framework evaluates multiple recommendation system algorithms on standard datasets using consistent metrics to provide fair and reproducible comparisons. The benchmark includes implementations of:

1. **Scoreformer**: A novel transformer-based model with Direct-Neighborhood-Graph scoring
2. **CFUIcA**: Collaborative Filtering with User-Item Context-aware Attention
3. **NCF**: Neural Collaborative Filtering

Additional models (currently being stabilized):
4. **STGCN**: Spatial-Temporal Graph Convolutional Network
5. **NGCF**: Neural Graph Collaborative Filtering
6. **GraphSAGE**: Graph Sample and Aggregate

## Current Status

The benchmark framework currently focuses on the three core models (Scoreformer, CFUIcA, and NCF) which provide stable and reliable benchmarking results. The graph-based models (STGCN, NGCF, and GraphSAGE) are available in the codebase but are currently disabled in the main benchmarking run due to dimensional compatibility issues that are being addressed.

## Features

- Automated dataset downloading and preprocessing
- Consistent evaluation metrics (HR@K, NDCG@K)
- Robust error handling for training and evaluation
- Early stopping to prevent overfitting
- Detailed logging and result visualization
- Modular design for easy extension

## Datasets

Currently supported datasets:
- **MovieLens-100K**: 100,000 ratings from 943 users on 1,682 movies

## Usage

To run the benchmark:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the benchmark
python benchmark_scoreformer.py
```

## Model Configuration

Each model can be configured through hyperparameters in the main function. Current default configurations:

```python
# Scoreformer
{
    'num_layers': 2,
    'd_model': 64,
    'num_heads': 4, 
    'd_feedforward': 128,
    'input_dim': 32,
    'dropout': 0.1,
    'use_transformer': True,
    'use_dng': True,
    'use_weights': True
}

# CFUIcA
{
    'embedding_dim': 64,
    'attention_dim': 32,
    'dropout': 0.1
}

# NCF
{
    'embedding_dim': 64,
    'layers': [128, 64, 32],
    'dropout': 0.1
}
```

## Extending the Framework

To add a new model to the benchmark:
1. Create a new model file in the `bench_models` directory
2. Implement a compatible model interface with predict(), forward() methods
3. Add the model to the MODELS dictionary in benchmark_scoreformer.py
4. Configure hyperparameters in the main function

## Results

Benchmark results are saved in the `results` directory as JSON files.

## Citation

If you use this benchmark in your research, please cite:

```
@article{...}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 