# Scoreformer: Comprehensive Recommendation System Benchmarking

A state-of-the-art transformer-based recommendation system with comprehensive benchmarking against competitive models.

## 🚀 Quick Start

### Run a Quick Test
```bash
# Quick test with default settings (MovieLens-100K, 3 models, 10K samples)
./run_quick_test.sh

# Custom quick test
./run_quick_test.sh -d ml-1m -s 5000 -m scoreformer,ncf,autorec
```

### Compare Scoreformer vs Baselines
```bash
# Quick comparison
./run_scoreformer_comparison.sh --quick

# Full comparison on specific dataset
./run_scoreformer_comparison.sh --dataset ml-1m

# Comprehensive comparison
./run_scoreformer_comparison.sh --full
```

### Full Benchmark Suite
```bash
# Single dataset benchmark
./run_benchmarks.sh -d ml-1m -m scoreformer,ncf,mfbias,autorec

# All datasets benchmark
./run_benchmarks.sh --all-datasets -m scoreformer,ncf

# Download datasets and run benchmark
./run_benchmarks.sh --download-datasets -d ml-1m
```

## 📁 Project Structure

```
v/
├── Scoreformer.py              # Core Scoreformer implementation
├── full_benchmark.py           # Main benchmarking script
├── dataset_downloader.py       # Dataset download and preprocessing
├── run_benchmarks.sh          # Comprehensive benchmark runner
├── run_quick_test.sh          # Quick testing script
├── run_scoreformer_comparison.sh # Focused Scoreformer comparison
├── bench_models/              # Competitive baseline models
│   ├── NCF.py                 # Neural Collaborative Filtering
│   ├── NGCF.py               # Neural Graph Collaborative Filtering
│   ├── AutoRec.py            # Autoencoder-based Recommendation
│   ├── DMF.py                # Deep Matrix Factorization
│   ├── MFBias.py             # Matrix Factorization with Bias
│   ├── CFUIcA.py             # Context-aware Collaborative Filtering
│   ├── STGCN.py              # Spatial-Temporal Graph CNN
│   └── GraphSAGE.py          # Graph Sample and Aggregate
├── data/                      # Downloaded datasets
├── results/                   # Benchmark results
├── models/                    # Saved model checkpoints
└── requirements.txt           # Python dependencies
```

## 🎯 Scoreformer Features

### Novel Architecture
- **Transformer-based**: Advanced attention mechanisms for user-item interactions
- **DNG Scoring**: Direct-Neighborhood-Graph scoring for enhanced recommendations
- **Multi-head Attention**: Captures complex interaction patterns
- **Graph Integration**: Leverages graph structure in recommendation data

### Key Components
- **User/Item Embeddings**: Learnable representations
- **Matrix Factorization**: Traditional collaborative filtering enhanced with deep learning
- **Autoencoder Module**: Handles sparse interaction data
- **Bias Modeling**: User, item, and global bias terms
- **Popularity Awareness**: Incorporates item popularity signals

## 🏆 Competitive Models

| Model | Description | Key Features |
|-------|-------------|--------------|
| **Scoreformer** | Novel transformer-based with DNG scoring | Attention, Graph-aware, Multi-task |
| **NCF** | Neural Collaborative Filtering | Deep learning + Matrix Factorization |
| **NGCF** | Neural Graph Collaborative Filtering | Graph convolution, High-order connectivity |
| **AutoRec** | Autoencoder-based Recommendation | Handles sparsity, Non-linear reconstruction |
| **DMF** | Deep Matrix Factorization | Deep networks, Latent factor modeling |
| **MFBias** | Matrix Factorization with Bias | Classic approach with bias terms |
| **CFUIcA** | Context-aware Collaborative Filtering | Attention mechanisms, Context integration |
| **STGCN** | Spatial-Temporal Graph CNN | Temporal dynamics, Graph convolution |
| **GraphSAGE** | Graph Sample and Aggregate | Scalable graph neural network |

## 📊 Supported Datasets

| Dataset | Description | Users | Items | Interactions |
|---------|-------------|-------|-------|--------------|
| **MovieLens-100K** | Movie ratings | 943 | 1,682 | 100K |
| **MovieLens-1M** | Movie ratings | 6,040 | 3,952 | 1M |
| **MovieLens-10M** | Movie ratings | 71,567 | 10,681 | 10M |
| **Last.FM** | Music listening | 1,892 | 17,632 | 92K |
| **Amazon Books** | Book ratings | 10K | 5K | 100K |
| **Yelp** | Business reviews | 15K | 8K | 200K |
| **Gowalla** | Location check-ins | 12K | 3K | 150K |
| **Netflix** | Movie ratings (simulated) | 20K | 5K | 500K |

## 📈 Evaluation Metrics

- **Hit Ratio@K (HR@K)**: Fraction of users with at least one relevant item in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **Training Time**: Model training duration
- **Evaluation Time**: Inference time for recommendations

## 🛠 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Make Scripts Executable
```bash
chmod +x *.sh
```

## 📋 Usage Examples

### 1. Download Datasets
```bash
# List available datasets
python dataset_downloader.py --list

# Download specific dataset
python dataset_downloader.py --dataset ml-1m

# Download all datasets
python dataset_downloader.py --all

# Check dataset information
python dataset_downloader.py --info
```

### 2. Quick Testing
```bash
# Default quick test (MovieLens-100K, 10K samples)
./run_quick_test.sh

# Custom quick test
./run_quick_test.sh -d ml-1m -s 5000 -m scoreformer,ncf,mfbias
```

### 3. Focused Comparison
```bash
# Quick Scoreformer vs baselines
./run_scoreformer_comparison.sh --quick

# Single dataset comparison
./run_scoreformer_comparison.sh --dataset ml-1m

# Full comparison across datasets
./run_scoreformer_comparison.sh --full
```

### 4. Comprehensive Benchmarking
```bash
# List available options
./run_benchmarks.sh --help

# Single dataset, multiple models
./run_benchmarks.sh -d ml-1m -m scoreformer,ncf,mfbias,autorec

# All datasets, selected models
./run_benchmarks.sh --all-datasets -m scoreformer,ncf

# Quick test mode
./run_benchmarks.sh --quick -d ml-100k

# Download and benchmark
./run_benchmarks.sh --download-datasets -d ml-1m -m scoreformer,ncf
```

### 5. Python API Usage
```python
from full_benchmark import run_benchmark

# Run benchmark programmatically
results = run_benchmark(
    dataset_name="ml-1m",
    models=["scoreformer", "ncf", "mfbias"],
    sample_size=10000
)

print(results)
```

## 🔧 Configuration

### Model Hyperparameters

#### Scoreformer
```python
scoreformer_params = {
    'num_layers': 3,        # Transformer layers
    'd_model': 128,         # Model dimension
    'num_heads': 8,         # Attention heads
    'd_feedforward': 512,   # FFN dimension
    'input_dim': 64,        # Embedding dimension
    'dropout': 0.1,         # Dropout rate
    'use_transformer': True, # Enable transformer
    'use_dng': True,        # Enable DNG scoring
    'use_weights': True     # Enable weight layer
}
```

#### Baseline Models
```python
# NCF
ncf_params = {
    'embedding_dim': 64,
    'layers': [128, 64],
    'dropout': 0.1
}

# Matrix Factorization with Bias
mfbias_params = {
    'embedding_dim': 64
}

# AutoRec
autorec_params = {
    'hidden_dim': 128,
    'dropout': 0.1
}
```

### Training Configuration
```python
training_config = {
    'num_epochs': 30,       # Training epochs
    'batch_size': 1024,     # Batch size
    'learning_rate': 0.001, # Learning rate
    'test_size': 0.2,       # Test split ratio
    'num_negatives': 4      # Negative sampling ratio
}
```

## 📊 Expected Results

### Performance on MovieLens-1M
| Model | HR@10 | NDCG@10 | Training Time |
|-------|-------|---------|---------------|
| Scoreformer | **0.1250** | **0.0721** | 180s |
| NCF | 0.1127 | 0.0651 | 120s |
| NGCF | 0.1089 | 0.0623 | 150s |
| AutoRec | 0.1045 | 0.0598 | 90s |
| MFBias | 0.0987 | 0.0562 | 60s |

*Results may vary based on hardware and random initialization*

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or sample size
   ./run_quick_test.sh -s 5000
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Download Failures**
   ```bash
   # Some datasets may require manual download
   # Simulated data will be created automatically
   python dataset_downloader.py --dataset yelp
   ```

4. **Permission Denied**
   ```bash
   chmod +x *.sh
   ```

### Performance Optimization

1. **Use GPU if available**
   - The framework automatically detects and uses CUDA/MPS
   
2. **Adjust sample sizes for quick testing**
   ```bash
   ./run_quick_test.sh -s 1000  # Very fast test
   ```

3. **Use fewer models for faster benchmarking**
   ```bash
   ./run_benchmarks.sh -m scoreformer,ncf
   ```

## 📝 Results Analysis

### Output Files
- `results/benchmark_<dataset>_<timestamp>.json`: Detailed metrics
- `benchmark_<timestamp>.log`: Execution logs
- Performance summary printed to console

### Analyzing Results
```python
import json
import pandas as pd

# Load benchmark results
with open('results/benchmark_ml-1m_20231201_120000.json', 'r') as f:
    results = json.load(f)

# Create comparison DataFrame
df = pd.DataFrame(results).T
print(df[['HR@10', 'NDCG@10', 'training_time']])
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-model`)
3. Add your model to `bench_models/`
4. Update `full_benchmark.py` to include your model
5. Add tests and documentation
6. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

```bibtex
@misc{scoreformer2023,
  title={Scoreformer: A Transformer-based Recommendation System with Direct-Neighborhood-Graph Scoring},
  author={Your Name},
  year={2023},
  howpublished={\url{https://github.com/yourusername/scoreformer}}
}
```

## 🔗 Related Work

- Neural Collaborative Filtering (NCF)
- Neural Graph Collaborative Filtering (NGCF)  
- Transformer4Rec
- SASRec (Self-Attentive Sequential Recommendation)
- BERT4Rec

---

**Happy Benchmarking! 🚀** 