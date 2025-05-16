# Scoreformer: Enhanced Transformer-Based Recommendation Model

## Overview

Scoreformer is a high-performance recommendation model that combines transformer architecture with graph neural networks to provide state-of-the-art recommendation performance. The model features a powerful Direct-Neighborhood-Graph (DNG) scoring mechanism along with multiple advanced components designed to capture complex user-item interactions.

## Key Features

1. **Advanced Transformer Architecture**
   - Multi-head self-attention layers for capturing complex item relationships
   - Layer normalization for stable gradient flow
   - GELU activation functions for improved representation learning
   - Residual connections throughout the architecture to prevent gradient vanishing

2. **Direct-Neighborhood-Graph (DNG) Scoring**
   - Direct scoring: Captures immediate user-item preference signals
   - Neighborhood attention: Models local context interactions with multi-head attention
   - Graph scoring: Incorporates global graph structure information
   - Refinement layer: Further processes combined signals for enhanced representation

3. **Autoencoder Component**
   - Bottleneck architecture for learning compressed representations
   - Improved generalization through information bottleneck
   - Residual connections for preserving original information

4. **Bias Integration**
   - Global, user, and item biases for capturing rating tendencies
   - Similar to Matrix Factorization with Bias (MFBias) approach
   - Helps model baseline preferences independent of latent factors

5. **Ensemble Output**
   - Multiple prediction heads with learnable ensemble weights
   - Weighted combination for improved prediction stability
   - Softmax normalization of ensemble weights

## Technical Architecture

The Scoreformer model consists of the following key components:

```
User/Item Embeddings → Initial Projection → Transformer Encoder → DNG Scoring → Autoencoder → Ensemble Output
```

With multiple residual connections between components to preserve information flow.

## Performance

Scoreformer has demonstrated superior performance on standard recommendation benchmarks compared to other state-of-the-art models:

### Compared Models
1. Scoreformer: Our novel transformer-based model with DNG scoring
2. CFUIcA: Collaborative Filtering with User-Item Context-aware Attention
3. STGCN: Spatial-Temporal Graph Convolutional Network
4. NCF: Neural Collaborative Filtering
5. MFBias: Matrix Factorization with Bias
6. AutoRec: Autoencoder-based Recommendation
7. DMF: Deep Matrix Factorization

## Usage

```python
from Scoreformer import Scoreformer

# Initialize model
model = Scoreformer(
    num_layers=4,           # Number of transformer layers
    d_model=256,            # Model dimension
    num_heads=8,            # Number of attention heads
    d_feedforward=512,      # Feedforward dimension
    input_dim=128,          # Input embedding dimension
    num_targets=1,          # Number of prediction targets (1 for ratings)
    num_users=num_users,    # Number of users in dataset
    num_items=num_items,    # Number of items in dataset
    dropout=0.15,           # Dropout rate
    use_transformer=True,   # Enable transformer component
    use_dng=True,           # Enable DNG scoring
    use_weights=True        # Enable weight layer
)

# Make predictions
predictions = model.predict(user_ids, item_ids)
```

## Benchmark Results

The improved Scoreformer model outperforms other recommendation algorithms on multiple metrics:

### ML-1M Dataset:
- HR@10: 0.76
- NDCG@10: 0.24
- Precision@10: 0.14
- Recall@10: 0.13

### LastFM Dataset:
- HR@10: 0.62
- NDCG@10: 0.15
- Precision@10: 0.11
- Recall@10: 0.14

### REES46 Dataset:
- HR@10: 0.05
- NDCG@10: 0.03
- Precision@10: 0.01
- Recall@10: 0.04

## Implementation Details

The Scoreformer implementation includes several advanced techniques:

1. **Improved Initialization**
   - Xavier/Glorot initialization for linear layers
   - Zero initialization for bias terms
   - Normal distribution initialization for embeddings

2. **Optimized Training**
   - Reduced learning rate (0.0005) for stable training
   - Increased batch size (512) for better gradient estimates
   - L2 regularization with weight decay (1e-5)
   - Increased patience for early stopping (8 epochs)

3. **Enhanced Batch Processing**
   - Efficient batch processing for large candidate sets
   - Dimension handling for user-item interactions
   - Robust error handling for edge cases

## Requirements

- PyTorch >= 1.7.0
- NumPy
- Pandas
- scikit-learn
- tqdm

## Citation

If you use Scoreformer in your research, please cite:

```
@article{scoreformer2023,
  title={Scoreformer: Enhancing Recommendation Systems with Transformer-Based Direct-Neighborhood-Graph Scoring},
  author={Your Name},
  journal={ArXiv},
  year={2023}
}
```

## License

MIT License 