#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np
from typing import List, Tuple

class UltraFastHashClassifier(nn.Module):
    """
    Ultra-fast classifier using locality-sensitive hashing (LSH)
    No backprop needed - uses random projections and hash lookups
    """
    def __init__(self, input_dim=512, num_classes=10, num_hashes=64, hash_dim=8):
        super().__init__()
        self.num_classes = num_classes
        self.num_hashes = num_hashes
        self.hash_dim = hash_dim
        
        # Random projection matrices (fixed, no learning)
        self.register_buffer('projections', torch.randn(num_hashes, input_dim, hash_dim))
        
        # Hash table for each class (learned once, then frozen)
        self.register_buffer('class_signatures', torch.zeros(num_classes, num_hashes, dtype=torch.long))
        
        # Simple linear projection (optional, can be identity)
        self.feature_proj = nn.Linear(input_dim, input_dim, bias=False)
        
    def compute_hash_signature(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LSH signature for input batch"""
        B = x.size(0)
        # Project to hash dimensions: [B, num_hashes, hash_dim]
        projected = torch.einsum('bi,hij->bhj', x, self.projections)
        # Convert to binary hash (sign-based LSH)
        hashes = (projected > 0).long()
        # Convert binary to integer signature
        powers = 2 ** torch.arange(self.hash_dim, device=x.device)
        signatures = (hashes * powers).sum(dim=-1)  # [B, num_hashes]
        return signatures
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional feature transformation
        features = self.feature_proj(x)
        
        # Compute hash signatures
        signatures = self.compute_hash_signature(features)  # [B, num_hashes]
        
        # Compare with class signatures using Hamming distance
        B = signatures.size(0)
        scores = torch.zeros(B, self.num_classes, device=x.device)
        
        for c in range(self.num_classes):
            # Hamming distance = number of matching hashes
            matches = (signatures == self.class_signatures[c]).float()
            scores[:, c] = matches.mean(dim=1)  # Average match rate
        
        return scores * 10  # Scale for better logits

class DirectEmbeddingClassifier(nn.Module):
    """
    Direct embedding classification - no iterations, just matrix operations
    """
    def __init__(self, input_dim=512, embed_dim=64, num_classes=10):
        super().__init__()
        # Single transformation to embedding space
        self.to_embedding = nn.Linear(input_dim, embed_dim, bias=False)
        
        # Class prototypes (learned)
        self.register_buffer('class_prototypes', torch.randn(num_classes, embed_dim))
        
        # Optional: learned temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direct embedding
        embeddings = self.to_embedding(x)  # [B, embed_dim]
        
        # Cosine similarity to prototypes
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.class_prototypes, p=2, dim=1)
        
        # Compute similarities
        similarities = embeddings_norm @ prototypes_norm.T  # [B, num_classes]
        
        return similarities / self.temperature

class LinearOnlyClassifier(nn.Module):
    """
    Ultra-simple linear-only classifier with precomputed basis
    """
    def __init__(self, input_dim=512, num_classes=10, basis_dim=128):
        super().__init__()
        # Fixed random basis (no learning)
        self.register_buffer('basis', torch.randn(input_dim, basis_dim))
        
        # Simple linear classifier
        self.classifier = nn.Linear(basis_dim, num_classes, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project onto fixed basis
        projected = x @ self.basis  # [B, basis_dim]
        
        # Apply ReLU for nonlinearity
        activated = F.relu(projected)
        
        # Linear classification
        return self.classifier(activated)

class QuantizedLookupClassifier(nn.Module):
    """
    Quantized lookup table classifier - ultra fast inference
    """
    def __init__(self, input_dim=512, num_classes=10, codebook_size=256, num_codebooks=4):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.num_classes = num_classes
        self.subvector_dim = input_dim // num_codebooks
        
        # Quantization codebooks
        self.register_buffer('codebooks', 
                           torch.randn(num_codebooks, codebook_size, self.subvector_dim))
        
        # Lookup tables for each codebook
        self.lookup_tables = nn.ModuleList([
            nn.Embedding(codebook_size, num_classes) 
            for _ in range(num_codebooks)
        ])
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input using product quantization"""
        B = x.size(0)
        x_reshaped = x.view(B, self.num_codebooks, self.subvector_dim)
        
        indices = torch.zeros(B, self.num_codebooks, dtype=torch.long, device=x.device)
        
        for i in range(self.num_codebooks):
            # Find nearest codebook entry
            dists = torch.cdist(x_reshaped[:, i:i+1], self.codebooks[i:i+1])
            indices[:, i] = dists.argmin(dim=-1).squeeze()
        
        return indices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input
        indices = self.quantize(x)  # [B, num_codebooks]
        
        # Lookup and aggregate
        scores = torch.zeros(x.size(0), self.num_classes, device=x.device)
        
        for i in range(self.num_codebooks):
            scores += self.lookup_tables[i](indices[:, i])
        
        return scores

def benchmark_ultra_fast_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    input_dim = 512
    num_classes = 10
    
    # Traditional baseline
    traditional = nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, num_classes)
    ).to(device)
    
    # Ultra-fast models
    hash_classifier = UltraFastHashClassifier(input_dim, num_classes).to(device)
    direct_classifier = DirectEmbeddingClassifier(input_dim, 64, num_classes).to(device)
    linear_classifier = LinearOnlyClassifier(input_dim, num_classes).to(device)
    quantized_classifier = QuantizedLookupClassifier(input_dim, num_classes).to(device)
    
    # Initialize hash signatures (simulate training)
    with torch.no_grad():
        dummy_data = torch.randn(num_classes * 10, input_dim, device=device)
        for c in range(num_classes):
            class_data = dummy_data[c*10:(c+1)*10]
            signatures = hash_classifier.compute_hash_signature(class_data)
            hash_classifier.class_signatures[c] = signatures.mode(dim=0).values
    
    # Test data
    x = torch.randn(batch_size, input_dim, device=device)
    
    models = {
        'Traditional NN': traditional,
        'Hash Classifier': hash_classifier,
        'Direct Embedding': direct_classifier,
        'Linear Only': linear_classifier,
        'Quantized Lookup': quantized_classifier,
    }
    
    results = {}
    
    for name, model in models.items():
        # Warmup
        with torch.no_grad():
            _ = model(x)
        
        # Benchmark
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        
        with torch.no_grad():
            for _ in range(1000):  # More iterations for better measurement
                _ = model(x)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        avg_time = (time.time() - start) / 1000
        
        results[name] = avg_time
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:20} | {avg_time*1000:6.3f} ms | {params:8,} params")
    
    # Calculate speedups
    baseline = results['Traditional NN']
    print(f"\nSpeedup vs Traditional NN:")
    for name, time_val in results.items():
        if name != 'Traditional NN':
            speedup = baseline / time_val
            print(f"{name:20} | {speedup:6.1f}x faster")
    
    return results

def create_minimal_ultra_fast_classifier(input_dim=512, num_classes=10):
    """
    Create the absolute fastest classifier possible
    Just matrix multiplications and element-wise operations
    """
    class MinimalFastClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # Single matrix multiplication
            self.W = nn.Parameter(torch.randn(input_dim, num_classes) * 0.01)
            self.b = nn.Parameter(torch.zeros(num_classes))
        
        def forward(self, x):
            return x @ self.W + self.b
    
    return MinimalFastClassifier()

if __name__ == "__main__":
    print("Benchmarking Ultra-Fast No-Backprop Classifiers")
    print("=" * 60)
    results = benchmark_ultra_fast_models()
    
    print("\n" + "=" * 60)
    print("Testing minimal classifier:")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    minimal = create_minimal_ultra_fast_classifier().to(device)
    x = torch.randn(128, 512, device=device)
    
    # Benchmark minimal
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        for _ in range(1000):
            _ = minimal(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    minimal_time = (time.time() - start) / 1000
    
    traditional_time = results['Traditional NN']
    speedup = traditional_time / minimal_time
    
    print(f"Minimal Classifier:    {minimal_time*1000:6.3f} ms | {speedup:6.1f}x faster") 