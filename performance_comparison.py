#!/usr/bin/env python3
import torch
import torch.nn as nn
import time

# Simple traditional classifier
class TraditionalClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# Simulate NoPropCT-style iterative inference
class IterativeClassifier(nn.Module):
    def __init__(self, input_dim=512, embed_dim=64, num_classes=10):
        super().__init__()
        self.feature_net = nn.Linear(input_dim, embed_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),  # features + embedding + time
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, embed_dim))
        
    def forward_traditional(self, x):
        """Single forward pass (like traditional NN)"""
        features = self.feature_net(x)
        # Direct classification via embedding similarity
        logits = features @ self.class_embeddings.T
        return logits
    
    def forward_iterative(self, x, T_steps=40):
        """Iterative inference (like NoPropCT)"""
        B = x.size(0)
        features = self.feature_net(x)
        
        # Start with random embedding
        z = torch.randn(B, self.class_embeddings.size(1), device=x.device)
        
        for i in range(T_steps):
            t = torch.full((B, 1), i/T_steps, device=x.device)
            # Concatenate features, current embedding, and time
            input_vec = torch.cat([features, z, t], dim=1)
            # Predict clean embedding
            predicted_clean = self.denoiser(input_vec)
            # Move towards predicted clean embedding
            z = z + 0.1 * (predicted_clean - z)
        
        # Final classification
        logits = z @ self.class_embeddings.T
        return logits

def benchmark_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    input_dim = 512
    num_classes = 10
    
    # Create models
    traditional = TraditionalClassifier(input_dim, num_classes).to(device)
    iterative = IterativeClassifier(input_dim, 64, num_classes).to(device)
    
    # Dummy data
    x = torch.randn(batch_size, input_dim, device=device)
    
    # Warm up
    with torch.no_grad():
        _ = traditional(x)
        _ = iterative.forward_traditional(x)
        _ = iterative.forward_iterative(x, 40)
    
    # Benchmark traditional
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = traditional(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    traditional_time = (time.time() - start) / 100
    
    # Benchmark iterative (single pass)
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = iterative.forward_traditional(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    single_pass_time = (time.time() - start) / 100
    
    # Benchmark iterative (full NoPropCT-style)
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        for _ in range(10):  # Fewer runs due to slowness
            _ = iterative.forward_iterative(x, 40)
    torch.cuda.synchronize() if device == 'cuda' else None
    iterative_time = (time.time() - start) / 10
    
    print(f"Performance Comparison (batch_size={batch_size}):")
    print(f"Traditional NN:      {traditional_time*1000:.2f} ms")
    print(f"Single Pass:         {single_pass_time*1000:.2f} ms")
    print(f"Iterative (40 steps): {iterative_time*1000:.2f} ms")
    print(f"Slowdown factor:     {iterative_time/traditional_time:.1f}x")
    
    # Memory usage (rough estimate)
    traditional_params = sum(p.numel() for p in traditional.parameters())
    iterative_params = sum(p.numel() for p in iterative.parameters())
    
    print(f"\nModel Parameters:")
    print(f"Traditional:  {traditional_params:,}")
    print(f"Iterative:    {iterative_params:,}")

if __name__ == "__main__":
    benchmark_models() 