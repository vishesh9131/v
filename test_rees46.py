#!/usr/bin/env python
"""
Test the REES46 dataset integration with the benchmarking framework.
"""

import sys
import os
from benchmark_scoreformer import DATASETS, preprocess_dataset, train_and_evaluate
from benchmark_scoreformer import MODELS

def main():
    """Test REES46 dataset integration"""
    # Only run on REES46 dataset
    dataset_name = "rees46"
    
    print(f"\n{'='*100}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*100}")
    
    # Process dataset
    data = preprocess_dataset(dataset_name)
    if data is None:
        print(f"Failed to process {dataset_name}.")
        return
    
    # Print basic statistics
    df = data['df']
    num_users = data['num_users']
    num_items = data['num_items']
    
    print(f"\nDataset statistics:")
    print(f"  Number of users: {num_users}")
    print(f"  Number of items: {num_items}")
    print(f"  Number of interactions: {len(df)}")
    print(f"  Density: {100 * len(df) / (num_users * num_items):.6f}%")
    
    # Define model hyperparameters for MFBias model (fastest to test)
    hyperparams = {
        'embedding_dim': 64,      # Smaller dimension for faster testing
        'dropout': 0.2,
        'num_epochs': 5,          # Fewer epochs for faster testing
        'learning_rate': 0.001,
        'weight_decay': 0.00001,
        'batch_size': 256,
        'patience': 2             # Lower patience for faster testing
    }
    
    # Test with just one model
    model_name = "MFBias"
    model_class = MODELS[model_name]
    
    print(f"\n{'-'*50}")
    print(f"Testing {model_name} on {dataset_name}")
    print(f"{'-'*50}")
    
    try:
        # Train and evaluate the model
        results = train_and_evaluate(model_class, model_name, dataset_name, hyperparams)
        
        # Print results
        print(f"\nTest results for {model_name}:")
        if "error" in results:
            print(f"  Error: {results['error']}")
        else:
            for metric, value in results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
    
    except Exception as e:
        print(f"Error testing {model_name} on {dataset_name}: {e}")

if __name__ == "__main__":
    main() 