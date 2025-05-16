#!/usr/bin/env python
"""
Test all recommendation models on the LastFM dataset.
"""

import sys
import os
import json
from benchmark_scoreformer import DATASETS, preprocess_dataset, train_and_evaluate
from benchmark_scoreformer import MODELS

def main():
    """Test all models on LastFM dataset"""
    # Set dataset to LastFM
    dataset_name = "lastfm"
    
    print(f"\n{'='*100}")
    print(f"Testing all models on dataset: {dataset_name}")
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
    
    # Define hyperparameters for all models
    hyperparams = {
        "Scoreformer": {
            'num_layers': 2,
            'd_model': 64,
            'num_heads': 4,
            'd_feedforward': 128,
            'input_dim': 32,
            'dropout': 0.2,
            'use_transformer': True,
            'use_dng': True,
            'use_weights': True,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 3
        },
        "CFUIcA": {
            'embedding_dim': 64,
            'attention_dim': 32,
            'dropout': 0.2,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 3
        },
        "NCF": {
            'embedding_dim': 64,
            'layers': [128, 64, 32],
            'dropout': 0.2,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 3
        },
        "MFBias": {
            'embedding_dim': 64,
            'dropout': 0.2,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 3
        },
        "AutoRec": {
            'hidden_dim': 128,
            'dropout': 0.2,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 3
        },
        "DMF": {
            'user_layers': [128, 64, 32],
            'item_layers': [128, 64, 32],
            'dropout': 0.2,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 3
        }
    }
    
    # Select models to test - use only a subset for faster testing
    # test_models = ["MFBias", "AutoRec", "DMF"]  # Uncomment to test fewer models
    test_models = ["Scoreformer", "CFUIcA", "NCF", "MFBias", "AutoRec", "DMF"]
    
    # Results storage
    results = {}
    
    # Test each model
    for model_name in test_models:
        print(f"\n{'-'*50}")
        print(f"Testing {model_name} on {dataset_name}")
        print(f"{'-'*50}")
        
        if model_name not in MODELS:
            print(f"Model {model_name} not found. Skipping.")
            continue
            
        model_class = MODELS[model_name]
        model_params = hyperparams.get(model_name, {})
        
        try:
            # Train and evaluate the model
            model_results = train_and_evaluate(model_class, model_name, dataset_name, model_params)
            results[model_name] = model_results
            
            # Print results
            print(f"\nResults for {model_name}:")
            if "error" in model_results:
                print(f"  Error: {model_results['error']}")
            else:
                for metric, value in model_results.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
        
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    # Save results to file
    with open(f"results/lastfm_all_models.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Print summary table
    print("\n\n=== LastFM Benchmark Results ===")
    print(f"{'Model':<15} {'HR@10':<10} {'NDCG@10':<10} {'Precision@10':<15} {'Recall@10':<10}")
    print("-" * 60)
    
    for model_name in test_models:
        if model_name in results:
            model_results = results[model_name]
            if "error" in model_results:
                print(f"{model_name:<15} {'ERROR':<10} {'ERROR':<10} {'ERROR':<15} {'ERROR':<10}")
            else:
                hr = model_results.get("HR@10", 0.0)
                ndcg = model_results.get("NDCG@10", 0.0)
                precision = model_results.get("Precision@10", 0.0)
                recall = model_results.get("Recall@10", 0.0)
                print(f"{model_name:<15} {hr:<10.4f} {ndcg:<10.4f} {precision:<15.4f} {recall:<10.4f}")
        else:
            print(f"{model_name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<15} {'N/A':<10}")

if __name__ == "__main__":
    main() 