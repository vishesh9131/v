import torch
import numpy as np
import pandas as pd
from ablation_study import load_data, train_model, validate_model
from Scoreformer import Scoreformer
import matplotlib.pyplot as plt


def detect_machine():
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    return device

def hyperparameter_study():
    train_loader, val_loader = load_data()
    input_dim = 17733
    num_targets = 5911

    # Define hyperparameter ranges
    learning_rates = [0.001, 0.003, 0.005, 0.007]
    gcn_layers = [1]
    hidden_units = [32, 128, 512]
    dropout_rates = [0.1, 0.3, 0.5, 0.7]

    results = {
        'learning_rate': [],
        'gcn_layers': [],
        'hidden_units': [],
        'dropout_rate': [],
        'HR@10': [],
        'NDCG@10': []
    }

    # Iterate over hyperparameter combinations
    for lr in learning_rates:
        for layers in gcn_layers:
            for units in hidden_units:
                for dropout in dropout_rates:
                    print(f"Testing: LR={lr}, Layers={layers}, Units={units}, Dropout={dropout}")
                    
                    # Define model with current hyperparameters
                    model = Scoreformer(
                        num_layers=layers,
                        d_model=units,
                        num_heads=4,
                        d_feedforward=128,
                        input_dim=input_dim,
                        num_targets=num_targets,
                        dropout=dropout,
                        use_transformer=True,
                        use_dng=True,
                        use_weights=True
                    )
                    
                    # Train and validate the model
                    train_model(model, train_loader, epochs=5)
                    _, _, hr, ndcg = validate_model(model, val_loader)
                    
                    # Store results
                    results['learning_rate'].append(lr)
                    results['gcn_layers'].append(layers)
                    results['hidden_units'].append(units)
                    results['dropout_rate'].append(dropout)
                    results['HR@10'].append(hr)
                    results['NDCG@10'].append(ndcg)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv("hyperparam_study_results.csv", index=False)
    print("Saved hyperparameter study results to hyperparam_study_results.csv")

    # Plot results
    plot_results(df_results)

def plot_results(df):
    # Example plot for learning rate vs HR@10
    plt.figure(figsize=(10, 6))
    for lr in df['learning_rate'].unique():
        subset = df[df['learning_rate'] == lr]
        plt.plot(subset['gcn_layers'], subset['HR@10'], label=f"LR={lr}")
    plt.xlabel('GCN Layers')
    plt.ylabel('HR@10')
    plt.title('HR@10 vs GCN Layers for Different Learning Rates')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print(detect_machine())
    hyperparameter_study() 