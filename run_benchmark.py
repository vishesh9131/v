#!/usr/bin/env python
"""
Run the recommendation system benchmark and generate visualizations.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import time

def run_benchmark():
    """Run the benchmark script"""
    print("Running benchmark...")
    start_time = time.time()
    subprocess.run(["python", "benchmark_scoreformer.py"], check=True)
    elapsed_time = time.time() - start_time
    print(f"Benchmark completed in {elapsed_time:.2f} seconds")

def generate_visualizations():
    """Generate visualizations from benchmark results"""
    print("Generating visualizations...")
    
    # Check if results exist
    if not os.path.exists("results/benchmark_results.json"):
        print("No benchmark results found. Run the benchmark first.")
        return
    
    # Load the results
    with open("results/benchmark_results.json", "r") as f:
        results = json.load(f)
    
    # Create plots directory
    os.makedirs("results/plots", exist_ok=True)
    
    # Create DataFrame for plotting
    plot_data = []
    
    for dataset_name, dataset_results in results.items():
        for model_name, metrics in dataset_results.items():
            if "error" not in metrics:
                # Extract metrics
                for metric in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]:
                    if metric in metrics:
                        plot_data.append({
                            "Dataset": dataset_name,
                            "Model": model_name,
                            "Metric": metric,
                            "Value": metrics[metric]
                        })
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Only proceed if we have data
    if len(plot_df) == 0:
        print("No valid results for visualization.")
        return
    
    # Set the style
    sns.set(style="whitegrid")
    
    # 1. Bar charts comparing all models for each dataset and metric
    metrics_to_plot = ["HR@10", "NDCG@10"]
    
    for dataset in plot_df["Dataset"].unique():
        for metric in metrics_to_plot:
            # Filter data
            data = plot_df[(plot_df["Dataset"] == dataset) & (plot_df["Metric"] == metric)]
            
            if len(data) == 0:
                continue
                
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x="Model", y="Value", hue="Model", data=data, palette="viridis", legend=False)
            
            # Customize the plot
            plt.title(f"{metric} Comparison on {dataset}", fontsize=16)
            plt.ylabel(metric, fontsize=14)
            plt.xlabel("", fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=12)
            plt.tight_layout()
            
            # Add values on bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.4f}", 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=10, rotation=0)
            
            # Save the figure
            plt.savefig(f"results/plots/{dataset}_{metric.replace('@', '_')}.png", dpi=300, bbox_inches="tight")
            plt.close()
    
    # 2. Heatmap of all metrics for each dataset
    for dataset in plot_df["Dataset"].unique():
        # Get data for this dataset
        data = plot_df[plot_df["Dataset"] == dataset]
        
        if len(data) == 0:
            continue
            
        # Pivot the data for the heatmap
        pivot_data = data.pivot_table(index="Model", columns="Metric", values="Value")
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5)
        
        plt.title(f"Performance Metrics Heatmap - {dataset}", fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"results/plots/{dataset}_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # 3. Model comparison across datasets
    for metric in metrics_to_plot:
        # Filter data
        data = plot_df[plot_df["Metric"] == metric]
        
        if len(data) == 0:
            continue
            
        plt.figure(figsize=(14, 8))
        sns.barplot(x="Model", y="Value", hue="Dataset", data=data, palette="Set2")
        
        plt.title(f"{metric} Comparison Across Datasets", fontsize=16)
        plt.ylabel(metric, fontsize=14)
        plt.xlabel("", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.legend(title="Dataset", fontsize=12)
        plt.tight_layout()
        
        plt.savefig(f"results/plots/cross_dataset_{metric.replace('@', '_')}.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    print("Visualizations saved to results/plots/")

def main():
    """Main function"""
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    
    # Parse arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Run recommendation system benchmark and visualize results")
    parser.add_argument("--no-run", action="store_true", help="Skip running the benchmark")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations only")
    
    args = parser.parse_args()
    
    if args.visualize:
        generate_visualizations()
    elif args.no_run:
        generate_visualizations()
    else:
        run_benchmark()
        generate_visualizations()

if __name__ == "__main__":
    main() 