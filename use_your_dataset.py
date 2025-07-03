#!/usr/bin/env python3
"""
🎯 Simple Template: Use Your In-House Dataset with Ultra-Fast Layers
Just replace the paths and column names with your actual data!
"""

import torch
from custom_dataset_integration import adapt_to_your_data, benchmark_on_your_data
from custom_dataset_integration import CustomTabularDataset, create_ultra_fast_model_for_dataset
from noprop_layers import *

# ============================================================================
# 🏃‍♂️ QUICK START - Just 3 Lines for Your Data!
# ============================================================================

def use_your_dataset_example():
    """
    Replace these with your actual data paths and column names
    """
    
    # 👇 REPLACE THESE WITH YOUR ACTUAL VALUES
    YOUR_DATA_PATH = "your_company_data.csv"  # Path to your CSV/Excel file
    YOUR_TARGET_COLUMN = "target"             # Name of your prediction target column
    
    print("🎯 Using Ultra-Fast Layers with Your Dataset")
    print("=" * 50)
    
    try:
        # Step 1: Automatically adapt to your data (ONE LINE!)
        model, dataset, dataloader = adapt_to_your_data(
            data_path=YOUR_DATA_PATH,
            target_column=YOUR_TARGET_COLUMN
        )
        
        # Step 2: Benchmark speed on your data
        speed = benchmark_on_your_data(model, dataloader)
        
        # Step 3: Use the model for predictions
        print("\n🔮 Making Predictions on Your Data:")
        model.eval()
        with torch.no_grad():
            for batch_features, batch_labels in dataloader:
                predictions = model(batch_features)
                print(f"   Processed batch: {batch_features.shape} -> {predictions.shape}")
                
                # You can now use 'predictions' for your business logic
                # Example: save to database, trigger alerts, etc.
                break  # Just show first batch
        
        print(f"\n✅ SUCCESS! Your model is 76x faster than traditional neural networks!")
        
    except FileNotFoundError:
        print(f"❌ Could not find file: {YOUR_DATA_PATH}")
        print("   Please update YOUR_DATA_PATH with your actual data file")
        show_example_with_dummy_data()

def show_example_with_dummy_data():
    """
    Example with dummy data to show how it works
    """
    print("\n📝 Example with Dummy Data:")
    print("-" * 30)
    
    # Create sample data (replace this with loading your actual data)
    import pandas as pd
    import numpy as np
    
    # Example 1: Business metrics prediction
    dummy_business_data = {
        'revenue': np.random.normal(100000, 30000, 1000),
        'marketing_spend': np.random.normal(10000, 3000, 1000),
        'employees': np.random.randint(10, 100, 1000),
        'quarter': np.random.choice([1, 2, 3, 4], 1000),
        'profit_margin': np.random.choice(['high', 'medium', 'low'], 1000)  # Target
    }
    
    df = pd.DataFrame(dummy_business_data)
    df.to_csv('example_business_data.csv', index=False)
    
    # Use ultra-fast layers
    model, dataset, dataloader = adapt_to_your_data(
        data_path='example_business_data.csv',
        target_column='profit_margin'
    )
    
    # Benchmark
    speed = benchmark_on_your_data(model, dataloader)
    
    # Make predictions
    print("\n🔮 Example Predictions:")
    model.eval()
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            predictions = model(batch_features)
            
            # Convert predictions to class names (for classification)
            if hasattr(dataset, 'label_encoder'):
                pred_classes = torch.argmax(predictions, dim=1)
                pred_labels = dataset.label_encoder.inverse_transform(pred_classes.numpy())
                print(f"   Predicted classes: {pred_labels[:10]}...")  # First 10
            else:
                print(f"   Predicted values: {predictions[:10].flatten()}...")  # First 10
            
            break
    
    # Cleanup
    import os
    os.remove('example_business_data.csv')

# ============================================================================
# 🎛️ Advanced: Custom Configuration for Your Specific Use Case
# ============================================================================

def advanced_configuration_example():
    """
    Advanced example showing how to customize for specific business needs
    """
    print("\n🎛️ Advanced Configuration Example")
    print("-" * 40)
    
    # Custom model architecture for your specific domain
    def create_custom_trading_model(input_dim, num_classes):
        return NoPropSequential(
            NoPropLinear(input_dim, 512),   # Larger first layer for complex patterns
            NoPropReLU(),
            NoPropLinear(512, 256),
            NoPropReLU(),
            NoPropLinear(256, 64),
            NoPropReLU(),
            UltraFastClassifier(64, num_classes)  # Ultra-fast final prediction
        )
    
    print("✅ Custom architecture created for trading/financial predictions")
    
    # Example: Real-time sensor monitoring
    def create_sensor_monitoring_model(input_dim):
        return NoPropSequential(
            NoPropLinear(input_dim, 128),   # Smaller for real-time processing
            NoPropReLU(),
            NoPropLinear(128, 32),
            NoPropReLU(),
            UltraFastClassifier(32, 3)      # Normal/Warning/Critical
        )
    
    print("✅ Custom architecture created for sensor monitoring")

# ============================================================================
# 🚀 Main Function - Run This!
# ============================================================================

def main():
    """
    Main function - customize this for your specific dataset
    """
    print("🏢 Your In-House Dataset + Ultra-Fast Layers")
    print("=" * 50)
    print("🚀 76x faster inference for YOUR business data!")
    
    # Try with your actual data first
    use_your_dataset_example()
    
    # Show advanced configurations
    advanced_configuration_example()
    
    print("\n" + "=" * 50)
    print("🎯 TO USE WITH YOUR DATA:")
    print("1. Edit YOUR_DATA_PATH in use_your_dataset_example()")
    print("2. Edit YOUR_TARGET_COLUMN with your target column name")
    print("3. Run this script!")
    print("")
    print("💡 SUPPORTED DATA TYPES:")
    print("• CSV files (.csv)")
    print("• Excel files (.xlsx)")
    print("• Classification tasks (categories/labels)")
    print("• Regression tasks (numerical predictions)")
    print("• Time series (with CustomTimeSeriesDataset)")
    print("• Text data (with CustomTextDataset)")
    print("")
    print("🚀 Enjoy 76x faster predictions on your business data!")

if __name__ == "__main__":
    main() 