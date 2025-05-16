import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
from Scoreformer import Scoreformer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate small synthetic dataset for testing
def generate_synthetic_data(num_users=200, num_items=100, num_interactions=5000):
    np.random.seed(42)
    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    ratings = np.random.normal(3.5, 1.0, num_interactions)
    ratings = np.clip(ratings, 1.0, 5.0)
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    # Ensure each user and item has at least one interaction
    for user in range(num_users):
        if user not in df['user_id'].values:
            item = np.random.randint(0, num_items)
            rating = np.random.normal(3.5, 1.0)
            rating = np.clip(rating, 1.0, 5.0)
            df = df.append({'user_id': user, 'item_id': item, 'rating': rating}, ignore_index=True)
    
    for item in range(num_items):
        if item not in df['item_id'].values:
            user = np.random.randint(0, num_users)
            rating = np.random.normal(3.5, 1.0)
            rating = np.clip(rating, 1.0, 5.0)
            df = df.append({'user_id': user, 'item_id': item, 'rating': rating}, ignore_index=True)
    
    return df

# Evaluate model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            user_ids, item_ids, ratings = [b.to(device) for b in batch]
            outputs = model.predict(user_ids, item_ids)
            loss = criterion(outputs, ratings)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

# Test function for recommendation
def recommend_items(model, user_id, num_items, top_k=10):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id] * num_items, dtype=torch.long).to(device)
        item_tensor = torch.tensor(range(num_items), dtype=torch.long).to(device)
        
        # Predict ratings
        ratings = model.predict(user_tensor, item_tensor)
        
        # Get top-k items
        _, indices = torch.topk(ratings, k=top_k)
        top_items = indices.cpu().numpy()
        
        return top_items

# Main function
def main():
    # Generate data
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    # Preprocess data
    print("Preprocessing data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Convert data to tensors
    train_users = torch.tensor(train_df['user_id'].values, dtype=torch.long)
    train_items = torch.tensor(train_df['item_id'].values, dtype=torch.long)
    train_ratings = torch.tensor(train_df['rating'].values, dtype=torch.float)
    
    test_users = torch.tensor(test_df['user_id'].values, dtype=torch.long)
    test_items = torch.tensor(test_df['item_id'].values, dtype=torch.long)
    test_ratings = torch.tensor(test_df['rating'].values, dtype=torch.float)
    
    # Create data loaders
    train_dataset = TensorDataset(train_users, train_items, train_ratings)
    test_dataset = TensorDataset(test_users, test_items, test_ratings)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set model parameters
    num_users = df['user_id'].max() + 1
    num_items = df['item_id'].max() + 1
    
    # Initialize models
    print("\nInitializing Scoreformer model...")
    scoreformer = Scoreformer(
        num_layers=4,
        d_model=256,
        num_heads=8,
        d_feedforward=512,
        input_dim=128,
        num_targets=1,
        num_users=num_users,
        num_items=num_items,
        dropout=0.15,
        use_transformer=True,
        use_dng=True,
        use_weights=True
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(scoreformer.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # Train model
    print("\nTraining Scoreformer model...")
    num_epochs = 10
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        scoreformer.train()
        train_loss = 0
        
        for batch in train_loader:
            user_ids, item_ids, ratings = [b.to(device) for b in batch]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = scoreformer.predict(user_ids, item_ids)
            loss = criterion(outputs, ratings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluate on test set
        test_loss = evaluate_model(scoreformer, test_loader, criterion)
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(scoreformer.state_dict(), "best_scoreformer.pt")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss:.4f}")
    
    # Load best model
    scoreformer.load_state_dict(torch.load("best_scoreformer.pt"))
    
    # Make recommendations for a few users
    print("\nMaking recommendations for sample users...")
    for user_id in range(5):
        top_items = recommend_items(scoreformer, user_id, num_items, top_k=10)
        print(f"Top 10 recommendations for User {user_id}: {top_items}")
    
    print("\nModel evaluation complete!")

if __name__ == "__main__":
    main() 