import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import requests
import zipfile
import io
import time
import matplotlib.pyplot as plt
import json
import importlib.util
import sys
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import torch.nn.functional as F
import gc

# Memory optimization: Set smaller sample size and batch size
MAX_SAMPLES = 100000  # Limit dataset size to reduce memory usage
BATCH_SIZE = 64       # Smaller batch size
EPOCHS = 3            # Fewer epochs

# Import Scoreformer
try:
    from Scoreformer import Scoreformer
except ImportError:
    print("Falling back to dump.py for Scoreformer")
    from dump import Scoreformer

# Directory setup
BASE_DIR = "benchmark_results"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

for directory in [BASE_DIR, DATA_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Metrics
def hit_ratio(recommended_items, ground_truth, k=10):
    hits = 0
    for gt in ground_truth:
        if gt in recommended_items[:k]:
            hits += 1
    return hits / len(ground_truth) if len(ground_truth) > 0 else 0

def ndcg(recommended_items, ground_truth, k=10):
    dcg = 0
    idcg = 0
    for i, item in enumerate(recommended_items[:k]):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)
    for i in range(min(len(ground_truth), k)):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0

def precision_at_k(recommended_items, ground_truth, k=10):
    hits = 0
    for item in recommended_items[:k]:
        if item in ground_truth:
            hits += 1
    return hits / k

def recall_at_k(recommended_items, ground_truth, k=10):
    hits = 0
    for item in ground_truth:
        if item in recommended_items[:k]:
            hits += 1
    return hits / len(ground_truth) if len(ground_truth) > 0 else 0

# MovieLens Dataset - Optimized for memory
class MovieLensDataset(Dataset):
    def __init__(self, ratings_file, users_file, movies_file, max_samples=None, transform=None):
        self.transform = transform
        
        # Load data
        self.ratings = pd.read_csv(ratings_file, sep='::', 
                                  names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                  engine='python')
                                  
        # Limit dataset size if specified
        if max_samples and len(self.ratings) > max_samples:
            self.ratings = self.ratings.sample(max_samples, random_state=42)
        
        self.users = pd.read_csv(users_file, sep='::', 
                                names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                                engine='python')
        self.movies = pd.read_csv(movies_file, sep='::', 
                                 names=['movie_id', 'title', 'genres'],
                                 engine='python', encoding='latin-1')
        
        # Create user and item mappings
        self.unique_users = self.ratings['user_id'].unique()
        self.unique_movies = self.ratings['movie_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(self.unique_users)}
        self.movie_to_idx = {movie: idx for idx, movie in enumerate(self.unique_movies)}
        
        self.ratings['user_idx'] = self.ratings['user_id'].apply(lambda x: self.user_to_idx.get(x, 0))
        self.ratings['movie_idx'] = self.ratings['movie_id'].apply(lambda x: self.movie_to_idx.get(x, 0))
        
        # Create adjacency matrix and graph metrics
        self.create_adjacency_matrix()
        
    def create_adjacency_matrix(self):
        # Memory optimization: Use sparse representation for large matrices
        n_users = len(self.unique_users)
        n_movies = len(self.unique_movies)
        
        # Create a simple adjacency matrix 
        # Use a smaller representation with only essential connections
        n_entities = min(n_users + n_movies, 10000)  # Limit total entities
        self.adjacency_matrix = torch.zeros((n_entities, n_entities), dtype=torch.float32)
        
        # Only add edges for items in our limited dataset
        for _, row in self.ratings.iterrows():
            user_idx = row['user_idx']
            movie_idx = row['movie_idx'] + n_users
            
            # Skip if indices exceed matrix dimensions
            if user_idx >= n_entities or movie_idx >= n_entities:
                continue
                
            rating = row['rating']
            
            # Add edge with weight based on rating
            self.adjacency_matrix[user_idx, movie_idx] = rating / 5.0
            self.adjacency_matrix[movie_idx, user_idx] = rating / 5.0
        
        # Create simple graph metrics (smaller dimension)
        self.graph_metrics = torch.zeros((n_entities, 5), dtype=torch.float32)  # Reduced from 10 to 5
        
        # Calculate simple graph metrics like degree
        degrees = self.adjacency_matrix.sum(dim=1)
        self.graph_metrics[:, 0] = degrees / (degrees.max() + 1e-8)  # Normalized degree
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        user_idx = self.ratings.iloc[idx]['user_idx']
        movie_idx = self.ratings.iloc[idx]['movie_idx']
        rating = self.ratings.iloc[idx]['rating']
        
        # Return only necessary tensors
        sample = {
            'user_idx': user_idx,
            'movie_idx': movie_idx,
            'rating': rating,
            'adjacency_matrix': self.adjacency_matrix,
            'graph_metrics': self.graph_metrics
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

# Download and prepare MovieLens-1M dataset
def download_movielens_1m():
    if os.path.exists(os.path.join(DATA_DIR, "ml-1m")):
        print("MovieLens-1M dataset already exists.")
        return
    
    print("Downloading MovieLens-1M dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    response = requests.get(url)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print("MovieLens-1M dataset downloaded successfully!")

# Baseline models
class BPR(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(BPR, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        
    def forward(self, user_idx, item_idx):
        user_embedding = self.user_embeddings(user_idx)
        item_embedding = self.item_embeddings(item_idx)
        prediction = torch.sum(user_embedding * item_embedding, dim=1)
        return prediction

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, layers=[64, 32, 16]):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers (reduced size)
        self.fc_layers = nn.ModuleList()
        input_size = embedding_dim * 2
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size
        
        self.output_layer = nn.Linear(layers[-1], 1)
        
    def forward(self, user_idx, item_idx):
        user_embedding = self.user_embedding(user_idx)
        item_embedding = self.item_embedding(item_idx)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        
        output = torch.sigmoid(self.output_layer(x))
        return output.squeeze()

# Dictionary to store all models for benchmarking
models = {
    'BPR': BPR,
    'NCF': NCF
    # Start with fewer models to test memory usage
}

# Download model implementations if available
def download_model_implementation(model_name):
    print(f"Using built-in implementation for {model_name}")
    return

# Function to train and evaluate a model
def train_and_evaluate(model_class, model_name, dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=0.001):
    print(f"Training and evaluating {model_name}...")
    
    # Split dataset into train and test (with a smaller validation set)
    train_size = int(0.8 * len(dataset))
    test_size = min(len(dataset) - train_size, 10000)  # Limit test size to 10,000
    
    # Use a subset of indices to reduce memory
    all_indices = list(range(len(dataset)))
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:train_size+test_size]
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    num_users = len(dataset.unique_users)
    num_items = len(dataset.unique_movies)
    embedding_dim = 32  # Reduced embedding dimension
    
    if model_name == 'Scoreformer':
        # Check which Scoreformer implementation we're using
        import inspect
        sig = inspect.signature(model_class.__init__)
        params = list(sig.parameters.keys())
        
        if 'num_targets' in params:
            # Scoreformer.py implementation
            model = model_class(
                num_layers=1,  # Reduced from 2
                d_model=embedding_dim,
                num_heads=4,  # Reduced from 8
                d_feedforward=64,  # Reduced from 128
                input_dim=embedding_dim,
                num_targets=1,
                dropout=0.1,
                use_transformer=True,
                use_dng=True,
                use_weights=True
            )
        else:
            # dump.py implementation
            model = model_class(
                num_layers=1,  # Reduced from 2
                d_model=embedding_dim,
                num_heads=4,  # Reduced from 8
                d_feedforward=64,  # Reduced from 128
                input_dim=embedding_dim,
                num_weights=5,  # Reduced from 10
                use_weights=True,
                dropout=0.1
            )
            
        # Create user and item embeddings for Scoreformer
        user_embedding = nn.Embedding(num_users, embedding_dim)
        item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(user_embedding.weight, std=0.1)
        nn.init.normal_(item_embedding.weight, std=0.1)
    else:
        model = model_class(num_users, num_items, embedding_dim=embedding_dim)
    
    # Move model to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    if model_name == 'Scoreformer':
        user_embedding = user_embedding.to(device)
        item_embedding = item_embedding.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Add user and item embeddings to optimizer if using Scoreformer
    if model_name == 'Scoreformer':
        optimizer.add_param_group({'params': user_embedding.parameters()})
        optimizer.add_param_group({'params': item_embedding.parameters()})
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Manual memory cleanup before each epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            user_idx = batch['user_idx'].to(device)
            movie_idx = batch['movie_idx'].to(device)
            ratings = (batch['rating'] > 3).float().to(device)  # Binary: like/dislike
            
            optimizer.zero_grad()
            
            if model_name == 'Scoreformer':
                # Get embeddings
                user_emb = user_embedding(user_idx)
                item_emb = item_embedding(movie_idx)
                
                # Create input for Scoreformer
                x = user_emb * item_emb  # Simple element-wise product as input features
                
                # Get only a slice of adjacency matrix to save memory
                adjacency_matrix = batch['adjacency_matrix'].to(device)
                graph_metrics = batch['graph_metrics'].to(device)
                
                # Handle different Scoreformer implementations
                import inspect
                sig = inspect.signature(model.forward)
                forward_params = list(sig.parameters.keys())
                
                if 'weights' in forward_params:
                    # Using dump.py implementation
                    # Dummy weights for demonstration
                    weights = torch.ones(user_idx.size(0), 5).to(device)  # Reduced from 10
                    outputs = model(x, adjacency_matrix, graph_metrics, weights)
                else:
                    # Using Scoreformer.py implementation
                    outputs = model(x, adjacency_matrix, graph_metrics)
            else:
                outputs = model(user_idx, movie_idx)
            
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Free memory after each batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}")
    
    # Evaluation
    model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            user_idx = batch['user_idx'].to(device)
            movie_idx = batch['movie_idx'].to(device)
            ratings = (batch['rating'] > 3).float().to(device)  # Binary: like/dislike
            
            if model_name == 'Scoreformer':
                # Get embeddings
                user_emb = user_embedding(user_idx)
                item_emb = item_embedding(movie_idx)
                
                # Create input for Scoreformer
                x = user_emb * item_emb
                
                adjacency_matrix = batch['adjacency_matrix'].to(device)
                graph_metrics = batch['graph_metrics'].to(device)
                
                # Handle different Scoreformer implementations
                import inspect
                sig = inspect.signature(model.forward)
                forward_params = list(sig.parameters.keys())
                
                if 'weights' in forward_params:
                    # Using dump.py implementation
                    # Dummy weights for demonstration
                    weights = torch.ones(user_idx.size(0), 5).to(device)  # Reduced from 10
                    outputs = model(x, adjacency_matrix, graph_metrics, weights)
                else:
                    # Using Scoreformer.py implementation
                    outputs = model(x, adjacency_matrix, graph_metrics)
            else:
                outputs = model(user_idx, movie_idx)
            
            predictions.extend(outputs.cpu().numpy())
            ground_truth.extend(ratings.cpu().numpy())
            
            # Free memory after each batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    # Clean up memory
    del model
    if model_name == 'Scoreformer':
        del user_embedding
        del item_embedding
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    # Calculate metrics
    k = 10
    hr_at_k = hit_ratio(predictions, ground_truth, k)
    ndcg_at_k = ndcg(predictions, ground_truth, k)
    precision = precision_at_k(predictions, ground_truth, k)
    recall = recall_at_k(predictions, ground_truth, k)
    
    metrics = {
        'HR@10': hr_at_k,
        'NDCG@10': ndcg_at_k,
        'Precision@10': precision,
        'Recall@10': recall
    }
    
    print(f"Metrics for {model_name}:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    return metrics

# Main benchmarking function
def run_benchmark():
    # Download and prepare dataset
    download_movielens_1m()
    
    # Prepare dataset
    ratings_file = os.path.join(DATA_DIR, "ml-1m", "ratings.dat")
    users_file = os.path.join(DATA_DIR, "ml-1m", "users.dat")
    movies_file = os.path.join(DATA_DIR, "ml-1m", "movies.dat")
    
    if not all(os.path.exists(f) for f in [ratings_file, users_file, movies_file]):
        print("Dataset files not found. Please check the paths.")
        return
    
    print(f"Loading dataset with max {MAX_SAMPLES} samples...")
    dataset = MovieLensDataset(ratings_file, users_file, movies_file, max_samples=MAX_SAMPLES)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Include Scoreformer
    models['Scoreformer'] = Scoreformer
    
    # Results container
    results = {}
    
    # Run benchmark for each model
    for model_name, model_class in models.items():
        if model_class is not None:  # Skip models without implementations
            try:
                metrics = train_and_evaluate(model_class, model_name, dataset)
                results[model_name] = metrics
                
                # Save results after each model to preserve progress
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = os.path.join(RESULT_DIR, f"benchmark_results_{model_name}_{timestamp}.json")
                
                with open(result_file, 'w') as f:
                    json.dump({model_name: metrics}, f, indent=4)
                
                print(f"Results for {model_name} saved to {result_file}")
                
                # Free memory
                gc.collect()
                
            except Exception as e:
                print(f"Error benchmarking {model_name}: {str(e)}")
                continue
    
    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(RESULT_DIR, f"benchmark_results_combined_{timestamp}.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate comparison table
    df_results = pd.DataFrame(columns=['Method', 'HR@10', 'NDCG@10', 'Precision@10', 'Recall@10'])
    
    for model_name, metrics in results.items():
        new_row = {
            'Method': model_name,
            'HR@10': metrics['HR@10'],
            'NDCG@10': metrics['NDCG@10'],
            'Precision@10': metrics['Precision@10'],
            'Recall@10': metrics['Recall@10']
        }
        df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save table
    table_file = os.path.join(RESULT_DIR, f"comparison_table_{timestamp}.csv")
    df_results.to_csv(table_file, index=False)
    
    print(f"Benchmark results saved to {result_file}")
    print(f"Comparison table saved to {table_file}")
    
    # Display table
    print("\nPerformance Comparison on MovieLens-1M Dataset")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)
    
    # Generate plots
    plt.figure(figsize=(12, 8))
    
    metrics = ['HR@10', 'NDCG@10', 'Precision@10', 'Recall@10']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.bar(df_results['Method'], df_results[metric])
        plt.title(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plot_file = os.path.join(RESULT_DIR, f"comparison_plot_{timestamp}.png")
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Comparison plot saved to {plot_file}")

if __name__ == "__main__":
    run_benchmark() 