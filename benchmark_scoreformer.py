import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import json
import urllib.request
import zipfile
import gzip
import shutil
from tqdm import tqdm
import pickle
import sys
import importlib
import scipy.sparse as sp
from Scoreformer import Scoreformer
import torch.nn.functional as F
from bench_models.CFUIcA import CFUIcA
from bench_models.STGCN import STGCN
from bench_models.NCF import NCF
from bench_models.NGCF import NGCF
from bench_models.GraphSAGE import GraphSAGE
from bench_models.MFBias import MFBias
from bench_models.AutoRec import AutoRec
from bench_models.DMF import DMF
import argparse

"""
Recommendation System Benchmark Tool

This script provides a comprehensive benchmarking framework for recommendation systems,
comparing multiple state-of-the-art algorithms on standard datasets. The following models 
are included in this benchmark:

1. Scoreformer:
   - A novel transformer-based recommendation model with Direct-Neighborhood-Graph (DNG) scoring
   - Combines transformer attention mechanisms with graph-based representations
   - Features configurable transformer depth, embedding dimensions, and attention heads
   - Modular architecture with options to enable/disable transformer and DNG components

2. CFUIcA (Collaborative Filtering with User-Item Context-aware Attention):
   - Leverages context-aware attention mechanisms to capture complex user-item interactions
   - Combines collaborative filtering with attention weights for improved recommendations
   - Particularly effective for capturing subtle preference patterns

3. STGCN (Spatial-Temporal Graph Convolutional Network):
   - Extends GCN by incorporating temporal dynamics in user-item interactions
   - Processes the evolving graph structure over multiple time steps
   - Applies temporal attention to weight the importance of different time periods
   - Well-suited for recommendation scenarios with time-varying preferences

4. NCF (Neural Collaborative Filtering):
   - Combines matrix factorization with multi-layer perceptrons for recommendation
   - Learns non-linear interactions between user and item latent features
   - Uses embeddings and deep neural networks for flexible modeling
   - Effective for both explicit and implicit feedback scenarios

5. NGCF (Neural Graph Collaborative Filtering):
   - Explicitly encodes the collaborative signal in user-item interactions
   - Exploits high-order connectivity in the user-item graph
   - Propagates embeddings through the interaction graph structure
   - Leverages both direct and higher-order user-item connections

6. GraphSAGE (Graph Sample and Aggregate):
   - Generates node embeddings by sampling and aggregating features from node neighborhoods
   - Scales to large graphs by using neighborhood sampling strategies
   - Supports different aggregation functions (mean, max, LSTM)
   - Effectively captures structural information for recommendation

7. MFBias (Matrix Factorization with Bias):
   - Classic matrix factorization approach with user and item biases
   - Decomposes the user-item interaction matrix into latent factors
   - Includes global, user and item bias terms to model rating deviations
   - Foundational approach that serves as a strong baseline

8. AutoRec (Autoencoder-based Recommendation):
   - Uses autoencoders to learn compact representations of user-item interactions
   - Reconstructs user-item vectors through a bottleneck architecture
   - Captures non-linear relationships without requiring negative sampling
   - Effective for handling sparse interaction data

9. DMF (Deep Matrix Factorization):
   - Extends matrix factorization with deep neural networks
   - Processes user and item interaction vectors with separate networks
   - Projects users and items into a common latent space
   - Captures complex non-linear relationships between users and items
   
This benchmark supports:
- Model training and evaluation on multiple datasets
- Comparison using standard metrics: HR@K and NDCG@K
- Detailed logging and result visualization
- Modular design for easy extension with new models
"""

# Create bench_models directory if it doesn't exist
os.makedirs("bench_models", exist_ok=True)

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Datasets to benchmark
DATASETS = {
    # MovieLens 100K: 100,000 ratings from 943 users on 1,682 movies
    # Contains demographic information for users (age, gender, occupation, zip)
    # Rating scale: 1-5 stars
    # Timestamp information available for temporal analysis
    # Source: GroupLens Research Project at the University of Minnesota
    "ml-100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "filename": "ml-100k.zip",
        "data_path": "data/ml-100k",
        "rating_file_path": "u.data",  # Path inside the extracted directory
        "sep": "\t",
        "header": None,
        "names": ["user_id", "item_id", "rating", "timestamp"],
        "description": "MovieLens 100K dataset with 100,000 ratings (1-5) from 943 users on 1,682 movies."
    },
    
    # MovieLens 1M: 1 million ratings from 6,000 users on 4,000 movies
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "filename": "ml-1m.zip",
        "data_path": "data/ml-1m",
        "rating_file_path": "ratings.dat",  # Path inside the extracted directory
        "sep": "::",
        "header": None,
        "names": ["user_id", "item_id", "rating", "timestamp"],
        "description": "MovieLens 1M dataset with 1 million ratings from 6,000 users on 4,000 movies."
    },
    
    # Last.FM: Music listening data
    "lastfm": {
        "url": "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
        "filename": "hetrec2011-lastfm-2k.zip",
        "data_path": "data/lastfm",
        "rating_file_path": "user_artists.dat",  # Path inside the extracted directory
        "sep": "\t",
        "header": 0,
        "names": ["user_id", "item_id", "weight"],
        "description": "Last.FM dataset with 92,834 artist listening records from 1,892 users."
    },
    
    # REES46 Ecommerce Behavior Dataset
    "rees46": {
        "url": None,  # Local file
        "filename": "events.csv",
        "data_path": "data/rees46",
        "rating_file_path": "REES46/events.csv",  # Path to the original file
        "local_file": True,  # Flag to indicate it's a local file
        "sep": ",",
        "header": 0,
        "names": ["event_time", "event_type", "product_id", "category_id", "category_code", "brand", "price", "user_id", "user_session"],
        "description": "REES46 e-commerce behavior dataset with user interactions with products."
    }
}

# Simple NCF model for comparison
class SimpleNCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, layers=[64, 32], dropout=0.1):
        super(SimpleNCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.ModuleList()
        input_size = 2 * embedding_dim
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size
        self.output_layer = nn.Linear(layers[-1], 1)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        x = torch.cat([user_emb, item_emb], dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        output = self.output_layer(x)
        return output.squeeze()
    
    def predict(self, user_indices, item_indices):
        # Handle potential tensor shape mismatch
        if len(user_indices) != len(item_indices):
            if len(user_indices) == 1:
                user_indices = user_indices.repeat(len(item_indices))
            elif len(item_indices) == 1:
                item_indices = item_indices.repeat(len(user_indices))
            else:
                # If dimensions don't match and can't be broadcast
                raise ValueError(f"Mismatched dimensions: user_indices {len(user_indices)}, item_indices {len(item_indices)}")
        
        # Ensure indices don't exceed the embedding dimensions
        user_indices = torch.clamp(user_indices, 0, self.user_embedding.num_embeddings - 1)
        item_indices = torch.clamp(item_indices, 0, self.item_embedding.num_embeddings - 1)
        
        # Process in smaller batches
        batch_size = 256
        num_samples = len(user_indices)
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch_users = user_indices[i:i+batch_size]
            batch_items = item_indices[i:i+batch_size]
            with torch.no_grad():
                batch_preds = self.forward(batch_users, batch_items)
            predictions.append(batch_preds)
        
        return torch.cat(predictions, dim=0)

# Models to compare - only using Scoreformer for now
MODELS = {
    # Scoreformer: Novel transformer-based model with Direct-Neighborhood-Graph scoring
    "Scoreformer": Scoreformer,
    
    # CFUIcA: Collaborative Filtering with User-Item Context-aware Attention
    "CFUIcA": CFUIcA,
    
    # STGCN: Spatial-Temporal Graph Convolutional Network for temporal dynamics
    "STGCN": STGCN,
    
    # NCF: Neural Collaborative Filtering with matrix factorization and MLPs
    "NCF": NCF,
    
    # NGCF: Neural Graph Collaborative Filtering using high-order connectivity
    "NGCF": NGCF,
    
    # GraphSAGE: Graph Sample and Aggregate for neighborhood embeddings
    "GraphSAGE": GraphSAGE,
    
    # MFBias: Matrix Factorization with bias terms
    "MFBias": MFBias,
    
    # AutoRec: Autoencoder-based recommendation model
    "AutoRec": AutoRec,
    
    # DMF: Deep Matrix Factorization with neural networks
    "DMF": DMF
}

# Metrics
def hit_ratio(ranked_list, ground_truth):
    """Calculate hit ratio"""
    return 1.0 if ground_truth in ranked_list else 0.0

def ndcg(ranked_list, ground_truth):
    """Calculate NDCG (Normalized Discounted Cumulative Gain)"""
    if ground_truth in ranked_list:
        index = ranked_list.index(ground_truth)
        return np.reciprocal(np.log2(index + 2))
    return 0.0

class RecommendationDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'rating': self.ratings[idx]
        }

def preprocess_dataset(dataset_name):
    """Download and preprocess a dataset"""
    dataset_info = DATASETS[dataset_name]
    
    # Create dataset directory
    os.makedirs(dataset_info["data_path"], exist_ok=True)
    
    # Handle local files differently
    if dataset_info.get("local_file", False):
        ratings_file = dataset_info["rating_file_path"]
        print(f"Using local file for {dataset_name} dataset: {ratings_file}")
    else:
        # Path to the downloaded file
        download_path = os.path.join(dataset_info["data_path"], dataset_info["filename"])
        
        # Download dataset if it doesn't exist and URL is provided
        if dataset_info["url"] and not os.path.exists(download_path):
            print(f"Downloading {dataset_name} dataset...")
            # For Yelp which requires Kaggle authentication, we'll assume manual download
            if dataset_name == "yelp":
                print("Please manually download the Yelp dataset from Kaggle and place it in data/yelp/")
                return None
            urllib.request.urlretrieve(dataset_info["url"], download_path)
        
        # Path to the actual ratings file
        ratings_file = dataset_info["rating_file_path"]
        
        # Extract if it's a zip file and we don't have the ratings file yet
        if download_path.endswith('.zip') and not os.path.exists(os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])):
            print(f"Extracting {dataset_name} dataset...")
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_info["data_path"])
            
            # For some datasets, get the correct path after extraction
            if dataset_name == "lastfm":
                ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
            elif dataset_name == "ml-100k":
                ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
            elif dataset_name == "ml-1m":
                ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
        elif download_path.endswith('.gz') and not os.path.exists(ratings_file):
            with gzip.open(download_path, 'rb') as f_in:
                with open(download_path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
    # Read ratings
    if dataset_name == "yelp":
        # Yelp requires special handling for JSON lines
        df = pd.read_json(os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"]), lines=True)
        df = df[['user_id', 'business_id', 'stars']]
        df.rename(columns={'business_id': 'item_id', 'stars': 'rating'}, inplace=True)
    elif dataset_name == "rees46":
        # Special handling for REES46 dataset
        df = pd.read_csv(ratings_file)
        # Filter for view and cart events as implicit positive feedback
        df = df[df['event_type'].isin(['view', 'cart'])]
        # Create a rating column (cart = 2, view = 1)
        df['rating'] = df['event_type'].apply(lambda x: 2 if x == 'cart' else 1)
        # Select relevant columns
        df = df[['user_id', 'product_id', 'rating']]
        # Rename columns to match our standard
        df.rename(columns={'product_id': 'item_id'}, inplace=True)
        # Take a random sample if the dataset is too large
        if len(df) > 500000:
            df = df.sample(n=500000, random_state=42)
    elif dataset_name == "lastfm":
        # Special handling for LastFM dataset
        lastfm_ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
        if not os.path.exists(lastfm_ratings_file):
            print(f"LastFM ratings file not found at expected path: {lastfm_ratings_file}")
            # Try finding it
            lastfm_files = [f for f in os.listdir(dataset_info["data_path"]) if f.endswith('.dat')]
            print(f"Available .dat files: {lastfm_files}")
            for file in lastfm_files:
                if 'user_artists' in file:
                    lastfm_ratings_file = os.path.join(dataset_info["data_path"], file)
                    print(f"Found LastFM ratings file: {lastfm_ratings_file}")
                    break
        
        df = pd.read_csv(
            lastfm_ratings_file, 
            sep=dataset_info["sep"], 
            header=dataset_info["header"],
            names=dataset_info["names"]
        )
        # Rename 'weight' to 'rating' for consistency
        df.rename(columns={'weight': 'rating'}, inplace=True)
        
        # Convert listen counts (weights) to implicit ratings
        # Normalize weights to a 1-5 scale for consistency with other datasets
        min_weight = df['rating'].min()
        max_weight = df['rating'].max()
        if max_weight > min_weight:  # Avoid division by zero
            # Use log scale for better distribution since listening counts can be highly skewed
            df['rating'] = 1 + 4 * (np.log1p(df['rating']) - np.log1p(min_weight)) / (np.log1p(max_weight) - np.log1p(min_weight))
        else:
            # If all weights are the same, set them to mid-range
            df['rating'] = 3.0
    elif dataset_name == "ml-100k" or dataset_name == "ml-1m":
        # Handle MovieLens datasets
        ml_ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
        if not os.path.exists(ml_ratings_file):
            print(f"MovieLens ratings file not found at expected path: {ml_ratings_file}")
            # Try finding the file
            ml_files = [f for f in os.listdir(dataset_info["data_path"])]
            print(f"Files in dataset directory: {ml_files}")
            
            # For ML-100K, look for 'u.data' file
            if dataset_name == "ml-100k":
                subdirs = [d for d in ml_files if os.path.isdir(os.path.join(dataset_info["data_path"], d))]
                for subdir in subdirs:
                    subdir_path = os.path.join(dataset_info["data_path"], subdir)
                    subdir_files = os.listdir(subdir_path)
                    if "u.data" in subdir_files:
                        ml_ratings_file = os.path.join(subdir_path, "u.data")
                        print(f"Found ML-100K ratings file: {ml_ratings_file}")
                        break
            
            # For ML-1M, look for 'ratings.dat' file
            elif dataset_name == "ml-1m":
                subdirs = [d for d in ml_files if os.path.isdir(os.path.join(dataset_info["data_path"], d))]
                for subdir in subdirs:
                    subdir_path = os.path.join(dataset_info["data_path"], subdir)
                    subdir_files = os.listdir(subdir_path)
                    if "ratings.dat" in subdir_files:
                        ml_ratings_file = os.path.join(subdir_path, "ratings.dat")
                        print(f"Found ML-1M ratings file: {ml_ratings_file}")
                        break
        
        df = pd.read_csv(
            ml_ratings_file, 
            sep=dataset_info["sep"], 
            header=dataset_info["header"],
            names=dataset_info["names"]
        )
    else:
        # For other datasets, read from the correct path
        df = pd.read_csv(
            ratings_file, 
            sep=dataset_info["sep"], 
            header=dataset_info["header"],
            names=dataset_info["names"]
        )
    
    # Encode user_ids and item_ids
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    df['item_id'] = item_encoder.fit_transform(df['item_id'])
    
    # Save encoders for later use
    with open(os.path.join(dataset_info["data_path"], "user_encoder.pkl"), "wb") as f:
        pickle.dump(user_encoder, f)
    with open(os.path.join(dataset_info["data_path"], "item_encoder.pkl"), "wb") as f:
        pickle.dump(item_encoder, f)
    
    # Create adjacency matrix for graph algorithms
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    
    # Create user-item interaction matrix
    user_item_matrix = sp.coo_matrix(
        (np.ones(len(df)), (df['user_id'].values, df['item_id'].values)),
        shape=(num_users, num_items)
    ).tocsr()
    
    # Save processed data
    processed_data = {
        'df': df,
        'user_item_matrix': user_item_matrix,
        'num_users': num_users,
        'num_items': num_items
    }
    
    with open(os.path.join(dataset_info["data_path"], "processed_data.pkl"), "wb") as f:
        pickle.dump(processed_data, f)
    
    return processed_data

def get_train_test_data(df, test_size=0.2):
    """Split data into train and test sets"""
    # Group by user to ensure each user has test items
    user_groups = df.groupby('user_id')
    train_data = []
    test_data = []
    
    for user_id, group in user_groups:
        # For each user, sample test_size of their interactions for testing
        # Make sure each user has at least 2 interactions to allow for train/test split
        if len(group) < 2:
            # If user has only one interaction, keep it in training
            train_data.append(group)
            continue
            
        n_test = max(1, int(test_size * len(group)))
        test_indices = np.random.choice(group.index, n_test, replace=False)
        train_indices = list(set(group.index) - set(test_indices))
        
        train_data.append(df.loc[train_indices])
        test_data.append(df.loc[test_indices])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    return train_df, test_df

def evaluate_model(model, test_df, user_item_matrix, num_items, top_k=10, model_name="", max_items_per_user=500):
    """Evaluate a model using HR@K and NDCG@K metrics with more robust methodology"""
    print(f"Evaluating {model_name} with {top_k} recommendations per user")
    
    # Dictionary to store metrics at multiple cutoff points
    metrics = {}
    for k in [5, 10, 20]:
        metrics[f'HR@{k}'] = []
        metrics[f'NDCG@{k}'] = []
        metrics[f'Precision@{k}'] = []
        metrics[f'Recall@{k}'] = []
    
    # Group test data by user for more efficient evaluation
    test_users = test_df['user_id'].unique()
    
    # Use a reasonable sample for evaluation
    if len(test_users) > 200:
        print(f"Sampling 200 users from {len(test_users)} for evaluation")
        np.random.seed(42)  # For reproducibility
        test_users = np.random.choice(test_users, 200, replace=False)
    
    error_count = 0
    max_errors = 15  # Increased tolerance for errors
    
    # Process users in parallel or sequentially
    progress_bar = tqdm(test_users, desc=f"Evaluating {model_name}")
    
    for user_id in progress_bar:
        try:
            # Get ground truth items for this user
            user_test_items = test_df[test_df['user_id'] == user_id]['item_id'].values
            
            if len(user_test_items) == 0:
                continue  # Skip users with no test items
            
            # Get items that the user has not interacted with (in training set)
            user_interactions = user_item_matrix[user_id].indices
            all_items = np.arange(num_items)
            candidate_items = np.setdiff1d(all_items, user_interactions)
            
            # If there are too many candidate items, use a hybrid sampling approach
            if len(candidate_items) > max_items_per_user:
                # Create a candidate pool:
                # 1. Include all ground truth items
                # 2. Include some popular items (top 20%)
                # 3. Include some random items from the rest
                
                # Get item popularity
                item_popularity = np.array(user_item_matrix.sum(axis=0)).flatten()
                item_ranks = np.argsort(-item_popularity)  # Descending order
                
                # Top 20% popular items (excluding already interacted ones)
                num_popular = min(int(max_items_per_user * 0.2), num_items // 5)
                popular_candidates = [item for item in item_ranks[:num_popular] if item not in user_interactions]
                
                # Random items to fill the rest
                remaining_candidates = [item for item in candidate_items if item not in popular_candidates and item not in user_test_items]
                num_random = max_items_per_user - len(popular_candidates) - len(user_test_items)
                
                if num_random > 0 and len(remaining_candidates) > 0:
                    random_candidates = np.random.choice(
                        remaining_candidates, 
                        size=min(num_random, len(remaining_candidates)), 
                        replace=False
                    )
                else:
                    random_candidates = []
                
                # Combine pools
                candidate_items = np.concatenate([
                    popular_candidates,
                    random_candidates,
                    user_test_items
                ])
            
            # Make sure all ground truth items are in the candidates
            for gt_item in user_test_items:
                if gt_item not in candidate_items:
                    candidate_items = np.append(candidate_items, gt_item)
            
            # Convert to tensors
            user_tensor = torch.LongTensor([user_id]).to(device)  # Single user
            item_tensor = torch.LongTensor(candidate_items).to(device)  # All candidate items
            
            # Get predictions in smaller batches to prevent dimension issues
            all_scores = []
            batch_size = 100
            
            for i in range(0, len(item_tensor), batch_size):
                batch_items = item_tensor[i:i+batch_size]
                batch_users = user_tensor.repeat(len(batch_items))
                
                # Make sure both tensors have the same length
                assert len(batch_users) == len(batch_items), "Tensor length mismatch"
                
                with torch.no_grad():  # No gradients needed for evaluation
                    try:
                        batch_scores = model.predict(batch_users, batch_items)
                        all_scores.append(batch_scores)
                    except RuntimeError as e:
                        print(f"Batch error for user {user_id}, items {i}:{i+batch_size}: {e}")
                        # Use zeros as fallback scores for this batch
                        all_scores.append(torch.zeros(len(batch_items), device=device))
            
            # Combine all batches
            scores = torch.cat(all_scores)
            
            # Make sure scores are valid
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(f"NaN or Inf scores detected for user {user_id}, skipping")
                continue
            
            # Sort items by score
            _, indices = torch.sort(scores, descending=True)
            ranked_items = candidate_items[indices.cpu().numpy()]
            
            # Calculate metrics at multiple cutoff points
            for k in [5, 10, 20]:
                if k > len(ranked_items):
                    continue
                    
                # Get top-k items
                recommended_items = ranked_items[:k]
                
                # Hit Ratio - if any ground truth item is in recommendations
                hit = any(item in recommended_items for item in user_test_items)
                metrics[f'HR@{k}'].append(float(hit))
                
                # NDCG - normalized discounted cumulative gain
                dcg = 0
                idcg = 0
                
                # Calculate DCG
                for i, item in enumerate(recommended_items):
                    if item in user_test_items:
                        # Rank is 0-indexed, so +1 for the formula
                        dcg += 1 / np.log2(i + 2)
                
                # Calculate IDCG
                ideal_ranking = min(k, len(user_test_items))
                for i in range(ideal_ranking):
                    idcg += 1 / np.log2(i + 2)
                
                # Avoid division by zero
                if idcg > 0:
                    ndcg = dcg / idcg
                else:
                    ndcg = 0
                
                metrics[f'NDCG@{k}'].append(ndcg)
                
                # Precision - ratio of relevant items among recommendations
                relevant_count = sum(1 for item in recommended_items if item in user_test_items)
                precision = relevant_count / k
                metrics[f'Precision@{k}'].append(precision)
                
                # Recall - ratio of relevant items that were recommended
                recall = relevant_count / len(user_test_items) if len(user_test_items) > 0 else 0
                metrics[f'Recall@{k}'].append(recall)
                
        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
            error_count += 1
            if error_count >= max_errors:
                print(f"Too many errors encountered ({error_count}). Stopping evaluation.")
                if all(len(metrics[f'HR@{k}']) == 0 for k in [5, 10, 20]):
                    # Return zeros if no metrics were calculated
                    return {metric: 0.0 for metric in metrics.keys()}
                break
            continue
    
    # Calculate aggregate metrics
    results = {}
    for metric, values in metrics.items():
        if values:
            results[metric] = float(np.mean(values))
        else:
            results[metric] = 0.0
    
    # Print summary
    print(f"\nEvaluation results for {model_name}:")
    for k in [5, 10, 20]:
        print(f"  @{k}: HR={results.get(f'HR@{k}', 'N/A'):.4f}, NDCG={results.get(f'NDCG@{k}', 'N/A'):.4f}, " +
              f"Precision={results.get(f'Precision@{k}', 'N/A'):.4f}, Recall={results.get(f'Recall@{k}', 'N/A'):.4f}")
    
    return results

def train_and_evaluate(model_class, model_name, dataset_name, hyperparams):
    """Train and evaluate a model on a dataset"""
    print(f"Training {model_name} on {dataset_name}")
    
    # Load processed data
    dataset_info = DATASETS[dataset_name]
    with open(os.path.join(dataset_info["data_path"], "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    df = data['df']
    user_item_matrix = data['user_item_matrix']
    num_users = data['num_users']
    num_items = data['num_items']
    
    # Get hyperparameters
    num_epochs = hyperparams.get('num_epochs', 20)
    batch_size = hyperparams.get('batch_size', 256)
    lr = hyperparams.get('learning_rate', 0.001)
    weight_decay = hyperparams.get('weight_decay', 0.00001)
    patience = hyperparams.get('patience', 5)
    
    # Add timestamp-based features if available
    if 'timestamp' in df.columns:
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        # Extract time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
    
    # Split data into train, validation, and test sets - use stratified sampling by user
    # to ensure each user has items in each split, but handle cases where stratification fails
    try:
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, 
                                          stratify=df['user_id'] if len(df) > 10000 else None)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42,
                                        stratify=temp_df['user_id'] if len(temp_df) > 10000 else None)
    except ValueError as e:
        # If stratification fails, fall back to non-stratified split
        print(f"Stratification failed, using non-stratified split: {e}")
        train_df, temp_df = get_train_test_data(df, test_size=0.3)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create train user-item matrix
    train_user_item = sp.coo_matrix(
        (np.ones(len(train_df)), (train_df['user_id'].values, train_df['item_id'].values)),
        shape=(num_users, num_items)
    ).tocsr()
    
    # Initialize model with hyperparameters
    if model_name == "Scoreformer":
        model = model_class(
            num_layers=hyperparams.get('num_layers', 4),
            d_model=hyperparams.get('d_model', 256),
            num_heads=hyperparams.get('num_heads', 8),
            d_feedforward=hyperparams.get('d_feedforward', 512),
            input_dim=hyperparams.get('input_dim', 128),
            num_targets=1,
            num_users=num_users,
            num_items=num_items,
            dropout=hyperparams.get('dropout', 0.15),
            use_transformer=hyperparams.get('use_transformer', True),
            use_dng=hyperparams.get('use_dng', True),
            use_weights=hyperparams.get('use_weights', True)
        ).to(device)
    elif model_name == "CFUIcA":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            attention_dim=hyperparams.get('attention_dim', 64),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "NCF":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            layers=hyperparams.get('layers', [256, 128, 64]),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "NGCF":
        # Create a properly normalized adjacency matrix for NGCF
        ngcf_adj_matrix = NGCF.create_adj_matrix(train_user_item)  # Use train matrix only
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            adj_matrix=ngcf_adj_matrix,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            layers=hyperparams.get('layers', [128, 64]),
            node_dropout=hyperparams.get('node_dropout', 0.2),
            mess_dropout=hyperparams.get('mess_dropout', 0.2)
        ).to(device)
    elif model_name == "GraphSAGE":
        # Create adjacency lists from user-item matrix
        adj_lists = GraphSAGE.create_adj_lists(train_user_item)  # Use train matrix only
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            adj_lists=adj_lists,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            aggregator_type=hyperparams.get('agg_type', 'mean'),
            num_sample=hyperparams.get('num_samples', 10),
            num_layers=hyperparams.get('num_layers', 2),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "STGCN":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            adj_matrix=train_user_item,  # Use train matrix only
            embedding_dim=hyperparams.get('embedding_dim', 128),
            num_time_steps=hyperparams.get('num_time_steps', 3),
            num_layers=hyperparams.get('num_layers', 2),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "MFBias":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "AutoRec":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            hidden_dim=hyperparams.get('hidden_dim', 256),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
        
        # AutoRec requires the item rating matrix to be set
        # Convert sparse matrix to dense with safe indexing
        try:
            # Transpose the matrix to have items as rows (I-AutoRec approach)
            item_user_matrix = train_user_item.transpose().toarray()
            if item_user_matrix.shape[0] > 0 and item_user_matrix.shape[1] > 0:
                item_ratings = torch.FloatTensor(item_user_matrix).to(device)
                model.set_item_rating_matrix(item_ratings)
            else:
                print(f"Warning: Empty user-item matrix for AutoRec on {dataset_name}")
                return {"error": "Empty user-item matrix for AutoRec"}
        except Exception as e:
            print(f"Error setting up AutoRec item rating matrix: {e}")
            return {"error": f"Failed to initialize AutoRec: {e}"}
    elif model_name == "DMF":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            user_layers=hyperparams.get('user_layers', [256, 128, 64]),
            item_layers=hyperparams.get('item_layers', [256, 128, 64]),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
        
        # DMF requires the interaction matrices to be set
        model.set_interaction_matrices(train_user_item)
    else:
        print(f"Model {model_name} not implemented. Skipping.")
        return {"error": "Model not implemented"}
    
    # Create a simplified version of the dataset to make training easier
    class SimpleDataset(Dataset):
        def __init__(self, df):
            self.user_ids = df['user_id'].values
            self.item_ids = df['item_id'].values
            self.ratings = df['rating'].values
            
        def __len__(self):
            return len(self.user_ids)
        
        def __getitem__(self, idx):
            return {
                'user_id': self.user_ids[idx],
                'item_id': self.item_ids[idx],
                'rating': self.ratings[idx]
            }
    
    # Create training dataset using the simple approach
    train_dataset = SimpleDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation dataset 
    val_dataset = SimpleDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True
    )
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    bpr_criterion = nn.BCEWithLogitsLoss()  # For ranking loss
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training monitoring
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {model_name} with {num_epochs} epochs")
    
    # Track error rate to abort if too many consecutive errors
    consecutive_errors = 0
    max_consecutive_errors = 50  # Abort if this many consecutive errors
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            ratings = batch['rating'].float().to(device)
            
            # Forward pass - rating prediction loss
            try:
                # Compute MSE loss for rating prediction
                predictions = model.predict(user_ids, item_ids)
                rating_loss = mse_criterion(predictions.view(-1), ratings)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                rating_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                total_loss += rating_loss.item()
                batch_count += 1
                consecutive_errors = 0  # Reset error counter on success
            except Exception as e:
                consecutive_errors += 1
                print(f"Error during training batch {batch_idx}: {e}")
                
                # Abort if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Too many consecutive errors ({consecutive_errors}). Aborting training.")
                    # If we have at least some successful batches, continue to evaluation
                    if batch_count > 0:
                        print("Some batches were successful. Proceeding to evaluation.")
                        break
                    else:
                        print("No successful batches. Unable to train model.")
                        return {"error": f"Training failed: {e}"}
                        
                continue
        
        # Skip if no batches were processed successfully
        if batch_count == 0:
            print("No batches were processed in this epoch, skipping validation")
            continue
            
        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        consecutive_errors = 0  # Reset for validation
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")):
                user_ids = batch['user_id'].to(device)
                item_ids = batch['item_id'].to(device)
                ratings = batch['rating'].float().to(device)
                
                try:
                    predictions = model.predict(user_ids, item_ids)
                    batch_loss = mse_criterion(predictions.view(-1), ratings)
                    val_loss += batch_loss.item()
                    val_batch_count += 1
                    consecutive_errors = 0  # Reset on success
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Error during validation batch {batch_idx}: {e}")
                    
                    # Abort if too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Too many consecutive validation errors. Skipping remainder of validation.")
                        break
                    continue
        
        # Skip if no validation batches were processed
        if val_batch_count == 0:
            print("No validation batches were processed, using training loss for early stopping")
            avg_val_loss = avg_train_loss
        else:
            avg_val_loss = val_loss / val_batch_count
            
        val_losses.append(avg_val_loss)
            
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Update learning rate based on validation performance
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), os.path.join("models", f"{model_name}_{dataset_name}_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model for evaluation
    try:
        model.load_state_dict(torch.load(os.path.join("models", f"{model_name}_{dataset_name}_best.pt")))
    except Exception as e:
        print(f"Warning: Could not load best model, using current model instead. Error: {e}")
    
    # Save trained model
    torch.save(model.state_dict(), os.path.join("models", f"{model_name}_{dataset_name}.pt"))
    
    # Evaluate model with improved evaluation procedure
    model.eval()
    results = evaluate_model(
        model, 
        test_df, 
        user_item_matrix, 
        num_items, 
        model_name=model_name,
        max_items_per_user=500  # Increased from 200 to 500 for more thorough evaluation
    )
    
    # Save results
    result_path = os.path.join("results", f"{model_name}_{dataset_name}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    """Run the benchmark"""
    # Datasets to benchmark
    dataset_names = ["ml-100k", "ml-1m", "lastfm", "rees46"]
    
    # For faster testing, you can use a subset of datasets
    # Uncomment the next line to test with fewer datasets
    # dataset_names = ["ml-100k", "rees46"]  # Use only these datasets for quicker testing
    
    # Models to benchmark - define a subset for faster running
    model_names = ["Scoreformer", "CFUIcA", "NCF", "MFBias", "AutoRec", "DMF"]
    # Uncomment the next line to test with fewer models
    # model_names = ["MFBias", "AutoRec"]  # Test just two models for quick results
    
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Run recommendation system benchmarks')
    parser.add_argument('--datasets', nargs='+', choices=dataset_names, help='Datasets to benchmark')
    parser.add_argument('--models', nargs='+', choices=model_names, help='Models to benchmark')
    parser.add_argument('--quick', action='store_true', help='Run a quick test (5 epochs)')
    args = parser.parse_args()
    
    # Use command line arguments if provided
    if args.datasets:
        dataset_names = args.datasets
    if args.models:
        model_names = args.models
    
    # Process all datasets
    for dataset_name in dataset_names:
        print(f"\n{'='*100}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*100}")
        preprocess_dataset(dataset_name)
    
    # Define model hyperparameters with improved settings
    hyperparams = {
        "Scoreformer": {
            'num_layers': 3,             # Increased from 4
            'd_model': 128,              # Increased from 256
            'num_heads': 4,             # Increased from 8
            'd_feedforward': 256,        # Increased from 512
            'input_dim': 64,            # Increased from 128
            'dropout': 0.2,             # Kept at optimal value
            'use_transformer': True,
            'use_dng': True,
            'use_weights': True,
            'num_epochs': 40 if not (args.quick if hasattr(args, 'quick') else False) else 10, # More epochs for better fitting
            'learning_rate': 0.0003,     # Slightly lower for better stability
            'weight_decay': 0.00001,     # L2 regularization
            'batch_size': 512,           # Keep large batch size
            'patience': 12 if not (args.quick if hasattr(args, 'quick') else False) else 5 # More patience
        },
        "CFUIcA": {
            'embedding_dim': 128,       # Increased from 64
            'attention_dim': 64,        # Increased from 32
            'dropout': 0.2,             # Slightly increased
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "NCF": {
            'embedding_dim': 128,       # Increased from 64
            'layers': [256, 128, 64],   # Larger layers
            'dropout': 0.2,             # Slightly increased
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "MFBias": {
            'embedding_dim': 64,
            'dropout': 0.2,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5, 
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "AutoRec": {
            'hidden_dim': 256,
            'dropout': 0.2,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "DMF": {
            'user_layers': [256, 128, 64],
            'item_layers': [256, 128, 64],
            'dropout': 0.2,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        }
    }
    
    # Results storage
    all_results = {}
    
    # Train and evaluate on all datasets
    for dataset_name in dataset_names:
        print(f"\n{'='*100}")
        print(f"Benchmarking on dataset: {dataset_name}")
        print(f"{'='*100}")
        
        dataset_results = {}
        
        for model_name in model_names:
            print(f"\n{'-'*50}")
            print(f"Benchmarking {model_name} on {dataset_name}")
            print(f"{'-'*50}")
            
            if model_name not in MODELS:
                print(f"Model {model_name} not found in available models. Skipping.")
                dataset_results[model_name] = {"error": "Model not available"}
                continue
                
            model_class = MODELS[model_name]
            model_params = hyperparams.get(model_name, {})
            
            try:
                eval_results = train_and_evaluate(model_class, model_name, dataset_name, model_params)
                dataset_results[model_name] = eval_results
            except Exception as e:
                print(f"Error training {model_name} on {dataset_name}: {e}")
                dataset_results[model_name] = {"error": str(e)}
        
        all_results[dataset_name] = dataset_results
    
    # Save overall results
    with open("results/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Generate performance comparison table
    create_comparison_table(all_results)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    for dataset_name in dataset_names:
        print(f"\n{dataset_name}:")
        for model_name in model_names:
            if model_name in all_results.get(dataset_name, {}):
                metrics = all_results[dataset_name][model_name]
                if "error" in metrics:
                    print(f"  {model_name}: Error - {metrics['error']}")
                else:
                    print(f"  {model_name}: HR@10={metrics.get('HR@10', 'N/A'):.4f}, NDCG@10={metrics.get('NDCG@10', 'N/A'):.4f}")
            else:
                print(f"  {model_name}: Not evaluated")

def create_comparison_table(results):
    """Create a formatted comparison table from benchmark results"""
    import pandas as pd
    
    # Extract datasets and models
    datasets = list(results.keys())
    models = []
    
    for dataset in datasets:
        models.extend(list(results[dataset].keys()))
    models = list(set(models))  # Remove duplicates
    
    # Create table structure
    table_data = []
    
    # Metrics to include in the table
    metrics = ["HR@10", "NDCG@10"]
    
    # For each dataset
    for dataset in datasets:
        dataset_results = results[dataset]
        
        # For each metric
        for metric in metrics:
            # Row with dataset and metric
            row = {"Dataset": dataset, "Metric": metric}
            
            # Base model for improvement calculation
            base_model = "MFBias"  # Use MFBias as the baseline
            base_value = None
            
            if base_model in dataset_results and "error" not in dataset_results[base_model]:
                base_value = dataset_results[base_model].get(metric, 0.0)
            
            # Get values for each model
            for model in models:
                if model in dataset_results:
                    if "error" in dataset_results[model]:
                        row[model] = "Error"
                    else:
                        value = dataset_results[model].get(metric, 0.0)
                        row[model] = f"{value:.4f}"
                        
                        # Add improvement over baseline
                        if base_value is not None and base_value > 0 and model != base_model:
                            impr = ((value - base_value) / base_value) * 100
                            row[f"{model}_impr"] = f"{impr:.2f}%"
                else:
                    row[model] = "N/A"
            
            table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    df.to_csv("results/comparison_table.csv", index=False)
    print("Comparison table saved to results/comparison_table.csv")
    
    # Format for Markdown/HTML display
    md_table = "# Performance Comparison Table\n\n"
    
    for dataset in datasets:
        md_table += f"## {dataset}\n\n"
        
        # Filter for this dataset
        dataset_df = df[df["Dataset"] == dataset]
        
        # Create table header
        header = "| Metric | " + " | ".join(models) + " |\n"
        separator = "| --- | " + " | ".join(["---" for _ in models]) + " |\n"
        
        md_table += header + separator
        
        # Add rows
        for metric in metrics:
            metric_row = dataset_df[dataset_df["Metric"] == metric].iloc[0]
            row_str = f"| {metric} | "
            
            for model in models:
                value = metric_row.get(model, "N/A")
                improvement = metric_row.get(f"{model}_impr", "")
                
                if improvement and value != "Error" and value != "N/A":
                    row_str += f"{value}<br>({improvement}) | "
                else:
                    row_str += f"{value} | "
            
            md_table += row_str + "\n"
        
        md_table += "\n"
    
    # Save markdown table
    with open("results/comparison_table.md", "w") as f:
        f.write(md_table)
    print("Markdown comparison table saved to results/comparison_table.md")

if __name__ == "__main__":
    main() 