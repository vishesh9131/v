import os
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import networkx as nx
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

from Scoreformer import Scoreformer, GNNLayer

# --- New Model Implementations for Comparison ---

class TrainableGNNLayer(nn.Module):
    """A GNN Layer with a trainable linear transformation, for use in GC-MC."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features, edge_index):
        source_nodes, dest_nodes = edge_index
        neighbor_features = node_features[source_nodes]
        aggregated_features = torch.zeros_like(node_features)
        aggregated_features.index_add_(0, dest_nodes, neighbor_features)
        return self.linear(aggregated_features)

class ItemKNN:
    """A classic Item-based K-Nearest Neighbors model."""
    def __init__(self, train_df, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        print("Building ItemKNN similarity matrix...")
        # Create a sparse user-item matrix
        self.user_item_matrix = lil_matrix((num_users, num_items))
        for _, row in train_df.iterrows():
            self.user_item_matrix[int(row['user_idx']), int(row['item_idx'])] = 1
        
        # Compute item-item cosine similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T.tocsr())
        # FIX: Ensure the similarity matrix is float32 to match other tensors
        self.item_similarity = torch.tensor(self.item_similarity, device=DEVICE, dtype=torch.float)

    def predict(self, user_indices, item_indices, *args, **kwargs):
        """Predict scores based on item similarity."""
        user_histories = self.user_item_matrix[user_indices.cpu().numpy()].toarray()
        user_histories = torch.tensor(user_histories, device=DEVICE, dtype=torch.float)
        
        # Score is the sum of similarities between candidate item and items in user history
        item_sims = self.item_similarity[item_indices] # [batch_size, num_all_items]
        
        # We need to compute dot product between user history and item similarities
        # This is a bit tricky with batching. A simple loop is clearer here.
        scores = []
        for i in range(len(user_indices)):
            user_history_for_item = user_histories[i] # vector of items user i has interacted with
            sim_for_candidate_item = item_sims[i] # similarity vector for candidate item i
            
            score = torch.dot(user_history_for_item, sim_for_candidate_item)
            scores.append(score)
            
        return torch.tensor(scores, device=DEVICE)

    def train(self): pass
    def eval(self): pass
    def to(self, device): return self

class AutoRec(nn.Module):
    """An Item-based Autoencoder for recommendation (I-AutoRec)."""
    def __init__(self, num_users, num_items, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Linear(num_users, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_users)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.decoder(self.activation(self.encoder(x)))

    def predict(self, user_indices, item_indices, full_user_item_matrix):
        """Reconstruct item vectors and get scores for specific users."""
        reconstructed_vectors = self.forward(full_user_item_matrix.T).T # [num_users, num_items]
        return reconstructed_vectors[user_indices, item_indices]

class GC_MC(nn.Module):
    """A simple Graph Convolutional Matrix Completion model."""
    def __init__(self, num_users, num_items, edge_index, embedding_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.edge_index = edge_index
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # Simple GNN layer for propagation
        self.gnn_layer = TrainableGNNLayer(embedding_dim, embedding_dim)

    def forward(self, user_indices, item_indices, *args, **kwargs):
        all_user_features = self.user_embedding.weight
        all_item_features = self.item_embedding.weight
        all_node_features = torch.cat([all_user_features, all_item_features], dim=0)

        propagated_features = self.gnn_layer(all_node_features, self.edge_index)
        
        prop_user_emb, prop_item_emb = torch.split(propagated_features, [self.num_users, self.num_items])
        
        user_emb = prop_user_emb[user_indices]
        item_emb = prop_item_emb[item_indices]
        
        return torch.sum(user_emb * item_emb, dim=1, keepdim=True)

    def predict(self, user_indices, item_indices, *args, **kwargs):
        return self.forward(user_indices, item_indices).squeeze(-1)

class PopularityModel:
    """A simple baseline that recommends the most popular items."""
    def __init__(self, train_df, num_items):
        self.item_counts = train_df['item_idx'].value_counts()
        self.num_items = num_items
        
        # Create a tensor of popularity scores
        self.popularity_scores = torch.zeros(num_items, device=DEVICE)
        for item_idx, count in self.item_counts.items():
            self.popularity_scores[item_idx] = count

    def predict(self, user_indices, item_indices, *args, **kwargs):
        """Returns the popularity score for each item, ignoring the user."""
        return self.popularity_scores[item_indices]

    # Add dummy methods to conform to the training script structure
    def train(self): pass
    def eval(self): pass
    def to(self, device): return self

class MatrixFactorization(nn.Module):
    """A standard Matrix Factorization model with BPR loss."""
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_indices, item_indices, *args, **kwargs):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        return torch.sum(user_emb * item_emb, dim=1, keepdim=True)
    
    def predict(self, user_indices, item_indices, *args, **kwargs):
        return self.forward(user_indices, item_indices).squeeze(-1)

class LightGCN(nn.Module):
    """A simplified LightGCN model for recommendation."""
    def __init__(self, num_users, num_items, edge_index, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.edge_index = edge_index

        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def get_final_embeddings(self):
        """Propagate embeddings through GNN layers and combine them."""
        all_embeddings = [self.embedding.weight]
        current_features = self.embedding.weight
        
        for _ in range(self.num_layers):
            # This is a simplified propagation step
            source_nodes, dest_nodes = self.edge_index
            neighbor_features = current_features[source_nodes]
            
            aggregated_features = torch.zeros_like(current_features)
            aggregated_features.index_add_(0, dest_nodes, neighbor_features)
            current_features = aggregated_features
            all_embeddings.append(current_features)
        
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        return final_user_emb, final_item_emb

    def forward(self, user_indices, item_indices, *args, **kwargs):
        # FIX: Embeddings are now calculated inside the forward pass
        final_user_emb, final_item_emb = self.get_final_embeddings()
        user_emb = final_user_emb[user_indices]
        item_emb = final_item_emb[item_indices]
        return torch.sum(user_emb * item_emb, dim=1, keepdim=True)

    def predict(self, user_indices, item_indices, *args, **kwargs):
        # FIX: Embeddings are now calculated inside the predict pass
        final_user_emb, final_item_emb = self.get_final_embeddings()
        user_emb = final_user_emb[user_indices]
        item_emb = final_item_emb[item_indices]
        return self.forward(user_indices, item_indices).squeeze(-1)

# --- Configuration ---
DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATASET_PATH = "ml-1m"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
D_FEEDFORWARD = 256
TOP_K = 10

# --- 1. Data Loading and Preparation ---

def download_and_unzip_movielens():
    """Downloads and unzips the MovieLens 1M dataset if not present."""
    if not os.path.exists(DATASET_PATH):
        print("Downloading MovieLens 1M dataset...")
        zip_filename = "ml-1m.zip"
        response = requests.get(DATASET_URL, stream=True)
        with open(zip_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        print("Unzipping dataset...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_filename)
        print("Dataset downloaded and ready.")

def load_and_process_data():
    """
    Loads ratings, creates user/item mappings, and performs a leave-one-out split.
    """
    print("Loading and processing data...")
    ratings_file = os.path.join(DATASET_PATH, 'ratings.dat')
    df = pd.read_csv(
        ratings_file,
        sep='::',
        engine='python',
        names=['userId', 'movieId', 'rating', 'timestamp']
    )

    # Encode user and item IDs to start from 0
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['userId'])
    df['item_idx'] = item_encoder.fit_transform(df['movieId'])

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()
    
    # Sort by user and timestamp to facilitate leave-one-out split
    df.sort_values(by=['user_idx', 'timestamp'], inplace=True)
    
    train_data = []
    test_data = {} # {user_idx: test_item_idx}
    
    for user_idx, group in tqdm(df.groupby('user_idx'), desc="Splitting data (leave-one-out)"):
        # The last item for each user is for testing
        test_item = group.iloc[-1]
        test_data[user_idx] = test_item['item_idx']
        
        # The rest is for training
        train_df = group.iloc[:-1]
        train_data.append(train_df)
        
    train_df = pd.concat(train_data)
    
    # Create the full interaction graph from training data
    G = nx.Graph()
    G.add_nodes_from(range(num_users), bipartite=0) # User nodes
    G.add_nodes_from(range(num_users, num_users + num_items), bipartite=1) # Item nodes
    
    # Item indices in the graph are offset by num_users
    edges = list(zip(train_df['user_idx'].values, train_df['item_idx'].values + num_users))
    G.add_edges_from(edges)
    
    print("Calculating graph structural features (PageRank)...")
    pagerank = nx.pagerank(G)
    
    # Create structural features tensor
    graph_structural_features = torch.zeros((num_users + num_items, 1), device=DEVICE)
    for node, pr_value in pagerank.items():
        graph_structural_features[node, 0] = pr_value
        
    # Create edge index for PyTorch
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous().to(DEVICE)

    return train_df, test_data, num_users, num_items, edge_index, graph_structural_features

# --- 2. Evaluation Metrics ---

def hit_ratio_at_k(predictions, true_item, k):
    """Calculates Hit Ratio @ k."""
    _, top_k_indices = torch.topk(predictions, k)
    return 1.0 if true_item in top_k_indices else 0.0

def ndcg_at_k(predictions, true_item, k):
    """Calculates NDCG @ k."""
    _, top_k_indices = torch.topk(predictions, k)
    
    if true_item in top_k_indices:
        # Find the rank of the true item
        rank = (top_k_indices == true_item).nonzero(as_tuple=True)[0].item() + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0

def evaluate_model(model, test_data, num_items, edge_index, graph_structural_features, k=10, full_user_item_matrix=None):
    """
    Evaluates the model using HR@k and NDCG@k.
    For each user, we rank the true test item against 100 random negative items.
    """
    print(f"Evaluating model: {model.__class__.__name__}...")
    model.eval()
    
    hr_list = []
    ndcg_list = []
    
    test_users = list(test_data.keys())
    
    # FIX: Removed pre-calculation of embeddings for LightGCN. The model handles it now.
    # final_user_emb_lgcn, final_item_emb_lgcn = (None, None)
    # if isinstance(model, LightGCN):
    #     final_user_emb_lgcn, final_item_emb_lgcn = model.get_final_embeddings()

    with torch.no_grad():
        for user_idx in tqdm(test_users, desc="Evaluating"):
            true_item_idx = test_data[user_idx]
            
            # Sample 100 negative items that the user has not interacted with
            negative_items = []
            while len(negative_items) < 100:
                sampled_ids = np.random.choice(num_items, size=100)
                # Ensure true item and already chosen negatives are not in the sample
                sampled_ids = [item_id for item_id in sampled_ids if item_id != true_item_idx and item_id not in negative_items]
                negative_items.extend(sampled_ids)
            negative_items = negative_items[:100]

            # Combine the true item with the negative items. True item is at index 0.
            test_item_indices = torch.tensor([true_item_idx] + negative_items, dtype=torch.long).to(DEVICE)
            user_indices = torch.full_like(test_item_indices, fill_value=user_idx)
            
            # Get predictions from the model, handling different signatures
            if isinstance(model, AutoRec):
                predictions = model.predict(user_indices, test_item_indices, full_user_item_matrix)
            else:
                predictions = model.predict(user_indices, test_item_indices, edge_index, graph_structural_features)
            
            # --- FIX: Correctly calculate rank of the true item ---
            # The true item is at index 0 in our `test_item_indices` and `predictions` tensors.
            true_item_score = predictions[0]
            
            # Count how many of the 100 negative items have a score higher than the true item
            # Add 1 to get the rank (1-based)
            rank = (predictions[1:] > true_item_score).sum().item() + 1
            
            # Calculate metrics based on the rank
            if rank <= k:
                hr_list.append(1.0)
                ndcg_list.append(1.0 / np.log2(rank + 1))
            else:
                hr_list.append(0.0)
                ndcg_list.append(0.0)
            
    return np.mean(hr_list), np.mean(ndcg_list)

# --- 3. Main Training and Evaluation Script ---

if __name__ == "__main__":
    download_and_unzip_movielens()
    train_df, test_data, num_users, num_items, edge_index, graph_structural_features = load_and_process_data()
    
    # Create the full user-item matrix required for AutoRec and ItemKNN
    full_user_item_matrix = lil_matrix((num_users, num_items))
    for _, row in train_df.iterrows():
        full_user_item_matrix[int(row['user_idx']), int(row['item_idx'])] = 1.0
    full_user_item_matrix = torch.tensor(full_user_item_matrix.toarray(), dtype=torch.float, device=DEVICE)

    print(f"\nDevice: {DEVICE}")
    print(f"Num Users: {num_users}, Num Items: {num_items}\n")

    # --- 4. Model Definitions and Comparison ---
    models_to_compare = {
        "Popularity": PopularityModel(train_df, num_items),
        "ItemKNN": ItemKNN(train_df, num_users, num_items),
        "MatrixFactorization": MatrixFactorization(num_users, num_items, embedding_dim=D_MODEL),
        "AutoRec": AutoRec(num_users=num_users, num_items=num_items, hidden_dim=D_MODEL),
        "GC_MC": GC_MC(num_users, num_items, edge_index, embedding_dim=D_MODEL),
        "LightGCN": LightGCN(num_users, num_items, edge_index, embedding_dim=D_MODEL, num_layers=3),
        "Scoreformer": Scoreformer(
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            input_dim=D_MODEL,
            num_targets=1,
            num_users=num_users,
            num_items=num_items,
            use_dng=True
        )
    }

    results = {}

    def bpr_loss(pos_scores, neg_scores):
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

    def info_nce_loss(user_emb, pos_item_emb, neg_item_emb, temperature=0.07):
        """Calculates the InfoNCE loss for contrastive learning."""
        # Calculate positive similarity
        pos_sim = F.cosine_similarity(user_emb, pos_item_emb, dim=-1)
        # Calculate negative similarities
        neg_sim = F.cosine_similarity(user_emb.unsqueeze(1), neg_item_emb, dim=-1)
        
        # Create logits
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temperature
        # Labels are always 0 (the positive similarity)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=DEVICE)
        
        return F.cross_entropy(logits, labels)

    for name, model in models_to_compare.items():
        print(f"\n{'='*20}\n--- Training and Evaluating: {name} ---\n{'='*20}")
        model = model.to(DEVICE)

        if name in ["Popularity", "ItemKNN"]:
            hr, ndcg = evaluate_model(model, test_data, num_items, edge_index, graph_structural_features, k=TOP_K, full_user_item_matrix=full_user_item_matrix)
            results[name] = {"HR@10": hr, "NDCG@10": ndcg}
            continue

        if name == "AutoRec":
            # AutoRec uses a reconstruction loss (MSE) and has a different training loop
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            criterion = nn.MSELoss()
            for epoch in range(EPOCHS):
                model.train()
                # We train on the item vectors (columns of the user-item matrix)
                item_vectors = full_user_item_matrix.T
                reconstructed = model(item_vectors)
                loss = criterion(reconstructed, item_vectors)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch+1}/{EPOCHS} - Reconstruction Loss: {loss.item():.4f}")
            
            hr, ndcg = evaluate_model(model, test_data, num_items, edge_index, graph_structural_features, k=TOP_K, full_user_item_matrix=full_user_item_matrix)
            results[name] = {"HR@10": hr, "NDCG@10": ndcg}
            continue

        if name == "Scoreformer":
            # Scoreformer has a more complex loop with a combined loss
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            contrastive_loss_alpha = 0.1 # Weight for the contrastive loss component

            for epoch in range(EPOCHS):
                model.train()
                total_loss, total_bpr, total_cl = 0, 0, 0

                users = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
                pos_items = torch.tensor(train_df['item_idx'].values, dtype=torch.long)
                permutation = torch.randperm(len(users))
                users, pos_items = users[permutation], pos_items[permutation]

                pbar = tqdm(range(0, len(users), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}")
                for i in pbar:
                    batch_users = users[i:i+BATCH_SIZE].to(DEVICE)
                    batch_pos_items = pos_items[i:i+BATCH_SIZE].to(DEVICE)
                    batch_neg_items = torch.randint(0, num_items, (len(batch_users),)).to(DEVICE)
                    
                    optimizer.zero_grad()
                    
                    # Unpack the new model output
                    pos_scores, user_emb_pos, pos_item_emb = model(batch_users, batch_pos_items, edge_index, graph_structural_features)
                    # FIX: Get negative item embeddings from the correct forward pass
                    neg_scores, _, neg_item_emb = model(batch_users, batch_neg_items, edge_index, graph_structural_features)

                    # Calculate combined loss
                    bpr = bpr_loss(pos_scores, neg_scores)
                    cl = info_nce_loss(user_emb_pos, pos_item_emb, neg_item_emb.unsqueeze(1)) # Unsqueeze for broadcasting
                    loss = bpr + contrastive_loss_alpha * cl

                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_bpr += bpr.item()
                    total_cl += cl.item()
                    pbar.set_postfix({"BPR": f"{bpr.item():.3f}", "CL": f"{cl.item():.3f}"})
                
                avg_loss = total_loss / (len(pbar) or 1)
                print(f"Epoch {epoch+1}/{EPOCHS} - Avg Combined Loss: {avg_loss:.4f}")
            
            # Evaluate BEFORE fine-tuning
            hr, ndcg = evaluate_model(model, test_data, num_items, edge_index, graph_structural_features, k=TOP_K, full_user_item_matrix=full_user_item_matrix)
            results[name] = {"HR@10": hr, "NDCG@10": ndcg}
            print(f"Evaluation for {name} (Before FT) - HR@{TOP_K}: {hr:.4f}, NDCG@{TOP_K}: {ndcg:.4f}")
            
            # --- Cold-Start Fine-Tuning Stage ---
            print("\n--- Starting Cold-Start Fine-Tuning for Scoreformer ---")
            user_counts = train_df['user_idx'].value_counts()
            cold_start_users = user_counts[user_counts < 5].index.tolist()
            
            if not cold_start_users:
                print("No cold-start users found. Skipping fine-tuning.")
                results[name + " (FT)"] = results[name] # Copy results
            else:
                ft_df = train_df[train_df['user_idx'].isin(cold_start_users)]
                print(f"Found {len(cold_start_users)} cold-start users with {len(ft_df)} interactions.")

                ft_epochs = 5
                for epoch in range(ft_epochs):
                    model.train()
                    ft_users = torch.tensor(ft_df['user_idx'].values, dtype=torch.long)
                    ft_pos_items = torch.tensor(ft_df['item_idx'].values, dtype=torch.long)
                    permutation = torch.randperm(len(ft_users))
                    ft_users, ft_pos_items = ft_users[permutation], ft_pos_items[permutation]

                    pbar_ft = tqdm(range(0, len(ft_users), BATCH_SIZE), desc=f"FT Epoch {epoch+1}/{ft_epochs}")
                    for i in pbar_ft:
                        batch_users = ft_users[i:i+BATCH_SIZE].to(DEVICE)
                        batch_pos_items = ft_pos_items[i:i+BATCH_SIZE].to(DEVICE)
                        batch_neg_items = torch.randint(0, num_items, (len(batch_users),)).to(DEVICE)
                        
                        optimizer.zero_grad()
                        pos_scores, _, _ = model(batch_users, batch_pos_items, edge_index, graph_structural_features)
                        neg_scores, _, _ = model(batch_users, batch_neg_items, edge_index, graph_structural_features)
                        loss = bpr_loss(pos_scores, neg_scores) # Using simpler BPR loss for stable fine-tuning
                        loss.backward()
                        optimizer.step()
                        pbar_ft.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                # Evaluate AFTER fine-tuning
                hr_ft, ndcg_ft = evaluate_model(model, test_data, num_items, edge_index, graph_structural_features, k=TOP_K, full_user_item_matrix=full_user_item_matrix)
                results[name + " (FT)"] = {"HR@10": hr_ft, "NDCG@10": ndcg_ft}
                print(f"Evaluation for {name} (After FT) - HR@{TOP_K}: {hr_ft:.4f}, NDCG@{TOP_K}: {ndcg_ft:.4f}")

            continue # Skip to the next model in the main loop

        # Standard BPR training loop for MF, GC-MC, LightGCN
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            
            users = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
            pos_items = torch.tensor(train_df['item_idx'].values, dtype=torch.long)
            permutation = torch.randperm(len(users))
            users, pos_items = users[permutation], pos_items[permutation]

            pbar = tqdm(range(0, len(users), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}")
            for i in pbar:
                batch_users = users[i:i+BATCH_SIZE].to(DEVICE)
                batch_pos_items = pos_items[i:i+BATCH_SIZE].to(DEVICE)
                batch_neg_items = torch.randint(0, num_items, (len(batch_users),)).to(DEVICE)
                
                optimizer.zero_grad()
                
                pos_scores = model(batch_users, batch_pos_items, edge_index, graph_structural_features)
                neg_scores = model(batch_users, batch_neg_items, edge_index, graph_structural_features)

                loss = bpr_loss(pos_scores, neg_scores)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            print(f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {total_loss / (len(users) // BATCH_SIZE):.4f}")
        
        hr, ndcg = evaluate_model(model, test_data, num_items, edge_index, graph_structural_features, k=TOP_K, full_user_item_matrix=full_user_item_matrix)
        results[name] = {"HR@10": hr, "NDCG@10": ndcg}
        print(f"Final Evaluation for {name} - HR@{TOP_K}: {hr:.4f}, NDCG@{TOP_K}: {ndcg:.4f}")

    # --- 5. Print Final Results Table ---
    print("\n\n" + "="*50)
    print("---           FINAL MODEL COMPARISON           ---")
    print(f"{'Model':<25} | {'HR@10':<10} | {'NDCG@10':<10}")
    print("-"*50)
    for name, metrics in results.items():
        hr = metrics.get('HR@10', -1)
        ndcg = metrics.get('NDCG@10', -1)
        print(f"{name:<25} | {hr:<10.4f} | {ndcg:<10.4f}")
    print("="*50) 