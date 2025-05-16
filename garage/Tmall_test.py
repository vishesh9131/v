MAX_U = 5000
TOP_N = 10
CHUNK_S = 700
TRAIN_SAMP = 5000
BATCH_SIZE = 1024
NUM_EPOCHS = 2
NUM_NEG = 4  
SEED = 42

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn import functional as F
from torch.nn import ModuleList
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from datetime import datetime
from scipy.sparse import csr_matrix
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
from Scoreformer import * 
from sklearn.model_selection import train_test_split

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MultiBehaviorDataset:
    def __init__(self, max_users=None):
        self.max_users = max_users
        self.user_pos_items = {}  # Dictionary to store positive items for each user
        self.user_behaviors = {}  # Dictionary to store behavior data for each user
        self.num_items = 0  # Will be set when loading data
        
        # Load and process the data
        self.data = self.load_data()
        
        # Process the loaded data
        for user_id, item_id, label, view, cart, purchase, favorite in self.data:
            if label > 0:  # Positive interaction
                if user_id not in self.user_pos_items:
                    self.user_pos_items[user_id] = set()
                self.user_pos_items[user_id].add(item_id)
            
            if user_id not in self.user_behaviors:
                self.user_behaviors[user_id] = {'view': 0, 'cart': 0, 'purchase': 0, 'favorite': 0}
            self.user_behaviors[user_id]['view'] += view
            self.user_behaviors[user_id]['cart'] += cart
            self.user_behaviors[user_id]['purchase'] += purchase
            self.user_behaviors[user_id]['favorite'] += favorite
            
            # Update number of items
            self.num_items = max(self.num_items, item_id + 1)
        
        print(f"Total instances: {len(self.data)}")
        
    def load_data(self):
        """
        Create a synthetic dataset with more complex patterns
        Returns:
            list of (user_id, item_id, label, view, cart, purchase, favorite) tuples
        """
        np.random.seed(SEED)  # For reproducibility
        data = []
        num_users = self.max_users if self.max_users else 1000
        num_items = 5000
        
        # Create user groups with different behavior patterns
        for user_id in range(num_users):
            # User preference vector (determines likelihood of interacting with items)
            user_preference = np.random.normal(0.5, 0.2)
            
            # Number of items this user will interact with
            num_interactions = np.random.randint(5, 50)
            
            # Generate positive interactions
            for _ in range(num_interactions):
                item_id = np.random.randint(0, num_items)
                
                # Generate behavior counts with some correlation
                base_interest = np.random.random()
                view = np.random.poisson(10 * base_interest)
                cart = np.random.poisson(3 * base_interest)
                purchase = np.random.poisson(1 * base_interest)
                favorite = np.random.poisson(2 * base_interest)
                
                # Determine label based on user preference and behaviors
                label_prob = user_preference * (0.3 + 0.7 * (
                    0.4 * (view > 0) + 
                    0.3 * (cart > 0) + 
                    0.2 * (purchase > 0) + 
                    0.1 * (favorite > 0)
                ))
                label = 1 if np.random.random() < label_prob else 0
                
                data.append((user_id, item_id, label, view, cart, purchase, favorite))
            
            # Generate some negative samples
            num_negatives = np.random.randint(num_interactions // 2, num_interactions)
            for _ in range(num_negatives):
                item_id = np.random.randint(0, num_items)
                # Add occasional noise in behavior for negative samples
                view = np.random.poisson(1) if np.random.random() < 0.1 else 0
                cart = np.random.poisson(0.5) if np.random.random() < 0.05 else 0
                purchase = 0  # No purchases for negative samples
                favorite = np.random.poisson(0.5) if np.random.random() < 0.05 else 0
                
                data.append((user_id, item_id, 0, view, cart, purchase, favorite))
        
        # Convert to numpy array for easier slicing
        data = np.array(data)
        
        # Shuffle the data
        np.random.shuffle(data)
        
        return data
    
    def prepare_train_instances(self, max_samples=None):
        """
        Prepare training instances
        Args:
            max_samples: maximum number of samples to return
        Returns:
            list of training instances
        """
        if max_samples:
            return self.data[:max_samples]
        return self.data
    
    def sample_test_batch(self, test_instances, num_negatives):
        """
        Sample negative items for each test instance
        Args:
            test_instances: list of test instances (user, pos_item) pairs
            num_negatives: number of negative items to sample for each positive item
        Returns:
            users: tensor of user ids
            pos_items: tensor of positive item ids
            neg_items: tensor of negative item ids [batch_size, num_negatives]
        """
        users = []
        pos_items = []
        neg_items = []
        
        for user, pos_item in test_instances:
            users.append(user)
            pos_items.append(pos_item)
            
            # Get user's positive items
            user_pos = self.user_pos_items.get(user, set())
            
            # Sample negative items
            neg_samples = []
            while len(neg_samples) < num_negatives:
                neg_item = np.random.randint(0, self.num_items)
                # Check if this item is not a positive interaction
                if neg_item not in user_pos:
                    neg_samples.append(neg_item)
            
            neg_items.append(neg_samples)
        
        # Convert to tensors
        users = torch.LongTensor(users).to(device)
        pos_items = torch.LongTensor(pos_items).to(device)
        neg_items = torch.LongTensor(neg_items).to(device)
        
        return users, pos_items, neg_items

    def get_test_instances(self):
        """
        Get test instances
        Returns:
            list of test instances (user, pos_item) pairs
        """
        test_instances = []
        for user_id in self.user_pos_items:
            # Get positive items for this user
            pos_items = list(self.user_pos_items[user_id])
            # Use last item as test item
            if pos_items:
                test_instances.append((user_id, pos_items[-1]))
        
        return test_instances

    def create_adjacency_matrix(self, users):
        """
        Create adjacency matrix for a batch of users
        Args:
            users: tensor of user ids [batch_size]
        Returns:
            adjacency matrix [batch_size, batch_size]
        """
        batch_size = len(users)
        adj_matrix = torch.zeros((batch_size, batch_size), device=users.device)
        
        # Convert users tensor to list for indexing
        users = users.cpu().tolist()
        
        # Create adjacency matrix based on common items between users
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    adj_matrix[i][j] = 1.0  # Self-connection
                    continue
                
                user1_items = self.user_pos_items.get(users[i], set())
                user2_items = self.user_pos_items.get(users[j], set())
                
                # Calculate similarity based on common items
                common_items = len(user1_items.intersection(user2_items))
                if common_items > 0:
                    # Normalize by the total number of unique items
                    total_items = len(user1_items.union(user2_items))
                    adj_matrix[i][j] = common_items / total_items
        
        return adj_matrix

    def create_graph_metrics(self, users):
        """
        Create graph metrics for a batch of users
        Args:
            users: tensor of user ids [batch_size]
        Returns:
            graph metrics tensor [batch_size, num_metrics]
        """
        batch_size = len(users)
        num_metrics = 3  # degree, clustering coefficient, and behavior diversity
        metrics = torch.zeros((batch_size, num_metrics), device=users.device)
        
        # Convert users tensor to list for indexing
        users = users.cpu().tolist()
        
        for i, user_id in enumerate(users):
            # 1. Degree (normalized by max possible connections)
            user_items = self.user_pos_items.get(user_id, set())
            degree = len(user_items) / self.num_items
            metrics[i][0] = degree
            
            # 2. Behavior diversity (number of different behavior types)
            if user_id in self.user_behaviors:
                behaviors = self.user_behaviors[user_id]
                num_behavior_types = sum(1 for v in behaviors.values() if v > 0)
                metrics[i][1] = num_behavior_types / 4.0  # Normalize by total number of behavior types
            
            # 3. Activity level (total behavior count normalized)
            if user_id in self.user_behaviors:
                behaviors = self.user_behaviors[user_id]
                total_behaviors = sum(behaviors.values())
                # Normalize by some reasonable maximum (e.g., 100)
                metrics[i][2] = min(total_behaviors / 100.0, 1.0)
        
        return metrics

    def get_user_behaviors(self, user_id):
        """
        Get behavior features for a user
        Args:
            user_id: user identifier
        Returns:
            tensor of behavior features [1, 4] (view, cart, purchase, favorite)
        """
        # Initialize behavior vector
        behaviors = torch.zeros(1, 4, device=device)  # [view, cart, purchase, favorite]
        
        # Get user's behavior data
        if user_id in self.user_behaviors:
            user_data = self.user_behaviors[user_id]
            behaviors[0][0] = user_data['view']
            behaviors[0][1] = user_data['cart']
            behaviors[0][2] = user_data['purchase']
            behaviors[0][3] = user_data['favorite']
        
        # Normalize behaviors (optional)
        max_val = max(1.0, behaviors.max().item())  # Avoid division by zero
        behaviors = behaviors / max_val
        
        return behaviors

def evaluate_model(model, dataset, test_instances, k=10, batch_size=1000, num_negatives=100, debug=False):
    model.eval()
    hits = []
    ndcgs = []
    
    for i in range(0, len(test_instances), batch_size):
        batch = test_instances[i:i + batch_size]
        users, pos_items, neg_items = dataset.sample_test_batch(batch, num_negatives)
        
        with torch.no_grad():
            all_items = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
            behaviors = []
            
            # Get user behaviors and ensure correct dimensions
            for user in users:
                beh = dataset.get_user_behaviors(user.item())
                # Remove any extra dimensions and ensure 2D
                beh = beh.squeeze()  # Remove any extra dimensions
                if beh.dim() == 1:
                    beh = beh.unsqueeze(0)  # Make it 2D if it's 1D
                # Add padding
                beh = torch.cat([
                    beh,
                    torch.zeros(1, model.input_dim - beh.size(1), device=device)
                ], dim=1)
                behaviors.append(beh)
            
            # Stack behaviors and ensure correct shape
            behaviors = torch.cat(behaviors, dim=0).to(device)  # [batch_size, input_dim]
            
            batch_size_current, num_samples = all_items.size()
            
            # Reshape behaviors for all items
            behaviors_repeated = behaviors.unsqueeze(1)  # [batch_size, 1, input_dim]
            behaviors_repeated = behaviors_repeated.expand(batch_size_current, num_samples, -1)  # [batch_size, num_samples, input_dim]
            behaviors_repeated = behaviors_repeated.reshape(-1, behaviors.size(-1))  # [batch_size * num_samples, input_dim]
            
            # Create adjacency matrix
            adj_matrix = dataset.create_adjacency_matrix(users)  # [batch_size, batch_size]
            adj_repeated = adj_matrix.unsqueeze(1)  # [batch_size, 1, batch_size]
            adj_repeated = adj_repeated.expand(batch_size_current, num_samples, -1)  # [batch_size, num_samples, batch_size]
            adj_repeated = adj_repeated.reshape(-1, adj_matrix.size(-1))  # [batch_size * num_samples, batch_size]
            
            # Create graph metrics
            graph_metrics = dataset.create_graph_metrics(users)  # [batch_size, metric_dim]
            graph_metrics_repeated = graph_metrics.unsqueeze(1)  # [batch_size, 1, metric_dim]
            graph_metrics_repeated = graph_metrics_repeated.expand(batch_size_current, num_samples, -1)  # [batch_size, num_samples, metric_dim]
            graph_metrics_repeated = graph_metrics_repeated.reshape(-1, graph_metrics.size(-1))  # [batch_size * num_samples, metric_dim]
            
            # Get predictions
            predictions = model(behaviors_repeated, adj_repeated, graph_metrics_repeated)
            predictions = predictions.view(batch_size_current, num_samples)
            
            # Calculate metrics
            for j, (preds, items_row) in enumerate(zip(predictions, all_items)):
                _, indices = torch.topk(preds, min(k, len(preds)))
                recommended_items = items_row[indices].cpu().numpy()
                pos_item = items_row[0].item()
                
                hit = pos_item in recommended_items
                hits.append(float(hit))
                
                if hit:
                    rank = np.where(recommended_items == pos_item)[0][0] + 1
                    ndcgs.append(1 / np.log2(rank + 1))
                else:
                    ndcgs.append(0.0)
                
                if debug and j < 5:
                    print(f"User {users[j].item()} - Pos Item {pos_item}")
                    print(f"Predicted Scores: {preds.cpu().numpy()}")
                    print(f"Top-{k} Recommendations: {recommended_items}")
                    print(f"Hit: {hit}, NDCG: {ndcgs[-1]}")
                    print("-" * 50)
    
    hr = np.mean(hits) if hits else 0.0
    ndcg = np.mean(ndcgs) if ndcgs else 0.0
    
    return hr, ndcg

def train_epoch(model, train_loader, criterion, optimizer, dataset):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (users, items, labels, behaviors) in enumerate(train_loader):
        users, items = users.long(), items.long()
        labels = labels.float()
        
        x = torch.cat([
            behaviors,
            torch.zeros(len(users), model.input_dim - behaviors.size(1), device=device)
        ], dim=1)
        
        adj_matrix = dataset.create_adjacency_matrix(users)
        graph_metrics = dataset.create_graph_metrics(users)
        
        optimizer.zero_grad()
        predictions = model(x, adj_matrix, graph_metrics)
        loss = criterion(predictions.view(-1), labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches

def main():
    config = {
        'd_model': 32,
        'num_heads': 2,
        'num_layers': 1,
        'd_feedforward': 64,
        'input_dim': 64,
        'num_weights': 4,
        'learning_rate': 0.001,
        'weight_decay': 1e-6,
        'dropout': 0.1,
        'gradient_clip': 1.0,
        'max_samples': MAX_U,
        'eval_k': TOP_N
    }
    
    print("Initializing Tmall dataset...")
    dataset = MultiBehaviorDataset(max_users=MAX_U)
    
    train_data = dataset.prepare_train_instances(max_samples=config['max_samples'])
    
    # Create train loader
    train_tensor_data = TensorDataset(
        torch.LongTensor(train_data[:, 0]),
        torch.LongTensor(train_data[:, 1]),
        torch.FloatTensor(train_data[:, 2]),
        torch.FloatTensor(train_data[:, 3:])
    )
    train_loader = DataLoader(train_tensor_data, batch_size=BATCH_SIZE, shuffle=True)

    model = Scoreformer(
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_feedforward=config['d_feedforward'],
        input_dim=config['input_dim'],
        num_weights=config['num_weights'],
        dropout=config['dropout']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])

    print("Starting training...")
    try:
        best_hr = 0
        best_ndcg = 0
        patience = 3
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            start_time = datetime.now()
            
            # Train
            avg_loss = train_epoch(model, train_loader, criterion, optimizer, dataset)
            
            # Evaluate
            test_instances = dataset.get_test_instances()
            hr, ndcg = evaluate_model(model, dataset, test_instances, k=config['eval_k'])
            
            epoch_time = (datetime.now() - start_time).total_seconds()
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"HR@{config['eval_k']}: {hr:.4f}")
            print(f"NDCG@{config['eval_k']}: {ndcg:.4f}")
            print(f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if hr > best_hr:
                best_hr = hr
                best_ndcg = ndcg
                torch.save(model.state_dict(), f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        print("\nTraining completed!")
        print(f"Best HR@{config['eval_k']}: {best_hr:.4f}")
        print(f"Best NDCG@{config['eval_k']}: {best_ndcg:.4f}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()