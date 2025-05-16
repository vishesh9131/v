MAX_U = 500
TOP_N = 10
CHUNK_S = 100
TRAIN_SAMP = 5000
BATCH_SIZE = 1024
NUM_EPOCHS = 1
NUM_NEG = 400
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
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import random
from Scoreformer import *

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class REES46Dataset:
    def __init__(self, data_path=None, max_users=500000):
        if data_path is None:
            data_path = './REES46/direct_msg/'
        
        print(f"Initializing REES46Dataset with path: {data_path}")
        
        self.data_path = data_path
        self.max_users = max_users
        
        # Load campaign data with specific columns
        print("Loading campaigns...")
        self.campaigns_df = pd.read_csv(
            os.path.join(data_path, 'campaigns.csv'),
            usecols=['id', 'campaign_type', 'channel', 'topic']
        )
        
        # Load messages with only necessary columns and limit rows
        print("Loading messages...")
        self.messages_df = pd.read_csv(
            os.path.join(data_path, 'messages-demo.csv'),
            usecols=['client_id', 'campaign_id', 'is_opened', 'is_clicked'],
            nrows=5000000,  # Increased from 100000 to 5M
            low_memory=False
        )
        
        # Load events if needed
        # print("Loading events...")
        # self.events_df = pd.read_csv(os.path.join(data_path, 'events.csv'))
        
        # Process campaign types and channels
        self.campaign_types = self.campaigns_df['campaign_type'].unique()
        self.channels = self.campaigns_df['channel'].unique()
        self.topics = self.campaigns_df['topic'].dropna().unique()
        
        # Create user and campaign mappings
        print("Creating user and campaign mappings...")
        self.user_map = {uid: idx for idx, uid in enumerate(self.messages_df['client_id'].unique())}
        self.campaign_map = {cid: idx for idx, cid in enumerate(self.campaigns_df['id'].unique())}
        
        self.n_users = len(self.user_map)
        self.n_campaigns = len(self.campaign_map)
        
        print(f"Loaded {self.n_users} users and {self.n_campaigns} campaigns")
        
        # Create interaction matrices
        self.create_interaction_matrices()

    def create_interaction_matrices(self):
        """Create interaction matrices for different behaviors"""
        print("Creating interaction matrices...")
        
        # Define behavior types
        self.behaviors = ['sent', 'opened', 'clicked']
        self.behavior_matrices = {}
        
        # Pre-allocate matrices
        for behavior in self.behaviors:
            self.behavior_matrices[behavior] = np.zeros((self.n_users, self.n_campaigns))
        
        # Process messages in chunks
        chunk_size = 100000  # Increased chunk size
        total_rows = 0
        max_rows = 5000000  # Increased from 1M to 5M
        
        for chunk in pd.read_csv(
            os.path.join(self.data_path, 'messages-demo.csv'),
            usecols=['client_id', 'campaign_id', 'is_opened', 'is_clicked'],
            chunksize=chunk_size,
            low_memory=False
        ):
            # Process each behavior
            for _, row in chunk.iterrows():
                user_idx = self.user_map.get(row['client_id'])
                campaign_idx = self.campaign_map.get(row['campaign_id'])
                if user_idx is not None and campaign_idx is not None:
                    # All messages in the dataset were sent
                    self.behavior_matrices['sent'][user_idx, campaign_idx] = 1
                    if row['is_opened']:
                        self.behavior_matrices['opened'][user_idx, campaign_idx] = 1
                    if row['is_clicked']:
                        self.behavior_matrices['clicked'][user_idx, campaign_idx] = 1
            
            total_rows += len(chunk)
            print(f"Processed {total_rows} messages...")
            
            if total_rows >= max_rows:
                print(f"Reached maximum number of messages ({max_rows})")
                break
        
        # Print statistics
        for behavior in self.behaviors:
            interactions = int(self.behavior_matrices[behavior].sum())
            print(f"Created {behavior} matrix with {interactions} interactions")
            print(f"Density: {interactions/(self.n_users * self.n_campaigns):.4%}")

    def prepare_train_instances(self, max_samples=5000, users=None):
        print("Preparing training instances...")
        train_data = []
        
        if users is None:
            users = np.random.choice(
                list(range(min(self.n_users, self.max_users))),
                size=min(self.max_users, self.n_users),
                replace=False
            )
        
        for user in users:
            # Get positive campaigns (clicked)
            pos_campaigns = np.where(self.behavior_matrices['clicked'][user] > 0)[0]
            
            if len(pos_campaigns) > 0:
                for campaign in pos_campaigns:
                    # Get behavior features
                    behaviors = [float(self.behavior_matrices[b][user, campaign]) 
                               for b in self.behaviors]
                    
                    # Add positive instance
                    train_data.append([user, campaign, 1.0] + behaviors)
                    
                    # Sample negative campaigns
                    neg_campaigns = np.random.choice(
                        list(set(range(self.n_campaigns)) - set(pos_campaigns)),
                        size=NUM_NEG,
                        replace=False
                    )
                    
                    # Add negative instances
                    for neg_campaign in neg_campaigns:
                        behaviors = [float(self.behavior_matrices[b][user, neg_campaign]) 
                                  for b in self.behaviors]
                        train_data.append([user, neg_campaign, 0.0] + behaviors)
            
            if len(train_data) >= max_samples:
                break
        
        train_data = np.array(train_data)
        print(f"Generated {len(train_data)} training instances")
        print(f"Number of unique users: {len(set(train_data[:,0]))}")
        return train_data

    def get_test_instances(self, num_neg_samples=99, users=None):
        """Generate test instances with harder negative samples"""
        print("Generating test instances...")
        test_data = []
        
        if users is None:
            users = np.random.choice(
                list(range(min(self.n_users, self.max_users))),
                size=min(self.max_users // 5, self.n_users),
                replace=False
            )
        
        for user in users:
            # Get positive campaigns (clicked)
            pos_campaigns = np.where(self.behavior_matrices['clicked'][user] > 0)[0]
            
            if len(pos_campaigns) > 0:
                # Sample only a subset of positive items for testing
                test_pos = np.random.choice(pos_campaigns, size=min(3, len(pos_campaigns)), replace=False)
                
                for pos_campaign in test_pos:
                    # Add positive instance
                    behaviors = [float(self.behavior_matrices[b][user, pos_campaign]) 
                               for b in self.behaviors]
                    test_data.append([user, pos_campaign, 1.0] + behaviors)
                    
                    # Initialize neg_from_sent as empty list
                    neg_from_sent = []
                    
                    # Sample negative items (prefer items that were sent but not clicked)
                    sent_not_clicked = np.where(
                        (self.behavior_matrices['sent'][user] > 0) & 
                        (self.behavior_matrices['clicked'][user] == 0)
                    )[0]
                    
                    if len(sent_not_clicked) > 0:
                        neg_from_sent = np.random.choice(
                            sent_not_clicked,
                            size=min(num_neg_samples // 2, len(sent_not_clicked)),
                            replace=False
                        ).tolist()  # Convert to list for length calculation
                        
                        for neg_campaign in neg_from_sent:
                            behaviors = [float(self.behavior_matrices[b][user, neg_campaign]) 
                                      for b in self.behaviors]
                            test_data.append([user, neg_campaign, 0.0] + behaviors)
                    
                    # Sample remaining negatives from unsent campaigns
                    remaining_neg = num_neg_samples - len(neg_from_sent)
                    if remaining_neg > 0:
                        never_sent = np.where(self.behavior_matrices['sent'][user] == 0)[0]
                        if len(never_sent) > 0:
                            neg_from_unsent = np.random.choice(
                                never_sent,
                                size=min(remaining_neg, len(never_sent)),
                                replace=False
                            )
                            
                            for neg_campaign in neg_from_unsent:
                                behaviors = [float(self.behavior_matrices[b][user, neg_campaign]) 
                                          for b in self.behaviors]
                                test_data.append([user, neg_campaign, 0.0] + behaviors)
        
        test_data = np.array(test_data)
        print(f"Generated {len(test_data)} test instances")
        print(f"Positive instances: {np.sum(test_data[:, 2] > 0)}")
        print(f"Negative instances: {np.sum(test_data[:, 2] == 0)}")
        return test_data

def create_adjacency_matrix(batch_size, device):
    """Create identity matrix as default adjacency matrix"""
    return torch.eye(batch_size, device=device)

def create_graph_metrics(batch_size, input_dim, device):
    """Create default graph metrics"""
    return torch.zeros((batch_size, input_dim), device=device)

def evaluate_model(model, dataset, test_instances, k=10):
    """Improved evaluation function"""
    model.eval()
    hits = []
    ndcgs = []
    
    # Group test instances by user
    user_test_items = {}
    for inst in test_instances:
        user = int(inst[0])
        item = int(inst[1])
        label = float(inst[2])
        if user not in user_test_items:
            user_test_items[user] = {'pos': [], 'neg': [], 'behaviors': {}}
        if label > 0:
            user_test_items[user]['pos'].append(item)
        else:
            user_test_items[user]['neg'].append(item)
        user_test_items[user]['behaviors'][item] = inst[3:]
    
    print(f"Evaluating {len(user_test_items)} users...")
    
    with torch.no_grad():
        for user, items in user_test_items.items():
            if not items['pos'] or not items['neg']:
                continue
            
            # Prepare input for all items
            all_items = items['pos'] + items['neg']
            behaviors = [items['behaviors'][item] for item in all_items]
            
            x = torch.FloatTensor(behaviors).to(device)
            x = torch.cat([
                x,
                torch.zeros(len(x), model.input_dim - x.size(1)).to(device)
            ], dim=1)
            
            # Create adjacency matrix and graph metrics
            batch_size = len(all_items)
            adj_matrix = create_adjacency_matrix(batch_size, device)
            graph_metrics = create_graph_metrics(batch_size, model.input_dim, device)
            
            # Get predictions
            predictions = model(x, adj_matrix, graph_metrics).squeeze()
            predictions = predictions.cpu().numpy()
            
            # Calculate metrics
            item_scores = list(zip(all_items, predictions))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_items = [x[0] for x in item_scores]
            
            # HR@k
            hit = 0
            for pos_item in items['pos']:
                if pos_item in ranked_items[:k]:
                    hit = 1
                    break
            hits.append(hit)
            
            # NDCG@k
            dcg = 0
            idcg = sum([1/np.log2(i+2) for i in range(min(k, len(items['pos'])))])
            for i, item in enumerate(ranked_items[:k]):
                if item in items['pos']:
                    dcg += 1/np.log2(i+2)
            if idcg > 0:
                ndcgs.append(dcg/idcg)
            else:
                ndcgs.append(0)
    
    hr = np.mean(hits)
    ndcg = np.mean(ndcgs)
    
    print(f"\nEvaluation Statistics:")
    print(f"Total users evaluated: {len(hits)}")
    print(f"Number of hits: {sum(hits)}")
    
    return hr, ndcg

def main():
    config = {
        'd_model': 16,
        'num_heads': 2,
        'num_layers': 1,
        'd_feedforward': 32,
        'input_dim': 64,
        'num_weights': 3,
        'learning_rate': 0.001,
        'weight_decay': 1e-3,
        'dropout': 0.5,
        'gradient_clip': 0.5,
        'max_samples': MAX_U,
        'eval_k': TOP_N,
        'patience': 3
    }
    
    print("Initializing REES46 dataset...")
    dataset = REES46Dataset(
        data_path='./REES46/direct_msg/',
        max_users=MAX_U
    )
    
    # Split users into train/test sets
    all_users = np.array(list(range(dataset.n_users)))
    np.random.shuffle(all_users)
    train_users = all_users[:int(0.8 * len(all_users))]
    test_users = all_users[int(0.8 * len(all_users)):]
    
    print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")
    
    train_data = dataset.prepare_train_instances(max_samples=config['max_samples'], users=train_users)
    print(f"Using {len(train_data)} training instances")

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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )

    print("Starting training...")
    try:
        best_hr = 0
        best_ndcg = 0
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            num_batches = 0
            
            # Create batches
            indices = np.random.permutation(len(train_data))
            for start_idx in range(0, len(indices), BATCH_SIZE):
                batch_indices = indices[start_idx:start_idx + BATCH_SIZE]
                batch_data = train_data[batch_indices]
                
                users = batch_data[:, 0]
                items = batch_data[:, 1]
                labels = torch.FloatTensor(batch_data[:, 2]).to(device)
                behaviors = torch.FloatTensor(batch_data[:, 3:]).to(device)
                
                x = torch.cat([
                    behaviors,
                    torch.zeros(len(behaviors), model.input_dim - behaviors.size(1)).to(device)
                ], dim=1)
                
                adj_matrix = create_adjacency_matrix(len(x), device)
                graph_metrics = create_graph_metrics(len(x), model.input_dim, device)
                
                optimizer.zero_grad()
                predictions = model(x, adj_matrix, graph_metrics).squeeze()
                loss = criterion(predictions, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Evaluate on test set
            test_instances = dataset.get_test_instances(users=test_users)
            hr, ndcg = evaluate_model(model, dataset, test_instances, k=config['eval_k'])
            
            # Update learning rate
            scheduler.step(hr)
            
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"HR@{config['eval_k']}: {hr:.4f}")
            print(f"NDCG@{config['eval_k']}: {ndcg:.4f}")
            
            # Early stopping
            if hr > best_hr:
                best_hr = hr
                best_ndcg = ndcg
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hr': hr,
                    'ndcg': ndcg,
                }, 'best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                print("Early stopping triggered!")
                break
        
        print("\nTraining completed!")
        print(f"Best HR@{config['eval_k']}: {best_hr:.4f}")
        print(f"Best NDCG@{config['eval_k']}: {best_ndcg:.4f}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 