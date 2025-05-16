import pandas as pd
import torch
from Scoreformer import Scoreformer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix
import torch.nn as nn
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
from sklearn.metrics import roc_auc_score
import datetime
import random
from synthetic_dataset import SyntheticDataset  # Import the new SyntheticDataset class


class MultiBehaviorDataset:
    def __init__(self, data_path, max_users=50000):
        self.data_path = data_path
        self.max_users = max_users
        self.load_data()
    
    def load_data(self):
        print(f"Loading data from {self.data_path}")
        
        # Load the koubei data
        train_path = os.path.join(self.data_path, 'ijcai2016_koubei_train')
        test_path = os.path.join(self.data_path, 'ijcai2016_koubei_test')
        
        koubei_train = pd.read_csv(train_path)
        koubei_test = pd.read_csv(test_path)
        
        # Combine train and test data
        all_data = pd.concat([koubei_train, koubei_test])
        
        # Create interaction matrix
        self.trn_mats = []
        mat = self._create_interaction_matrix(all_data)
        self.trn_mats.append(mat)  # Use same matrix for all behaviors for now
        self.trn_mats.append(mat)
        self.trn_mats.append(mat)
        
        self.trn_label = 1 * (self.trn_mats[-1] != 0)
        self.n_users, self.n_items = self.trn_mats[0].shape
        print(f"Loaded dataset with {self.n_users} users and {self.n_items} items")
    
    def _create_interaction_matrix(self, df):
        # Create user-item interaction matrix
        users = df['use_ID'].unique()
        items = df['mer_ID'].unique()
        
        # Limit users if needed
        if len(users) > self.max_users:
            users = users[:self.max_users]
            df = df[df['use_ID'].isin(users)]
        
        # Create user and item mappings
        user_idx = {u: i for i, u in enumerate(users)}
        item_idx = {i: j for j, i in enumerate(items)}
        
        # Filter and create arrays ensuring all elements exist in mappings
        valid_interactions = df[df['use_ID'].isin(user_idx.keys()) & df['mer_ID'].isin(item_idx.keys())]
        
        row = [user_idx[u] for u in valid_interactions['use_ID']]
        col = [item_idx[i] for i in valid_interactions['mer_ID']]
        data = [1] * len(row)
        
        # Verify arrays have the same length
        assert len(row) == len(col) == len(data), "Arrays must have the same length"
        
        return csr_matrix((data, (row, col)), shape=(len(users), len(items)))

def load_data():
    # Create an instance of the SyntheticDataset
    synthetic_data = SyntheticDataset(num_users=1000, num_items=500, interaction_density=0.05)
    
    # Get DataLoader objects for training and validation
    train_loader, val_loader = synthetic_data.get_dataloader(batch_size=32, train_split=0.8)
    
    return train_loader, val_loader

# Calculate NDCG
def calculate_ndcg(predictions, ground_truth, k=10):
    # Ensure the predictions and ground truth are numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Get the number of samples
    n_samples = len(predictions)
    ndcg_scores = []
    
    for i in range(n_samples):
        # Get top k indices
        top_k_idx = np.argsort(-predictions[i])[:k]
        
        # Calculate DCG
        dcg = np.sum(ground_truth[i][top_k_idx] / np.log2(np.arange(2, k + 2)))
        
        # Calculate IDCG
        ideal_idx = np.argsort(-ground_truth[i])[:k]
        idcg = np.sum(ground_truth[i][ideal_idx] / np.log2(np.arange(2, k + 2)))
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    # Return average NDCG
    return np.mean(ndcg_scores)

# Calculate HR
def calculate_hr(predictions, ground_truth, k=10):
    # Ensure the predictions and ground truth are numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Get the number of samples
    n_samples = len(predictions)
    hit_rates = []
    
    for i in range(n_samples):
        # Get top k indices
        top_k_idx = np.argsort(-predictions[i])[:k]
        
        # Calculate hit rate
        hit = int(np.sum(ground_truth[i][top_k_idx]) > 0)
        hit_rates.append(hit)
    
    # Return average hit rate
    return np.mean(hit_rates)

# Evaluate a model variant (kept here for reference, not used in this script)
def evaluate_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            try:
                batch_size = x_batch.size(0)
                x_batch = x_batch.reshape(batch_size, -1)
                # Create dummy adjacency matrix and graph metrics
                adj_matrix = torch.eye(batch_size)
                graph_metrics = torch.zeros(batch_size, 3)
                
                # Forward pass – model output is expected to be [batch_size, num_targets]
                output = model(x_batch, adj_matrix, graph_metrics)
                
                all_predictions.extend(output.cpu().numpy())
                all_ground_truth.extend(y_batch.cpu().numpy())
            except Exception as e:
                print(f"Error during model evaluation: {e}")
                return 0.0, 0.0
    
    predictions = np.array(all_predictions)
    ground_truth = np.array(all_ground_truth)
    
    ndcg = calculate_ndcg(predictions, ground_truth)
    hr = calculate_hr(predictions, ground_truth)
    
    return ndcg, hr

def train_model(model, dataloader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Adjust the unpacking to match the number of elements in the batch
                if len(batch) == 2:
                    x_batch, y_batch = batch
                elif len(batch) == 3:
                    x_batch, y_batch, _ = batch  # Assuming the third element is not needed
                else:
                    raise ValueError("Unexpected number of elements in batch")
                
                # Add feature dimension if it doesn't exist
                if len(x_batch.shape) == 1:
                    x_batch = x_batch.unsqueeze(1)  # Shape becomes [32, 1]
                
                # Convert to float and ensure correct shape
                x_batch = x_batch.float()
                
                batch_size = x_batch.size(0)
                
                # Create dummy adjacency matrix and graph metrics
                adj_matrix = torch.eye(batch_size)
                graph_metrics = torch.zeros(batch_size, 3)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(x_batch, adj_matrix, graph_metrics)
                
                # Reshape target tensor
                y_batch = y_batch.float().unsqueeze(1)  # Shape becomes [32, 1]
                
                # Compute loss
                loss = criterion(output, y_batch)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            except Exception as e:
                print(f"Error during training: {e}")
                print(f"Shapes - x: {x_batch.shape}, y: {y_batch.shape}")
                if 'output' in locals():
                    print(f"Output shape: {output.shape}")
                continue
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

def validate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_ground_truth = []
    criterion = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                if len(batch) == 2:
                    x_batch, y_batch = batch
                elif len(batch) == 3:
                    x_batch, y_batch, _ = batch  # Assuming the third element is not needed
                else:
                    raise ValueError("Unexpected number of elements in batch")
                
                batch_size = x_batch.size(0)
                
                # Expand x_batch to have 2 features instead of 1
                x_batch = x_batch.float()
                if x_batch.size(1) == 1:
                    # Duplicate the feature to make it [batch_size, 2]
                    x_batch = torch.cat([x_batch, x_batch], dim=1)
                
                adj_matrix = torch.eye(batch_size)
                graph_metrics = torch.zeros(batch_size, 3)
                
                output = model(x_batch, adj_matrix, graph_metrics)
                
                # Ensure y_batch has the right shape for the loss function
                y_batch = y_batch.float().view(batch_size, -1)
                
                loss = criterion(output, y_batch)
                total_loss += loss.item()
                
                all_predictions.append(output.cpu().numpy())
                all_ground_truth.append(y_batch.cpu().numpy())
            except Exception as e:
                print(f"Error during validation: {e}")
                print(f"Shapes - x: {x_batch.shape}, y: {y_batch.shape}")
                continue
    
    if not all_predictions or not all_ground_truth:
        print("No valid predictions or ground truth collected during validation")
        return 0.0, 0.0, 0.0, 0.0
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    hr = calculate_hr(predictions, ground_truth, k=10)
    ndcg = calculate_ndcg(predictions, ground_truth, k=10)
    
    try:
        # Flatten targets and predictions for AUC calculation.
        auc = roc_auc_score(ground_truth.ravel(), predictions.ravel())
    except Exception as e:
        print("Error computing AUC:", e)
        auc = 0.0
        
    return avg_loss, auc, hr, ndcg

def ablation_study():
    train_loader, val_loader = load_data()
    # Get input dimension from the training data
    for batch in train_loader:
        if len(batch) == 2:
            x_batch, _ = batch
        elif len(batch) == 3:
            x_batch, _, _ = batch
        input_dim = 2  # Set to 2 to match the expected input dimension
        break
    
    base_params = {
        'num_layers': 2,
        'd_model': 64,
        'num_heads': 4,
        'd_feedforward': 128,
        'input_dim': input_dim,
        'num_targets': 1,  # Change to 1 to match the target dimension
        'dropout': 0.1
    }
    
    # Define the desired variants with the associated flags.
    variants = {
        "Full Model": Scoreformer(**base_params, use_transformer=True, use_dng=True, use_weights=True),
        "No Transformer": Scoreformer(**base_params, use_transformer=False, use_dng=True, use_weights=True),
        "No DNG": Scoreformer(**base_params, use_transformer=True, use_dng=False, use_weights=True),
        "No Weights": Scoreformer(**base_params, use_transformer=True, use_dng=True, use_weights=False),
        "Basic Model": Scoreformer(**base_params, use_transformer=False, use_dng=False, use_weights=False)
    }
    
    results = {}
    for variant_name, model in variants.items():
        print(f"\nTraining {variant_name}...")
        train_model(model, train_loader, epochs=5)
        print(f"Validating {variant_name}...")
        val_loss, auc, hr, ndcg = validate_model(model, val_loader)
        results[variant_name] = {
            "HR@10": hr,
            "NDCG@10": ndcg,
            "Val_Loss": val_loss,
            "Val_AUC": auc,
            "use_transformer": model.use_transformer,
            "use_dng": model.use_dng,
            "use_weights": model.use_weights
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    df_results = pd.DataFrame.from_dict(results, orient="index")
    print("\nAblation Study Results:")
    print(df_results)
    
    # Save the results with a timestamp in the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"ablation_results/ablation_results_{timestamp}.csv"
    df_results.to_csv(csv_filename)
    print("Saved ablation study results to", csv_filename)
    return df_results

def run_ablation_experiment(num_runs=3):
    all_results = {}  # Dictionary to accumulate results

    for run in range(num_runs):
        print(f"\nStarting run {run+1} of {num_runs}")
        # Optionally vary the seed for each run for extra robustness.
        # For example, seed = 42 + run, and then reset the seed as shown above.
        
        # Set up the seed here if you choose to vary it.
        seed = 42 + run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        results = ablation_study()
        for variant, metrics in results.to_dict("index").items():
            if variant not in all_results:
                all_results[variant] = {key: [] for key in metrics.keys()}
            for key, value in metrics.items():
                all_results[variant][key].append(value)

    # Compute averages.
    avg_results = {}
    for variant, metrics in all_results.items():
        avg_results[variant] = {key: np.mean(values) for key, values in metrics.items()}
    
    df_avg_results = pd.DataFrame.from_dict(avg_results, orient="index")
    print("\nAverage Ablation Study Results over multiple runs:")
    print(df_avg_results)
    return df_avg_results

if __name__ == "__main__":
    run_ablation_experiment(num_runs=3)