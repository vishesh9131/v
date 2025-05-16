import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class REES46Dataset(Dataset):
    def __init__(self, data_path, max_users=5000):
        self.data_path = data_path
        self.max_users = max_users
        
        # Load and preprocess data
        self.load_data()
        self.split_data()
        
    def load_data(self):
        # Correct the file path
        file_path = os.path.join(self.data_path, 'campaigns.csv')
        self.df = pd.read_csv(file_path)
        
        # Check the column names
        print("Columns in CSV:", self.df.columns)
        
        # Identify the correct columns for users and items
        # Replace 'user_id' and 'item_id' with actual column names if available
        user_col = 'id'  # Example placeholder, replace with actual column name
        item_col = 'topic'  # Example placeholder, replace with actual column name
        
        # Check if the columns exist
        if user_col not in self.df.columns or item_col not in self.df.columns:
            raise ValueError("The specified user or item columns do not exist in the CSV file.")
        
        # Limit users if specified
        if self.max_users:
            self.df = self.df[self.df[user_col] < self.max_users]
        
        # Create user and item mappings
        self.user_mapping = {uid: idx for idx, uid in enumerate(self.df[user_col].unique())}
        self.item_mapping = {iid: idx for idx, iid in enumerate(self.df[item_col].unique())}
        
        # Map IDs to indices
        self.df['user_idx'] = self.df[user_col].map(self.user_mapping)
        self.df['item_idx'] = self.df[item_col].map(self.item_mapping)
        
    def split_data(self):
        # Split data into train, validation, and test sets
        train, test = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_instances = train.values
        self.test_instances = test.values
        
    def prepare_train_instances(self, max_samples=None):
        """Return training instances."""
        if max_samples:
            return self.train_instances[:max_samples]
        return self.train_instances
    
    def get_test_instances(self):
        """Return test instances."""
        return self.test_instances

    def create_adjacency_matrix(self, users):
        # Create a simple adjacency matrix for demonstration
        num_users = len(users)
        adj_matrix = torch.eye(num_users, device=users.device, dtype=torch.float)
        
        # Example: Fill in adjacency matrix based on some logic
        # Here, we simply create an identity matrix as a placeholder
        # You should replace this with your actual logic
        return adj_matrix

    def create_graph_metrics(self, users):
        # Create a simple graph metrics tensor for demonstration
        num_users = len(users)
        graph_metrics = torch.zeros(num_users, 1, device=users.device, dtype=torch.float)
        
        # Example: Fill in graph metrics based on some logic
        # Here, we simply create a zero tensor as a placeholder
        # You should replace this with your actual logic
        return graph_metrics