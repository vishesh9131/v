import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

class SyntheticDataset:
    def __init__(self, num_users=1000, num_items=500, interaction_density=0.05):
        self.num_users = num_users
        self.num_items = num_items
        self.interaction_density = interaction_density
        self.generate_data()
    
    def generate_data(self):
        # Generate random interactions
        num_interactions = int(self.num_users * self.num_items * self.interaction_density)
        user_indices = np.random.randint(0, self.num_users, num_interactions)
        item_indices = np.random.randint(0, self.num_items, num_interactions)
        
        # Create a sparse interaction matrix
        self.interaction_matrix = np.zeros((self.num_users, self.num_items))
        self.interaction_matrix[user_indices, item_indices] = 1
        
        # Convert to PyTorch tensors
        self.user_tensor = torch.tensor(user_indices, dtype=torch.long)
        self.item_tensor = torch.tensor(item_indices, dtype=torch.long)
        self.label_tensor = torch.tensor(self.interaction_matrix[user_indices, item_indices], dtype=torch.float)
    
    def get_dataloader(self, batch_size=32, train_split=0.8):
        dataset = TensorDataset(self.user_tensor, self.item_tensor, self.label_tensor)
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    synthetic_data = SyntheticDataset()
    train_loader, val_loader = synthetic_data.get_dataloader()
    
    # Now you can use train_loader and val_loader with your Scoreformer model 