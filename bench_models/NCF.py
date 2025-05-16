import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF)
    
    A simple implementation of NCF that combines matrix factorization with a neural network
    for recommendation tasks.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32], dropout=0.1):
        """
        Initialize NCF model
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            embedding_dim: Size of embedding vectors
            layers: List of layer sizes for the MLP component
            dropout: Dropout probability
        """
        super(NCF, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        input_size = 2 * embedding_dim
        
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1], 1)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with normal distribution"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, user_indices, item_indices):
        """
        Forward pass
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP layers
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Final prediction
        output = self.output_layer(x)
        
        return output.squeeze()
    
    def predict(self, user_indices, item_indices):
        """
        Make predictions for the given user-item pairs
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
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
        
        # Process in batches to handle large candidate sets
        batch_size = 256
        num_samples = len(user_indices)
        
        # For small batches, use forward directly
        if num_samples <= batch_size:
            return self.forward(user_indices, item_indices) 
        
        # For larger sets, process in batches
        predictions = []
        for i in range(0, num_samples, batch_size):
            batch_users = user_indices[i:i+batch_size]
            batch_items = item_indices[i:i+batch_size]
            
            # No need for torch.no_grad() during training
            batch_preds = self.forward(batch_users, batch_items)
            predictions.append(batch_preds)
        
        # Check if all tensors in predictions have the same shape
        shapes = [p.shape for p in predictions]
        if len(set(shapes)) > 1:
            # Handle inconsistent shapes by padding
            max_dim = max([len(p) for p in predictions])
            for i in range(len(predictions)):
                if len(predictions[i]) < max_dim:
                    pad_size = max_dim - len(predictions[i])
                    pad = torch.zeros(pad_size, device=predictions[i].device)
                    predictions[i] = torch.cat([predictions[i], pad])
        
        return torch.cat(predictions, dim=0) 