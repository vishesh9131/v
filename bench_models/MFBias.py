import torch
import torch.nn as nn
import torch.nn.functional as F

class MFBias(nn.Module):
    """
    Matrix Factorization with Bias (MFBias)
    
    A classic matrix factorization model that includes user and item biases.
    This is one of the foundational approaches in recommender systems that
    decomposes the user-item interaction matrix into user and item latent factors
    plus bias terms.
    
    The prediction is computed as: r_ui = μ + b_u + b_i + <p_u, q_i>
    where:
        - μ is the global mean rating
        - b_u is the user bias
        - b_i is the item bias
        - p_u is the user latent factor
        - q_i is the item latent factor
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, dropout=0.0):
        """
        Initialize MFBias model
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            embedding_dim: Size of latent factors
            dropout: Dropout probability
        """
        super(MFBias, self).__init__()
        
        # Embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with normal distribution"""
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_indices, item_indices):
        """
        Forward pass
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        # Get user and item embeddings
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        # Apply dropout
        user_emb = self.dropout(user_emb)
        item_emb = self.dropout(item_emb)
        
        # Get bias terms
        user_b = self.user_bias(user_indices).squeeze()
        item_b = self.item_bias(item_indices).squeeze()
        global_b = self.global_bias.expand(user_indices.size(0))
        
        # Matrix factorization prediction
        # r_ui = global_bias + user_bias + item_bias + user_emb · item_emb
        mf_output = (user_emb * item_emb).sum(dim=1)
        prediction = global_b + user_b + item_b + mf_output
        
        return prediction
    
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
        
        return torch.cat(predictions, dim=0) 