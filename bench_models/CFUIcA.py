import torch
import torch.nn as nn
import torch.nn.functional as F

class CFUIcA(nn.Module):
    """
    Collaborative Filtering with User-Item Context-aware Attention (CF-UIcA)
    
    This model combines collaborative filtering with context-aware attention mechanisms
    to capture complex user-item interactions.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, attention_dim=32, dropout=0.1):
        """
        Initialize CF-UIcA
        
        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Size of embedding vectors
            attention_dim: Size of attention layer
            dropout: Dropout probability
        """
        super(CFUIcA, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Context-aware attention
        self.attention_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()
        )
        
        # CF prediction layers
        self.prediction_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_indices, item_indices):
        """
        Forward pass
        
        Args:
            user_indices: Batch of user indices
            item_indices: Batch of item indices
            
        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        # Concatenated embeddings for attention
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        
        # Context-aware attention weights
        attention_weights = self.attention_layer(concat_emb)
        
        # Apply attention to user and item embeddings
        weighted_user_emb = user_emb * attention_weights
        weighted_item_emb = item_emb * attention_weights
        
        # Concatenate weighted embeddings
        weighted_concat = torch.cat([weighted_user_emb, weighted_item_emb], dim=1)
        
        # Make prediction
        prediction = self.prediction_layer(weighted_concat)
        
        return prediction.squeeze(-1)
    
    def predict(self, user_indices, item_indices):
        """
        Make predictions for given user-item pairs
        
        Args:
            user_indices: User indices tensor
            item_indices: Item indices tensor
            
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
        
        # For small batches, just use forward directly
        if num_samples <= batch_size:
            return self.forward(user_indices, item_indices)
        
        # For larger sets, process in batches but keep gradients flowing
        predictions = []
        for i in range(0, num_samples, batch_size):
            batch_users = user_indices[i:i+batch_size]
            batch_items = item_indices[i:i+batch_size]
            
            # Remove no_grad context to allow gradients to flow during training
            batch_preds = self.forward(batch_users, batch_items)
            predictions.append(batch_preds)
        
        return torch.cat(predictions, dim=0) 