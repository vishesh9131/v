import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoRec(nn.Module):
    """
    AutoRec: Autoencoders Meet Collaborative Filtering
    
    AutoRec is an autoencoder-based collaborative filtering model.
    It takes a partial user-item interaction vector as input,
    reconstructs it through a bottleneck autoencoder architecture,
    and uses the reconstructed vector for rating prediction.
    
    This implementation is based on the I-AutoRec (Item-based AutoRec) variant
    which takes item vectors as inputs.
    
    Reference: Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015, May).
    AutoRec: Autoencoders meet collaborative filtering. In Proceedings of the 24th
    international conference on World Wide Web.
    """
    
    def __init__(self, num_users, num_items, hidden_dim=256, dropout=0.2):
        """
        Initialize AutoRec model
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            hidden_dim: Hidden dimension of the autoencoder
            dropout: Dropout probability
        """
        super(AutoRec, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        # Item rating matrix for storage
        self.item_ratings = None
        
        # Encoder (item vector -> hidden representation)
        self.encoder = nn.Sequential(
            nn.Linear(num_users, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder (hidden representation -> reconstructed item vector)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, num_users),
            nn.Sigmoid()  # Bound ratings to [0, 1] range (will be rescaled later)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def set_item_rating_matrix(self, rating_matrix):
        """
        Set the item rating matrix for the model
        
        Args:
            rating_matrix: Sparse user-item rating matrix (users as columns, items as rows)
        """
        # Store rating matrix and convert to dense tensor if needed
        if not isinstance(rating_matrix, torch.Tensor):
            self.item_ratings = torch.FloatTensor(rating_matrix.toarray())
        else:
            self.item_ratings = rating_matrix
            
        # Ensure item_ratings is on the same device as model
        self.item_ratings = self.item_ratings.to(next(self.parameters()).device)
    
    def forward(self, item_vectors):
        """
        Forward pass through the autoencoder
        
        Args:
            item_vectors: Batch of item rating vectors (sparse or dense)
            
        Returns:
            Reconstructed item vectors
        """
        # Encode
        hidden = self.encoder(item_vectors)
        
        # Decode
        reconstructed = self.decoder(hidden)
        
        return reconstructed
    
    def predict(self, user_indices, item_indices):
        """
        Make predictions for given user-item pairs
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        # Ensure item_ratings is set
        if self.item_ratings is None:
            raise ValueError("Item rating matrix must be set using set_item_rating_matrix before prediction")
        
        # Handle potential tensor shape mismatch
        if len(user_indices) != len(item_indices):
            if len(user_indices) == 1:
                user_indices = user_indices.repeat(len(item_indices))
            elif len(item_indices) == 1:
                item_indices = item_indices.repeat(len(user_indices))
            else:
                # If dimensions don't match and can't be broadcast
                raise ValueError(f"Mismatched dimensions: user_indices {len(user_indices)}, item_indices {len(item_indices)}")
        
        # Ensure indices don't exceed dimensions
        user_indices = torch.clamp(user_indices, 0, self.num_users - 1)
        item_indices = torch.clamp(item_indices, 0, self.num_items - 1)
        
        # Process in batches to prevent memory issues
        predictions = []
        batch_size = 128
        
        for i in range(0, len(item_indices), batch_size):
            batch_items = item_indices[i:i+batch_size]
            batch_users = user_indices[i:i+batch_size]
            
            try:
                # Make sure batch_items indices are within bounds of item_ratings
                if torch.max(batch_items) >= self.item_ratings.shape[0]:
                    # Handle out-of-bounds by clipping
                    batch_items = torch.clamp(batch_items, 0, self.item_ratings.shape[0] - 1)
                    
                # Get the item vectors
                batch_item_vectors = self.item_ratings[batch_items]
                
                # Reconstruct item rating vectors through the autoencoder
                reconstructed_vectors = self.forward(batch_item_vectors)
                
                # Extract specific user ratings from reconstructed vectors
                batch_predictions = torch.zeros(len(batch_items), device=reconstructed_vectors.device)
                
                for j, (item_idx, user_idx) in enumerate(zip(range(len(batch_items)), batch_users)):
                    # Ensure user_idx is within bounds
                    if user_idx < reconstructed_vectors.shape[1]:
                        batch_predictions[j] = reconstructed_vectors[item_idx, user_idx]
                    else:
                        # For users not in the training data, provide a fallback
                        batch_predictions[j] = reconstructed_vectors[item_idx].mean()
            except Exception as e:
                # Handle errors by providing a fallback prediction
                print(f"Error during AutoRec prediction: {e}")
                batch_predictions = torch.zeros(len(batch_items), device=self.item_ratings.device)
                
            predictions.append(batch_predictions)
        
        # Combine all batch predictions
        return torch.cat(predictions, dim=0)
    
    def get_item_recommendations(self, user_idx, top_k=10):
        """
        Get top-k item recommendations for a specific user
        
        Args:
            user_idx: User index
            top_k: Number of items to recommend
            
        Returns:
            Top-k recommended item indices and their predicted ratings
        """
        if self.item_ratings is None:
            raise ValueError("Item rating matrix must be set using set_item_rating_matrix before prediction")
        
        # Get all item vectors
        all_item_vectors = self.item_ratings
        
        # Reconstruct all item vectors
        with torch.no_grad():
            reconstructed_vectors = self.forward(all_item_vectors)
        
        # Get predicted ratings for the user across all items
        user_ratings = reconstructed_vectors[:, user_idx]
        
        # Find top-k items
        top_values, top_indices = torch.topk(user_ratings, k=top_k)
        
        return top_indices, top_values 