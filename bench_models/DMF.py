import torch
import torch.nn as nn
import torch.nn.functional as F

class DMF(nn.Module):
    """
    Deep Matrix Factorization (DMF)
    
    DMF uses deep neural networks to process user and item representations
    from the interaction matrix. Unlike traditional MF, it learns
    complex non-linear user-item relationships with a multi-layer architecture.
    
    The model projects users and items into a common latent space through 
    separate neural networks and calculates their similarity for recommendation.
    
    Reference: Hong-Jian Xue, Xin-Yu Dai, Jianbing Zhang, Shujian Huang, and Jiajun Chen. 2017.
    Deep Matrix Factorization Models for Recommender Systems. In IJCAI.
    """
    
    def __init__(self, num_users, num_items, user_layers=[256, 128, 64], 
                 item_layers=[256, 128, 64], dropout=0.2):
        """
        Initialize DMF model
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            user_layers: List of layer sizes for user neural network
            item_layers: List of layer sizes for item neural network
            dropout: Dropout probability
        """
        super(DMF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        # Define user network
        user_network = []
        input_dim = num_items  # First layer takes user vector (items as features)
        
        for i, dim in enumerate(user_layers):
            user_network.append(nn.Linear(input_dim, dim))
            user_network.append(nn.ReLU())
            user_network.append(nn.Dropout(dropout))
            input_dim = dim
        
        self.user_network = nn.Sequential(*user_network)
        
        # Define item network
        item_network = []
        input_dim = num_users  # First layer takes item vector (users as features)
        
        for i, dim in enumerate(item_layers):
            item_network.append(nn.Linear(input_dim, dim))
            item_network.append(nn.ReLU())
            item_network.append(nn.Dropout(dropout))
            input_dim = dim
        
        self.item_network = nn.Sequential(*item_network)
        
        # Ensure output dimensions match
        assert user_layers[-1] == item_layers[-1], "Final layer dimensions must match"
        
        # Interaction matrix storage
        self.user_item_matrix = None
        self.item_user_matrix = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def set_interaction_matrices(self, user_item_matrix):
        """
        Set the interaction matrices for the model
        
        Args:
            user_item_matrix: User-item interaction matrix (CSR format)
        """
        # Store the matrices and convert to dense tensors if needed
        if not isinstance(user_item_matrix, torch.Tensor):
            self.user_item_matrix = torch.FloatTensor(user_item_matrix.toarray())
            self.item_user_matrix = torch.FloatTensor(user_item_matrix.transpose().toarray())
        else:
            self.user_item_matrix = user_item_matrix
            self.item_user_matrix = user_item_matrix.t()
        
        # Move to device
        device = next(self.parameters()).device
        self.user_item_matrix = self.user_item_matrix.to(device)
        self.item_user_matrix = self.item_user_matrix.to(device)
        
    def forward(self, user_vectors, item_vectors):
        """
        Forward pass
        
        Args:
            user_vectors: Batch of user interaction vectors
            item_vectors: Batch of item interaction vectors
            
        Returns:
            Prediction scores
        """
        # Process user and item networks
        user_embedding = self.user_network(user_vectors)
        item_embedding = self.item_network(item_vectors)
        
        # Normalize embeddings for cosine similarity
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        
        # Compute cosine similarity between user and item embeddings
        pred = torch.sum(user_embedding * item_embedding, dim=1)
        
        return pred
    
    def predict(self, user_indices, item_indices):
        """
        Make predictions for given user-item pairs
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        # Ensure interaction matrices are set
        if self.user_item_matrix is None or self.item_user_matrix is None:
            raise ValueError("Interaction matrices must be set using set_interaction_matrices before prediction")
        
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
        
        # Process in batches
        predictions = []
        batch_size = 128
        
        for i in range(0, len(user_indices), batch_size):
            batch_users = user_indices[i:i+batch_size]
            batch_items = item_indices[i:i+batch_size]
            
            # Get user and item vectors from interaction matrices
            batch_user_vectors = self.user_item_matrix[batch_users]
            batch_item_vectors = self.item_user_matrix[batch_items]
            
            # Forward pass
            batch_preds = self.forward(batch_user_vectors, batch_item_vectors)
            predictions.append(batch_preds)
        
        # Combine all batch predictions
        return torch.cat(predictions, dim=0) 