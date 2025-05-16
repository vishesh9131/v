import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network (ST-GCN) for recommendation
    
    This model extends GCN by incorporating temporal dynamics in user-item interactions.
    """
    
    def __init__(self, num_users, num_items, adj_matrix, embedding_dim=64, 
                 num_time_steps=3, num_layers=2, dropout=0.1):
        """
        Initialize ST-GCN
        
        Args:
            num_users: Number of users
            num_items: Number of items
            adj_matrix: Adjacency matrix of the user-item interaction graph
            embedding_dim: Size of embedding vectors
            num_time_steps: Number of temporal steps to consider
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super(STGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_time_steps = num_time_steps
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Normalize adjacency matrix
        self.adj_matrix = self._normalize_adj_matrix(adj_matrix)
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Temporal embeddings
        self.temporal_embeddings = nn.Parameter(
            torch.Tensor(num_time_steps, embedding_dim)
        )
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn_layers.append(
                GCNLayer(embedding_dim, embedding_dim)
            )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_time_steps),
            nn.Softmax(dim=1)
        )
        
        # Final prediction layer
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
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.temporal_embeddings)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _normalize_adj_matrix(self, adj_matrix):
        """
        Normalize adjacency matrix for GCN
        
        Args:
            adj_matrix: Adjacency matrix in scipy sparse format
            
        Returns:
            Normalized adjacency matrix as torch sparse tensor
        """
        # Add self-loops
        adj_matrix = adj_matrix.copy()
        adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
        
        # Calculate D^(-1/2) * A * D^(-1/2)
        rowsum = np.array(adj_matrix.sum(axis=1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # Calculate normalized adjacency
        normalized_adj = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        
        # Convert to torch sparse tensor
        normalized_adj = normalized_adj.tocoo()
        indices = torch.LongTensor([normalized_adj.row, normalized_adj.col])
        values = torch.FloatTensor(normalized_adj.data)
        shape = torch.Size(normalized_adj.shape)
        
        return torch.sparse.FloatTensor(indices, values, shape)
    
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
        
        # Initialize node features for GCN
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        # Apply GCN layers for each time step
        temporal_embeddings = []
        
        for t in range(self.num_time_steps):
            # Add temporal information
            x_t = x + self.temporal_embeddings[t]
            
            # Apply GCN
            h = x_t
            for gcn_layer in self.gcn_layers:
                h = gcn_layer(h, self.adj_matrix)
            
            # Extract user and item embeddings
            user_gcn_emb = h[:self.num_users]
            item_gcn_emb = h[self.num_users:]
            
            # Get relevant embeddings for current batch
            batch_user_gcn_emb = user_gcn_emb[user_indices]
            batch_item_gcn_emb = item_gcn_emb[item_indices]
            
            # Combine user and item embeddings for this time step
            temporal_emb = torch.cat([batch_user_gcn_emb, batch_item_gcn_emb], dim=1)
            temporal_embeddings.append(temporal_emb)
        
        # Stack temporal embeddings
        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)  # [batch_size, num_time_steps, embedding_dim*2]
        
        # Compute temporal attention
        concat_original = torch.cat([user_emb, item_emb], dim=1)
        attention_weights = self.temporal_attention(concat_original)  # [batch_size, num_time_steps]
        
        # Apply attention to temporal embeddings
        attention_weights = attention_weights.unsqueeze(2)  # [batch_size, num_time_steps, 1]
        weighted_temporal = temporal_embeddings * attention_weights
        
        # Sum over time steps
        summed_emb = torch.sum(weighted_temporal, dim=1)  # [batch_size, embedding_dim*2]
        
        # Make prediction
        prediction = self.prediction_layer(summed_emb)
        
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
        # Process in batches to handle large candidate sets
        batch_size = 256
        num_samples = len(user_indices)
        predictions = []
        
        # Ensure both tensors have the same length
        if len(user_indices) != len(item_indices):
            if len(user_indices) == 1:
                user_indices = user_indices.repeat(len(item_indices))
            elif len(item_indices) == 1:
                item_indices = item_indices.repeat(len(user_indices))
            else:
                raise ValueError(f"Incompatible shape: user_indices {len(user_indices)}, item_indices {len(item_indices)}")
        
        # Ensure indices are within valid range
        user_indices = torch.clamp(user_indices, 0, self.num_users - 1)
        item_indices = torch.clamp(item_indices, 0, self.num_items - 1)
        
        # Process in batches
        for i in range(0, num_samples, batch_size):
            batch_users = user_indices[i:i+batch_size]
            batch_items = item_indices[i:i+batch_size]
            
            with torch.no_grad():
                batch_preds = self.forward(batch_users, batch_items)
            predictions.append(batch_preds)
        
        return torch.cat(predictions, dim=0)


class GCNLayer(nn.Module):
    """Graph Convolutional Layer"""
    
    def __init__(self, in_features, out_features):
        """
        Initialize GCN Layer
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
        """
        super(GCNLayer, self).__init__()
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Move adjacency matrix to the same device as features
        if adj.device != x.device:
            adj = adj.to(x.device)
            
        # Linear transformation
        support = torch.mm(x, self.weight)
        
        # Graph convolution with sparse tensor handling
        try:
            output = torch.sparse.mm(adj, support)
        except RuntimeError as e:
            if "expected" in str(e) and "got" in str(e):
                # Handle shape mismatch by padding or trimming
                if adj.size(0) > support.size(0):
                    # Pad support with zeros
                    padding = torch.zeros(adj.size(0) - support.size(0), support.size(1), device=support.device)
                    support = torch.cat([support, padding], dim=0)
                else:
                    # Trim adjacency matrix
                    indices = adj._indices()
                    values = adj._values()
                    mask = (indices[0] < support.size(0)) & (indices[1] < support.size(0))
                    new_indices = indices[:, mask]
                    new_values = values[mask]
                    adj = torch.sparse.FloatTensor(new_indices, new_values, 
                                                 (support.size(0), support.size(0)))
                output = torch.sparse.mm(adj, support)
            else:
                raise e
        
        # Add bias
        output = output + self.bias
        
        # Non-linearity
        output = F.relu(output)
        
        return output 