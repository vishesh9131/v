import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class GraphSAGE(nn.Module):
    """
    GraphSAGE for recommendation systems
    
    GraphSAGE (Graph Sample and Aggregate) generates node embeddings by sampling and aggregating
    features from a node's local neighborhood.
    
    Reference: Hamilton, W., Ying, Z., & Leskovec, J. (2017). 
    Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems.
    """
    
    def __init__(self, num_users, num_items, adj_lists, embedding_dim=64, aggregator_type='mean', 
                 num_sample=10, num_layers=2, dropout=0.2):
        """
        Initialize GraphSAGE model
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            adj_lists: Dictionary of adjacency lists for each node
            embedding_dim: Size of embedding vectors
            aggregator_type: Type of neighborhood aggregator ('mean', 'maxpool', or 'lstm')
            num_sample: Number of neighbors to sample for each node
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
        """
        super(GraphSAGE, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.adj_lists = adj_lists
        self.embedding_dim = embedding_dim
        self.aggregator_type = aggregator_type
        self.num_sample = num_sample
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # GraphSAGE layers for users and items
        self.user_sage_layers = nn.ModuleList()
        self.item_sage_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            input_dim = self.embedding_dim if i == 0 else self.embedding_dim * 2
            
            # User layers
            self.user_sage_layers.append(
                SageLayer(input_dim, self.embedding_dim, self.aggregator_type, self.dropout)
            )
            
            # Item layers
            self.item_sage_layers.append(
                SageLayer(input_dim, self.embedding_dim, self.aggregator_type, self.dropout)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def sample_neighbors(self, nodes, num_sample):
        """
        Sample neighbors for each node
        
        Args:
            nodes: List of node ids
            num_sample: Number of neighbors to sample
            
        Returns:
            Sampled neighbors for each node
        """
        samples = []
        for node in nodes:
            # Get neighbors
            if node in self.adj_lists:
                neighbors = list(self.adj_lists[node])
                if len(neighbors) == 0:
                    samples.append(np.array([node] * num_sample))  # Self-loop if no neighbors
                else:
                    # Sample with replacement if not enough neighbors
                    if len(neighbors) >= num_sample:
                        samples.append(np.random.choice(neighbors, num_sample, replace=False))
                    else:
                        samples.append(np.random.choice(neighbors, num_sample, replace=True))
            else:
                # Handle nodes with no adjacency information
                samples.append(np.array([node] * num_sample))
                
        return torch.LongTensor(np.array(samples))
    
    def forward(self, user_nodes=None, item_nodes=None):
        """
        Forward pass through GraphSAGE
        
        Args:
            user_nodes: User node ids (if None, use all users)
            item_nodes: Item node ids (if None, use all items)
            
        Returns:
            User embeddings and item embeddings
        """
        if user_nodes is None:
            user_nodes = torch.arange(self.num_users).to(self.user_embedding.weight.device)
        
        if item_nodes is None:
            item_nodes = torch.arange(self.num_items).to(self.item_embedding.weight.device)
        
        # Initialize with embedding layer
        user_embs = self.user_embedding(user_nodes)
        item_embs = self.item_embedding(item_nodes)
        
        # For each layer, aggregate neighbor information
        for i in range(self.num_layers):
            # Sample neighbors
            user_neighbors = self.sample_neighbors(user_nodes.cpu().numpy(), self.num_sample).to(user_embs.device)
            item_neighbors = self.sample_neighbors(item_nodes.cpu().numpy(), self.num_sample).to(item_embs.device)
            
            # Get neighbor embeddings
            user_neigh_embs = self.user_embedding(user_neighbors.view(-1)).view(user_neighbors.shape[0], 
                                                                          user_neighbors.shape[1], -1)
            item_neigh_embs = self.item_embedding(item_neighbors.view(-1)).view(item_neighbors.shape[0], 
                                                                          item_neighbors.shape[1], -1)
            
            # Aggregate and update
            user_embs = self.user_sage_layers[i](user_embs, user_neigh_embs)
            item_embs = self.item_sage_layers[i](item_embs, item_neigh_embs)
        
        return user_embs, item_embs
    
    def predict(self, user_indices, item_indices):
        """
        Make predictions for the given user-item pairs
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        user_embs, item_embs = self.forward(user_indices, item_indices)
        
        # Inner product as the prediction score
        scores = torch.sum(user_embs * item_embs, dim=1)
        
        return scores
    
    def calculate_loss(self, user_indices, pos_item_indices, neg_item_indices, lambda_val=1e-5):
        """
        Calculate BPR loss
        
        Args:
            user_indices: Users
            pos_item_indices: Positive items
            neg_item_indices: Negative items
            lambda_val: L2 regularization coefficient
            
        Returns:
            Loss value
        """
        # Get embeddings
        user_embs, _ = self.forward(user_indices)
        _, pos_item_embs = self.forward(None, pos_item_indices)
        _, neg_item_embs = self.forward(None, neg_item_indices)
        
        # BPR loss
        pos_scores = torch.sum(user_embs * pos_item_embs, dim=1)
        neg_scores = torch.sum(user_embs * neg_item_embs, dim=1)
        
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # L2 regularization
        reg_loss = lambda_val * (torch.norm(user_embs) ** 2 + 
                                torch.norm(pos_item_embs) ** 2 + 
                                torch.norm(neg_item_embs) ** 2)
        
        return bpr_loss + reg_loss
    
    @staticmethod
    def create_adj_lists(user_item_matrix):
        """
        Create adjacency lists from user-item interaction matrix
        
        Args:
            user_item_matrix: User-item interaction matrix in CSR format
            
        Returns:
            Dictionary of adjacency lists
        """
        num_users, num_items = user_item_matrix.shape
        
        # Convert to COO format for easier access to row and column indices
        user_item_coo = user_item_matrix.tocoo()
        
        # Create adjacency lists
        adj_lists = {}
        
        # User -> Item adjacency
        for u, i, _ in zip(user_item_coo.row, user_item_coo.col, user_item_coo.data):
            if u not in adj_lists:
                adj_lists[u] = set()
            adj_lists[u].add(i + num_users)  # Offset item indices by num_users
        
        # Item -> User adjacency
        for u, i, _ in zip(user_item_coo.row, user_item_coo.col, user_item_coo.data):
            if i + num_users not in adj_lists:
                adj_lists[i + num_users] = set()
            adj_lists[i + num_users].add(u)
        
        return adj_lists


class SageLayer(nn.Module):
    """
    GraphSAGE layer that performs neighbor aggregation and node update
    """
    
    def __init__(self, input_dim, output_dim, aggregator_type, dropout):
        """
        Initialize SageLayer
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            aggregator_type: Type of aggregator ('mean', 'maxpool', or 'lstm')
            dropout: Dropout probability
        """
        super(SageLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator_type = aggregator_type
        
        # Aggregator
        if aggregator_type == 'mean':
            self.aggregator = MeanAggregator()
        elif aggregator_type == 'maxpool':
            self.aggregator = MaxPoolAggregator(input_dim, output_dim)
        elif aggregator_type == 'lstm':
            self.aggregator = LSTMAggregator(input_dim)
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")
        
        # Weight for self embedding
        self.weight_self = nn.Linear(input_dim, output_dim)
        
        # Weight for neighbor embedding
        self.weight_neigh = nn.Linear(input_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, self_embs, neigh_embs):
        """
        Forward pass through the layer
        
        Args:
            self_embs: Self embeddings of shape [num_nodes, input_dim]
            neigh_embs: Neighbor embeddings of shape [num_nodes, num_neighbors, input_dim]
            
        Returns:
            Updated embeddings of shape [num_nodes, output_dim]
        """
        # Aggregate neighbor information
        neigh_agg = self.aggregator(neigh_embs)
        
        # Transform self and neighbor embeddings
        self_transformed = self.weight_self(self_embs)
        neigh_transformed = self.weight_neigh(neigh_agg)
        
        # Combine and apply non-linearity
        combined = self_transformed + neigh_transformed
        combined = F.relu(combined)
        
        # Apply dropout
        combined = self.dropout(combined)
        
        # Normalize
        combined = F.normalize(combined, p=2, dim=1)
        
        return combined


class MeanAggregator(nn.Module):
    """
    Mean aggregator for GraphSAGE
    """
    
    def forward(self, neigh_embs):
        """
        Aggregate neighbor embeddings using mean pooling
        
        Args:
            neigh_embs: Neighbor embeddings of shape [num_nodes, num_neighbors, input_dim]
            
        Returns:
            Aggregated embeddings of shape [num_nodes, input_dim]
        """
        return torch.mean(neigh_embs, dim=1)


class MaxPoolAggregator(nn.Module):
    """
    Max pooling aggregator for GraphSAGE
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initialize MaxPoolAggregator
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super(MaxPoolAggregator, self).__init__()
        
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, neigh_embs):
        """
        Aggregate neighbor embeddings using max pooling
        
        Args:
            neigh_embs: Neighbor embeddings of shape [num_nodes, num_neighbors, input_dim]
            
        Returns:
            Aggregated embeddings of shape [num_nodes, input_dim]
        """
        neigh_embs = self.fc(neigh_embs)
        maxpool_embs = torch.max(neigh_embs, dim=1)[0]
        return maxpool_embs


class LSTMAggregator(nn.Module):
    """
    LSTM aggregator for GraphSAGE
    """
    
    def __init__(self, input_dim, hidden_dim=None):
        """
        Initialize LSTMAggregator
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for LSTM (defaults to input_dim)
        """
        super(LSTMAggregator, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, neigh_embs):
        """
        Aggregate neighbor embeddings using LSTM
        
        Args:
            neigh_embs: Neighbor embeddings of shape [num_nodes, num_neighbors, input_dim]
            
        Returns:
            Aggregated embeddings of shape [num_nodes, hidden_dim]
        """
        _, (hidden, _) = self.lstm(neigh_embs)
        return hidden.squeeze(0) 