import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import json
import urllib.request
import zipfile
import gzip
import shutil
from tqdm import tqdm
import pickle
import sys
import importlib
import scipy.sparse as sp
# from Scoreformer import Scoreformer
import torch.nn.functional as F
# from CFUIcA import CFUIcA
# from bench_models.STGCN import STGCN
# from bench_models.NCF import NCF
# from bench_models.NGCF import NGCF
# from bench_models.GraphSAGE import GraphSAGE
# from bench_models.MFBias import MFBias
# from bench_models.AutoRec import AutoRec
# from bench_models.DMF import DMF
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

class NGCF(nn.Module):
    """
    Neural Graph Collaborative Filtering (NGCF)
    
    NGCF leverages user-item interactions as a bipartite graph structure and applies
    graph convolutional networks to capture collaborative signals.
    
    Reference: Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019, July).
    Neural graph collaborative filtering. In Proceedings of the 42nd international ACM SIGIR conference.
    """
    
    def __init__(self, num_users, num_items, adj_matrix, embedding_dim=64, layers=[64, 64, 64], node_dropout=0.1, mess_dropout=0.1):
        """
        Initialize NGCF model
        
        Args:
            num_users: Number of users
            num_items: Number of items
            adj_matrix: Sparse adjacency matrix of the user-item interaction graph
            embedding_dim: Size of embedding vectors
            layers: List of layer sizes for the GCN component
            node_dropout: Dropout probability for nodes
            mess_dropout: Dropout probability for messages
        """
        super(NGCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.n_layers = len(self.layers)
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout
        
        # Create adjacency matrix with self-connections
        self.adj_matrix = adj_matrix
        
        # Initialize weights
        self.weight_dict = self._init_weights()
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Dropout for messages
        self.dropout = nn.ModuleList()
        for _ in range(self.n_layers):
            self.dropout.append(nn.Dropout(p=self.mess_dropout))
        
        # Initialize weights
        self._init_weight()
        
    def _init_weight(self):
        """Initialize weights"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for k in self.weight_dict:
            if 'W1' in k or 'W2' in k:
                nn.init.xavier_uniform_(self.weight_dict[k])
            if 'b' in k:
                nn.init.zeros_(self.weight_dict[k])
    
    def _init_weights(self):
        """Initialize weight dictionary for all layers"""
        weight_dict = nn.ParameterDict()
        
        initializer = nn.init.xavier_uniform_
        weight_size_list = [self.embedding_dim] + self.layers
        
        for k in range(self.n_layers):
            weight_dict[f'W1_GCN_{k}'] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1])))
            weight_dict[f'b1_GCN_{k}'] = nn.Parameter(torch.zeros(weight_size_list[k+1]))
            
            weight_dict[f'W2_GCN_{k}'] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1])))
            weight_dict[f'b2_GCN_{k}'] = nn.Parameter(torch.zeros(weight_size_list[k+1]))
        
        return weight_dict
    
    def _sparse_dropout(self, x, dropout_rate, noise_shape):
        """Dropout for sparse tensors"""
        random_tensor = 1 - dropout_rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - dropout_rate))
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        """Convert scipy sparse matrix to torch sparse tensor"""
        coo = X.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def forward(self, adj_matrix=None):
        """
        Forward pass through NGCF
        
        Args:
            adj_matrix: Optional adjacency matrix override
            
        Returns:
            User and item embeddings after graph convolution
        """
        if adj_matrix is None:
            adj_matrix = self._convert_sp_mat_to_sp_tensor(self.adj_matrix).to(self.user_embedding.weight.device)
        
        # Initialize embeddings
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        # Check dimensions for consistency
        if adj_matrix.shape[0] != ego_embeddings.shape[0]:
            print(f"Warning: Dimension mismatch between adjacency matrix ({adj_matrix.shape[0]}) and embeddings ({ego_embeddings.shape[0]})")
            
            # We need to resize one of them to match
            if adj_matrix.shape[0] > ego_embeddings.shape[0]:
                # Pad embeddings
                padding = torch.zeros(adj_matrix.shape[0] - ego_embeddings.shape[0], 
                                     ego_embeddings.shape[1], 
                                     device=ego_embeddings.device)
                ego_embeddings = torch.cat([ego_embeddings, padding], dim=0)
            else:
                # Trim adjacency matrix - create new sparse tensor with smaller size
                indices = adj_matrix._indices()
                values = adj_matrix._values()
                mask = (indices[0] < ego_embeddings.shape[0]) & (indices[1] < ego_embeddings.shape[0])
                new_indices = indices[:, mask]
                new_values = values[mask]
                adj_matrix = torch.sparse.FloatTensor(new_indices, new_values, 
                                                      (ego_embeddings.shape[0], ego_embeddings.shape[0]))
        
        # Node dropout
        if self.node_dropout > 0:
            adj_matrix = self._sparse_dropout(adj_matrix, self.node_dropout, adj_matrix._indices().shape[1])
        
        # Multi-layer Graph Convolution
        for k in range(self.n_layers):
            try:
                # Simple Graph Convolution
                side_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)
                
                # Transformation
                sum_embeddings = F.linear(side_embeddings, self.weight_dict[f'W1_GCN_{k}'], self.weight_dict[f'b1_GCN_{k}'])
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.linear(bi_embeddings, self.weight_dict[f'W2_GCN_{k}'], self.weight_dict[f'b2_GCN_{k}'])
                
                # Non-linear activation
                ego_embeddings = F.leaky_relu(sum_embeddings + bi_embeddings, negative_slope=0.2)
                ego_embeddings = self.dropout[k](ego_embeddings)
                
                # Normalize
                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings.append(norm_embeddings)
            except RuntimeError as e:
                print(f"Error in layer {k}: {e}")
                # Use previous embeddings
                all_embeddings.append(all_embeddings[-1])
                continue
        
        # Concatenate or sum all layers' embeddings
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        # Split user and item embeddings
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        
        return u_g_embeddings, i_g_embeddings
    
    def predict(self, user_indices, item_indices):
        """
        Make predictions for the given user-item pairs
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        try:
            u_g_embeddings, i_g_embeddings = self.forward()
            
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
            user_indices = torch.clamp(user_indices, 0, self.num_users - 1)
            item_indices = torch.clamp(item_indices, 0, self.num_items - 1)
            
            u_embeddings = u_g_embeddings[user_indices]
            i_embeddings = i_g_embeddings[item_indices]
            
            # Inner product as the prediction score
            scores = torch.sum(torch.mul(u_embeddings, i_embeddings), dim=1)
            
            return scores
        except Exception as e:
            print(f"Error in NGCF predict: {e}")
            
            # Fallback to simple dot product of raw embeddings
            print("Using fallback prediction method...")
            user_emb = self.user_embedding(user_indices)
            item_emb = self.item_embedding(item_indices)
            
            # Simple dot product
            scores = torch.sum(user_emb * item_emb, dim=1)
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
        u_g_embeddings, i_g_embeddings = self.forward()
        
        u_embeddings = u_g_embeddings[user_indices]
        pos_i_embeddings = i_g_embeddings[pos_item_indices]
        neg_i_embeddings = i_g_embeddings[neg_item_indices]
        
        # BPR loss
        pos_scores = torch.sum(torch.mul(u_embeddings, pos_i_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(u_embeddings, neg_i_embeddings), dim=1)
        
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # L2 regularization
        reg_loss = lambda_val * (torch.norm(u_embeddings) ** 2 + 
                                 torch.norm(pos_i_embeddings) ** 2 + 
                                 torch.norm(neg_i_embeddings) ** 2)
        
        return bpr_loss + reg_loss
        
    @staticmethod
    def create_adj_matrix(user_item_matrix):
        """
        Create adjacency matrix for NGCF
        
        Args:
            user_item_matrix: User-item interaction matrix in CSR format
            
        Returns:
            Adjacency matrix in CSR format
        """
        n_users, n_items = user_item_matrix.shape
        
        # Create adjacency matrix [user, item; item, user]
        adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        
        # User-item interactions
        adj_mat[:n_users, n_users:] = user_item_matrix
        adj_mat[n_users:, :n_users] = user_item_matrix.T
        
        # Convert to CSR for faster operations
        adj_mat = adj_mat.tocsr()
        
        # Compute D^(-1/2) * A * D^(-1/2)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        return norm_adj 




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


class Scoreformer(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        input_dim,
        num_targets,
        num_users=None,
        num_items=None,
        dropout=0.1,
        use_transformer=True,
        use_dng=True,
        use_weights=True
    ):
        super(Scoreformer, self).__init__()
        self.use_transformer = use_transformer
        self.use_dng = use_dng
        self.use_weights = use_weights
        self.num_users = num_users
        self.num_items = num_items
        self.input_dim = input_dim
        
        # For recommendation tasks - embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, input_dim) if num_users else None
        self.item_embedding = nn.Embedding(num_items, input_dim) if num_items else None
        
        # Project input to the model dimension
        self.initial_proj = nn.Linear(input_dim * 2, d_model)  # Modified to accept concatenated embeddings
        
        # Transformer encoder branch (if enabled)
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.encoder = None
        
        # DNG components (Direct, Neighborhood, Graph scoring)
        if self.use_dng:
            # Direct scoring component
            self.direct_layer = nn.Linear(d_model, d_model)
            
            # Neighborhood scoring component
            self.neighborhood_layer = nn.Linear(d_model, d_model)
            
            # Graph structure scoring component
            self.graph_layer = nn.Linear(d_model, d_model)
            
            # Combine the three scores
            self.dng_combine = nn.Linear(d_model * 3, d_model)
        else:
            self.direct_layer = None
            self.neighborhood_layer = None
            self.graph_layer = None
            self.dng_combine = None
        
        # Weight-specific logic can be added here if needed
        if self.use_weights:
            self.weight_layer = nn.Linear(d_model, d_model)
        else:
            self.weight_layer = None
        
        # Final layer projecting to the number of targets
        self.final_linear = nn.Linear(d_model, num_targets)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        if self.user_embedding is not None:
            nn.init.normal_(self.user_embedding.weight, std=0.01)
        if self.item_embedding is not None:
            nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, x=None, adj_matrix=None, graph_metrics=None, user_indices=None, item_indices=None):
        """
        Forward pass through Scoreformer
        
        For standard input:
            x: [batch_size, input_dim] - Feature vectors
            adj_matrix: Adjacency matrix for graph relations
            graph_metrics: Additional graph metrics
            
        For recommendation tasks:
            user_indices: [batch_size] - User indices
            item_indices: [batch_size] - Item indices
        """
        # Handle recommendation task input (user and item indices)
        if user_indices is not None and item_indices is not None and self.user_embedding is not None and self.item_embedding is not None:
            user_emb = self.user_embedding(user_indices)
            item_emb = self.item_embedding(item_indices)
            # Combine user and item embeddings
            x = torch.cat([user_emb, item_emb], dim=1) if x is None else x
            
        # Project input to model dimension
        h = self.initial_proj(x)  # [batch_size, d_model]
        
        # Transformer encoder
        if self.use_transformer and self.encoder is not None:
            # Transformer expects shape: [sequence_length, batch_size, d_model]
            h_trans = h.unsqueeze(0)  # add sequence dimension: [1, batch_size, d_model]
            h_trans = self.encoder(h_trans)
            h_trans = h_trans.squeeze(0)    # [batch_size, d_model]
        else:
            h_trans = h
            
        # DNG scoring mechanism
        if self.use_dng:
            # Direct score - based on node features directly
            d_score = self.direct_layer(h_trans)
            d_score = F.leaky_relu(d_score)
            
            # Neighborhood score - incorporate adjacency information if available
            if adj_matrix is not None:
                # This is a placeholder - real implementation would use the adjacency matrix
                n_score = self.neighborhood_layer(h_trans)
            else:
                n_score = self.neighborhood_layer(h_trans)
            n_score = F.leaky_relu(n_score)
            
            # Graph structure score - incorporate broader graph metrics if available
            if graph_metrics is not None:
                # This is a placeholder - real implementation would use graph metrics
                g_score = self.graph_layer(h_trans)
            else:
                g_score = self.graph_layer(h_trans)
            g_score = F.leaky_relu(g_score)
            
            # Combine the three scores
            combined_scores = torch.cat([d_score, n_score, g_score], dim=1)
            h = self.dng_combine(combined_scores)
            h = F.leaky_relu(h)
        else:
            h = h_trans
            
        # Apply weights if enabled
        if self.use_weights and self.weight_layer is not None:
            h = self.weight_layer(h)
            h = F.leaky_relu(h)

        # Final projection to target dimension
        output = self.final_linear(h)  # [batch_size, num_targets]
        return output
    
    def predict(self, user_indices, item_indices):
        """
        Make predictions for recommendation tasks
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted ratings/scores
        """
        # Process input in batches to handle large candidate sets
        if len(user_indices) != len(item_indices):
            # Handle case where we have a single user with many items (for recommendation)
            if len(user_indices) == 1:
                user_indices = user_indices.repeat(len(item_indices))
            # Handle case where we have a single item with many users
            elif len(item_indices) == 1:
                item_indices = item_indices.repeat(len(user_indices))
            else:
                raise ValueError(f"Incompatible shapes: user_indices {user_indices.shape}, item_indices {item_indices.shape}")
        
        batch_size = 256  # Process in batches to avoid memory issues
        num_samples = len(user_indices)
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch_users = user_indices[i:i+batch_size]
            batch_items = item_indices[i:i+batch_size]
            
            # Get embeddings for this batch
            user_emb = self.user_embedding(batch_users)
            item_emb = self.item_embedding(batch_items)
            
            # Concatenate embeddings
            x = torch.cat([user_emb, item_emb], dim=1)
            
            # Forward pass
            batch_predictions = self.forward(x=x)
            predictions.append(batch_predictions)
        
        # Combine predictions from all batches
        predictions = torch.cat(predictions, dim=0)
        return predictions.squeeze(-1)
       












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


import torch
import torch.nn as nn
import torch.nn.functional as F


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



import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import json
import urllib.request
import zipfile
import gzip
import shutil
from tqdm import tqdm
import pickle
import sys
import importlib
import scipy.sparse as sp
# from Scoreformer import Scoreformer
import torch.nn.functional as F
# from CFUIcA import CFUIcA
# from bench_models.STGCN import STGCN
# from bench_models.NCF import NCF
# from bench_models.NGCF import NGCF
# from bench_models.GraphSAGE import GraphSAGE
# from bench_models.MFBias import MFBias
# from bench_models.AutoRec import AutoRec
# from bench_models.DMF import DMF
import argparse

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






"""
Recommendation System Benchmark Tool

This script provides a comprehensive benchmarking framework for recommendation systems,
comparing multiple state-of-the-art algorithms on standard datasets. The following models 
are included in this benchmark:

1. Scoreformer:
   - A novel transformer-based recommendation model with Direct-Neighborhood-Graph (DNG) scoring
   - Combines transformer attention mechanisms with graph-based representations
   - Features configurable transformer depth, embedding dimensions, and attention heads
   - Modular architecture with options to enable/disable transformer and DNG components

2. CFUIcA (Collaborative Filtering with User-Item Context-aware Attention):
   - Leverages context-aware attention mechanisms to capture complex user-item interactions
   - Combines collaborative filtering with attention weights for improved recommendations
   - Particularly effective for capturing subtle preference patterns

3. STGCN (Spatial-Temporal Graph Convolutional Network):
   - Extends GCN by incorporating temporal dynamics in user-item interactions
   - Processes the evolving graph structure over multiple time steps
   - Applies temporal attention to weight the importance of different time periods
   - Well-suited for recommendation scenarios with time-varying preferences

4. NCF (Neural Collaborative Filtering):
   - Combines matrix factorization with multi-layer perceptrons for recommendation
   - Learns non-linear interactions between user and item latent features
   - Uses embeddings and deep neural networks for flexible modeling
   - Effective for both explicit and implicit feedback scenarios

5. NGCF (Neural Graph Collaborative Filtering):
   - Explicitly encodes the collaborative signal in user-item interactions
   - Exploits high-order connectivity in the user-item graph
   - Propagates embeddings through the interaction graph structure
   - Leverages both direct and higher-order user-item connections

6. GraphSAGE (Graph Sample and Aggregate):
   - Generates node embeddings by sampling and aggregating features from node neighborhoods
   - Scales to large graphs by using neighborhood sampling strategies
   - Supports different aggregation functions (mean, max, LSTM)
   - Effectively captures structural information for recommendation

7. MFBias (Matrix Factorization with Bias):
   - Classic matrix factorization approach with user and item biases
   - Decomposes the user-item interaction matrix into latent factors
   - Includes global, user and item bias terms to model rating deviations
   - Foundational approach that serves as a strong baseline

8. AutoRec (Autoencoder-based Recommendation):
   - Uses autoencoders to learn compact representations of user-item interactions
   - Reconstructs user-item vectors through a bottleneck architecture
   - Captures non-linear relationships without requiring negative sampling
   - Effective for handling sparse interaction data

9. DMF (Deep Matrix Factorization):
   - Extends matrix factorization with deep neural networks
   - Processes user and item interaction vectors with separate networks
   - Projects users and items into a common latent space
   - Captures complex non-linear relationships between users and items
   
This benchmark supports:
- Model training and evaluation on multiple datasets
- Comparison using standard metrics: HR@K and NDCG@K
- Detailed logging and result visualization
- Modular design for easy extension with new models
"""

# Create bench_models directory if it doesn't exist
os.makedirs("bench_models", exist_ok=True)

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Datasets to benchmark
DATASETS = {
    # MovieLens 100K: 100,000 ratings from 943 users on 1,682 movies
    # Contains demographic information for users (age, gender, occupation, zip)
    # Rating scale: 1-5 stars
    # Timestamp information available for temporal analysis
    # Source: GroupLens Research Project at the University of Minnesota
    "ml-100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "filename": "ml-100k.zip",
        "data_path": "data/ml-100k",
        "rating_file_path": "u.data",  # Path inside the extracted directory
        "sep": "\t",
        "header": None,
        "names": ["user_id", "item_id", "rating", "timestamp"],
        "description": "MovieLens 100K dataset with 100,000 ratings (1-5) from 943 users on 1,682 movies."
    },
    
    # MovieLens 1M: 1 million ratings from 6,000 users on 4,000 movies
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "filename": "ml-1m.zip",
        "data_path": "data/ml-1m",
        "rating_file_path": "ratings.dat",  # Path inside the extracted directory
        "sep": "::",
        "header": None,
        "names": ["user_id", "item_id", "rating", "timestamp"],
        "description": "MovieLens 1M dataset with 1 million ratings from 6,000 users on 4,000 movies."
    },
    
    # Last.FM: Music listening data
    "lastfm": {
        "url": "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
        "filename": "hetrec2011-lastfm-2k.zip",
        "data_path": "data/lastfm",
        "rating_file_path": "user_artists.dat",  # Path inside the extracted directory
        "sep": "\t",
        "header": 0,
        "names": ["user_id", "item_id", "weight"],
        "description": "Last.FM dataset with 92,834 artist listening records from 1,892 users."
    },
    
    # REES46 Ecommerce Behavior Dataset
    # "rees46": {
    #     "url": "https://drive.google.com/drive/folders/1gvPC9ZOr07w6DTEuogxHj9Lu_8VRs4YM?usp=sharing",  # Local file
    #     "filename": "events.csv",
    #     "data_path": "data/rees46",
    #     "rating_file_path": "REES46/events.csv",  # Path to the original file
    #     "local_file": True,  # Flag to indicate it's a local file
    #     "sep": ",",
    #     "header": 0,
    #     "names": ["event_time", "event_type", "product_id", "category_id", "category_code", "brand", "price", "user_id", "user_session"],
    #     "description": "REES46 e-commerce behavior dataset with user interactions with products."
    # }
}

# Simple NCF model for comparison
class SimpleNCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, layers=[64, 32], dropout=0.1):
        super(SimpleNCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.ModuleList()
        input_size = 2 * embedding_dim
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size
        self.output_layer = nn.Linear(layers[-1], 1)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        x = torch.cat([user_emb, item_emb], dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        output = self.output_layer(x)
        return output.squeeze()
    
    def predict(self, user_indices, item_indices):
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
        
        # Process in smaller batches
        batch_size = 256
        num_samples = len(user_indices)
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch_users = user_indices[i:i+batch_size]
            batch_items = item_indices[i:i+batch_size]
            with torch.no_grad():
                batch_preds = self.forward(batch_users, batch_items)
            predictions.append(batch_preds)
        
        return torch.cat(predictions, dim=0)

# Models to compare - only using Scoreformer for now
MODELS = {
    # Scoreformer: Novel transformer-based model with Direct-Neighborhood-Graph scoring
    "Scoreformer": Scoreformer,
    
    # CFUIcA: Collaborative Filtering with User-Item Context-aware Attention
    "CFUIcA": CFUIcA,
    
    # STGCN: Spatial-Temporal Graph Convolutional Network for temporal dynamics
    "STGCN": STGCN,
    
    # NCF: Neural Collaborative Filtering with matrix factorization and MLPs
    "NCF": NCF,
    
    # NGCF: Neural Graph Collaborative Filtering using high-order connectivity
    "NGCF": NGCF,
    
    # GraphSAGE: Graph Sample and Aggregate for neighborhood embeddings
    "GraphSAGE": GraphSAGE,
    
    # MFBias: Matrix Factorization with bias terms
    "MFBias": MFBias,
    
    # AutoRec: Autoencoder-based recommendation model
    "AutoRec": AutoRec,
    
    # DMF: Deep Matrix Factorization with neural networks
    "DMF": DMF
}

# Metrics
def hit_ratio(ranked_list, ground_truth):
    """Calculate hit ratio"""
    return 1.0 if ground_truth in ranked_list else 0.0

def ndcg(ranked_list, ground_truth):
    """Calculate NDCG (Normalized Discounted Cumulative Gain)"""
    if ground_truth in ranked_list:
        index = ranked_list.index(ground_truth)
        return np.reciprocal(np.log2(index + 2))
    return 0.0

class RecommendationDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'rating': self.ratings[idx]
        }

def preprocess_dataset(dataset_name):
    """Download and preprocess a dataset"""
    dataset_info = DATASETS[dataset_name]
    
    # Create dataset directory
    os.makedirs(dataset_info["data_path"], exist_ok=True)
    
    # Handle local files differently
    if dataset_info.get("local_file", False):
        ratings_file = dataset_info["rating_file_path"]
        print(f"Using local file for {dataset_name} dataset: {ratings_file}")
    else:
        # Path to the downloaded file
        download_path = os.path.join(dataset_info["data_path"], dataset_info["filename"])
        
        # Download dataset if it doesn't exist and URL is provided
        if dataset_info["url"] and not os.path.exists(download_path):
            print(f"Downloading {dataset_name} dataset...")
            # For Yelp which requires Kaggle authentication, we'll assume manual download
            if dataset_name == "yelp":
                print("Please manually download the Yelp dataset from Kaggle and place it in data/yelp/")
                return None
            urllib.request.urlretrieve(dataset_info["url"], download_path)
        
        # Path to the actual ratings file
        ratings_file = dataset_info["rating_file_path"]
        
        # Extract if it's a zip file and we don't have the ratings file yet
        if download_path.endswith('.zip') and not os.path.exists(os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])):
            print(f"Extracting {dataset_name} dataset...")
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_info["data_path"])
            
            # For some datasets, get the correct path after extraction
            if dataset_name == "lastfm":
                ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
            elif dataset_name == "ml-100k":
                ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
            elif dataset_name == "ml-1m":
                ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
        elif download_path.endswith('.gz') and not os.path.exists(ratings_file):
            with gzip.open(download_path, 'rb') as f_in:
                with open(download_path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
    # Read ratings
    if dataset_name == "yelp":
        # Yelp requires special handling for JSON lines
        df = pd.read_json(os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"]), lines=True)
        df = df[['user_id', 'business_id', 'stars']]
        df.rename(columns={'business_id': 'item_id', 'stars': 'rating'}, inplace=True)
    elif dataset_name == "rees46":
        # Special handling for REES46 dataset
        df = pd.read_csv(ratings_file)
        # Filter for view and cart events as implicit positive feedback
        df = df[df['event_type'].isin(['view', 'cart'])]
        # Create a rating column (cart = 2, view = 1)
        df['rating'] = df['event_type'].apply(lambda x: 2 if x == 'cart' else 1)
        # Select relevant columns
        df = df[['user_id', 'product_id', 'rating']]
        # Rename columns to match our standard
        df.rename(columns={'product_id': 'item_id'}, inplace=True)
        # Take a random sample if the dataset is too large
        if len(df) > 500000:
            df = df.sample(n=500000, random_state=42)
    elif dataset_name == "lastfm":
        # Special handling for LastFM dataset
        lastfm_ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
        if not os.path.exists(lastfm_ratings_file):
            print(f"LastFM ratings file not found at expected path: {lastfm_ratings_file}")
            # Try finding it
            lastfm_files = [f for f in os.listdir(dataset_info["data_path"]) if f.endswith('.dat')]
            print(f"Available .dat files: {lastfm_files}")
            for file in lastfm_files:
                if 'user_artists' in file:
                    lastfm_ratings_file = os.path.join(dataset_info["data_path"], file)
                    print(f"Found LastFM ratings file: {lastfm_ratings_file}")
                    break
        
        df = pd.read_csv(
            lastfm_ratings_file, 
            sep=dataset_info["sep"], 
            header=dataset_info["header"],
            names=dataset_info["names"]
        )
        # Rename 'weight' to 'rating' for consistency
        df.rename(columns={'weight': 'rating'}, inplace=True)
        
        # Convert listen counts (weights) to implicit ratings
        # Normalize weights to a 1-5 scale for consistency with other datasets
        min_weight = df['rating'].min()
        max_weight = df['rating'].max()
        if max_weight > min_weight:  # Avoid division by zero
            # Use log scale for better distribution since listening counts can be highly skewed
            df['rating'] = 1 + 4 * (np.log1p(df['rating']) - np.log1p(min_weight)) / (np.log1p(max_weight) - np.log1p(min_weight))
        else:
            # If all weights are the same, set them to mid-range
            df['rating'] = 3.0
    elif dataset_name == "ml-100k" or dataset_name == "ml-1m":
        # Handle MovieLens datasets
        ml_ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
        if not os.path.exists(ml_ratings_file):
            print(f"MovieLens ratings file not found at expected path: {ml_ratings_file}")
            # Try finding the file
            ml_files = [f for f in os.listdir(dataset_info["data_path"])]
            print(f"Files in dataset directory: {ml_files}")
            
            # For ML-100K, look for 'u.data' file
            if dataset_name == "ml-100k":
                subdirs = [d for d in ml_files if os.path.isdir(os.path.join(dataset_info["data_path"], d))]
                for subdir in subdirs:
                    subdir_path = os.path.join(dataset_info["data_path"], subdir)
                    subdir_files = os.listdir(subdir_path)
                    if "u.data" in subdir_files:
                        ml_ratings_file = os.path.join(subdir_path, "u.data")
                        print(f"Found ML-100K ratings file: {ml_ratings_file}")
                        break
            
            # For ML-1M, look for 'ratings.dat' file
            elif dataset_name == "ml-1m":
                subdirs = [d for d in ml_files if os.path.isdir(os.path.join(dataset_info["data_path"], d))]
                for subdir in subdirs:
                    subdir_path = os.path.join(dataset_info["data_path"], subdir)
                    subdir_files = os.listdir(subdir_path)
                    if "ratings.dat" in subdir_files:
                        ml_ratings_file = os.path.join(subdir_path, "ratings.dat")
                        print(f"Found ML-1M ratings file: {ml_ratings_file}")
                        break
        
        df = pd.read_csv(
            ml_ratings_file, 
            sep=dataset_info["sep"], 
            header=dataset_info["header"],
            names=dataset_info["names"]
        )
    else:
        # For other datasets, read from the correct path
        df = pd.read_csv(
            ratings_file, 
            sep=dataset_info["sep"], 
            header=dataset_info["header"],
            names=dataset_info["names"]
        )
    
    # Encode user_ids and item_ids
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    df['item_id'] = item_encoder.fit_transform(df['item_id'])
    
    # Save encoders for later use
    with open(os.path.join(dataset_info["data_path"], "user_encoder.pkl"), "wb") as f:
        pickle.dump(user_encoder, f)
    with open(os.path.join(dataset_info["data_path"], "item_encoder.pkl"), "wb") as f:
        pickle.dump(item_encoder, f)
    
    # Create adjacency matrix for graph algorithms
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    
    # Create user-item interaction matrix
    user_item_matrix = sp.coo_matrix(
        (np.ones(len(df)), (df['user_id'].values, df['item_id'].values)),
        shape=(num_users, num_items)
    ).tocsr()
    
    # Save processed data
    processed_data = {
        'df': df,
        'user_item_matrix': user_item_matrix,
        'num_users': num_users,
        'num_items': num_items
    }
    
    with open(os.path.join(dataset_info["data_path"], "processed_data.pkl"), "wb") as f:
        pickle.dump(processed_data, f)
    
    return processed_data

def get_train_test_data(df, test_size=0.2):
    """Split data into train and test sets"""
    # Group by user to ensure each user has test items
    user_groups = df.groupby('user_id')
    train_data = []
    test_data = []
    
    for user_id, group in user_groups:
        # For each user, sample test_size of their interactions for testing
        # Make sure each user has at least 2 interactions to allow for train/test split
        if len(group) < 2:
            # If user has only one interaction, keep it in training
            train_data.append(group)
            continue
            
        n_test = max(1, int(test_size * len(group)))
        test_indices = np.random.choice(group.index, n_test, replace=False)
        train_indices = list(set(group.index) - set(test_indices))
        
        train_data.append(df.loc[train_indices])
        test_data.append(df.loc[test_indices])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    return train_df, test_df

def evaluate_model(model, test_df, user_item_matrix, num_items, top_k=10, model_name="", max_items_per_user=500):
    """Evaluate a model using HR@K and NDCG@K metrics with more robust methodology"""
    print(f"Evaluating {model_name} with {top_k} recommendations per user")
    
    # Dictionary to store metrics at multiple cutoff points
    metrics = {}
    for k in [5, 10, 20]:
        metrics[f'HR@{k}'] = []
        metrics[f'NDCG@{k}'] = []
        metrics[f'Precision@{k}'] = []
        metrics[f'Recall@{k}'] = []
    
    # Group test data by user for more efficient evaluation
    test_users = test_df['user_id'].unique()
    
    # Use a reasonable sample for evaluation
    if len(test_users) > 200:
        print(f"Sampling 200 users from {len(test_users)} for evaluation")
        np.random.seed(42)  # For reproducibility
        test_users = np.random.choice(test_users, 200, replace=False)
    
    error_count = 0
    max_errors = 15  # Increased tolerance for errors
    
    # Process users in parallel or sequentially
    progress_bar = tqdm(test_users, desc=f"Evaluating {model_name}")
    
    for user_id in progress_bar:
        try:
            # Get ground truth items for this user
            user_test_items = test_df[test_df['user_id'] == user_id]['item_id'].values
            
            if len(user_test_items) == 0:
                continue  # Skip users with no test items
            
            # Get items that the user has not interacted with (in training set)
            user_interactions = user_item_matrix[user_id].indices
            all_items = np.arange(num_items)
            candidate_items = np.setdiff1d(all_items, user_interactions)
            
            # If there are too many candidate items, use a hybrid sampling approach
            if len(candidate_items) > max_items_per_user:
                # Create a candidate pool:
                # 1. Include all ground truth items
                # 2. Include some popular items (top 20%)
                # 3. Include some random items from the rest
                
                # Get item popularity
                item_popularity = np.array(user_item_matrix.sum(axis=0)).flatten()
                item_ranks = np.argsort(-item_popularity)  # Descending order
                
                # Top 20% popular items (excluding already interacted ones)
                num_popular = min(int(max_items_per_user * 0.2), num_items // 5)
                popular_candidates = [item for item in item_ranks[:num_popular] if item not in user_interactions]
                
                # Random items to fill the rest
                remaining_candidates = [item for item in candidate_items if item not in popular_candidates and item not in user_test_items]
                num_random = max_items_per_user - len(popular_candidates) - len(user_test_items)
                
                if num_random > 0 and len(remaining_candidates) > 0:
                    random_candidates = np.random.choice(
                        remaining_candidates, 
                        size=min(num_random, len(remaining_candidates)), 
                        replace=False
                    )
                else:
                    random_candidates = []
                
                # Combine pools
                candidate_items = np.concatenate([
                    popular_candidates,
                    random_candidates,
                    user_test_items
                ])
            
            # Make sure all ground truth items are in the candidates
            for gt_item in user_test_items:
                if gt_item not in candidate_items:
                    candidate_items = np.append(candidate_items, gt_item)
            
            # Convert to tensors
            user_tensor = torch.LongTensor([user_id]).to(device)  # Single user
            item_tensor = torch.LongTensor(candidate_items).to(device)  # All candidate items
            
            # Get predictions in smaller batches to prevent dimension issues
            all_scores = []
            batch_size = 100
            
            for i in range(0, len(item_tensor), batch_size):
                batch_items = item_tensor[i:i+batch_size]
                batch_users = user_tensor.repeat(len(batch_items))
                
                # Make sure both tensors have the same length
                assert len(batch_users) == len(batch_items), "Tensor length mismatch"
                
                with torch.no_grad():  # No gradients needed for evaluation
                    try:
                        batch_scores = model.predict(batch_users, batch_items)
                        all_scores.append(batch_scores)
                    except RuntimeError as e:
                        print(f"Batch error for user {user_id}, items {i}:{i+batch_size}: {e}")
                        # Use zeros as fallback scores for this batch
                        all_scores.append(torch.zeros(len(batch_items), device=device))
            
            # Combine all batches
            scores = torch.cat(all_scores)
            
            # Make sure scores are valid
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(f"NaN or Inf scores detected for user {user_id}, skipping")
                continue
            
            # Sort items by score
            _, indices = torch.sort(scores, descending=True)
            ranked_items = candidate_items[indices.cpu().numpy()]
            
            # Calculate metrics at multiple cutoff points
            for k in [5, 10, 20]:
                if k > len(ranked_items):
                    continue
                    
                # Get top-k items
                recommended_items = ranked_items[:k]
                
                # Hit Ratio - if any ground truth item is in recommendations
                hit = any(item in recommended_items for item in user_test_items)
                metrics[f'HR@{k}'].append(float(hit))
                
                # NDCG - normalized discounted cumulative gain
                dcg = 0
                idcg = 0
                
                # Calculate DCG
                for i, item in enumerate(recommended_items):
                    if item in user_test_items:
                        # Rank is 0-indexed, so +1 for the formula
                        dcg += 1 / np.log2(i + 2)
                
                # Calculate IDCG
                ideal_ranking = min(k, len(user_test_items))
                for i in range(ideal_ranking):
                    idcg += 1 / np.log2(i + 2)
                
                # Avoid division by zero
                if idcg > 0:
                    ndcg = dcg / idcg
                else:
                    ndcg = 0
                
                metrics[f'NDCG@{k}'].append(ndcg)
                
                # Precision - ratio of relevant items among recommendations
                relevant_count = sum(1 for item in recommended_items if item in user_test_items)
                precision = relevant_count / k
                metrics[f'Precision@{k}'].append(precision)
                
                # Recall - ratio of relevant items that were recommended
                recall = relevant_count / len(user_test_items) if len(user_test_items) > 0 else 0
                metrics[f'Recall@{k}'].append(recall)
                
        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
            error_count += 1
            if error_count >= max_errors:
                print(f"Too many errors encountered ({error_count}). Stopping evaluation.")
                if all(len(metrics[f'HR@{k}']) == 0 for k in [5, 10, 20]):
                    # Return zeros if no metrics were calculated
                    return {metric: 0.0 for metric in metrics.keys()}
                break
            continue
    
    # Calculate aggregate metrics
    results = {}
    for metric, values in metrics.items():
        if values:
            results[metric] = float(np.mean(values))
        else:
            results[metric] = 0.0
    
    # Print summary
    print(f"\nEvaluation results for {model_name}:")
    for k in [5, 10, 20]:
        print(f"  @{k}: HR={results.get(f'HR@{k}', 'N/A'):.4f}, NDCG={results.get(f'NDCG@{k}', 'N/A'):.4f}, " +
              f"Precision={results.get(f'Precision@{k}', 'N/A'):.4f}, Recall={results.get(f'Recall@{k}', 'N/A'):.4f}")
    
    return results

def train_and_evaluate(model_class, model_name, dataset_name, hyperparams):
    """Train and evaluate a model on a dataset"""
    print(f"Training {model_name} on {dataset_name}")
    
    # Load processed data
    dataset_info = DATASETS[dataset_name]
    with open(os.path.join(dataset_info["data_path"], "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    df = data['df']
    user_item_matrix = data['user_item_matrix']
    num_users = data['num_users']
    num_items = data['num_items']
    
    # Get hyperparameters
    num_epochs = hyperparams.get('num_epochs', 20)
    batch_size = hyperparams.get('batch_size', 256)
    lr = hyperparams.get('learning_rate', 0.001)
    weight_decay = hyperparams.get('weight_decay', 0.00001)
    patience = hyperparams.get('patience', 5)
    
    # Add timestamp-based features if available
    if 'timestamp' in df.columns:
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        # Extract time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
    
    # Split data into train, validation, and test sets - use stratified sampling by user
    # to ensure each user has items in each split, but handle cases where stratification fails
    try:
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, 
                                          stratify=df['user_id'] if len(df) > 10000 else None)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42,
                                        stratify=temp_df['user_id'] if len(temp_df) > 10000 else None)
    except ValueError as e:
        # If stratification fails, fall back to non-stratified split
        print(f"Stratification failed, using non-stratified split: {e}")
        train_df, temp_df = get_train_test_data(df, test_size=0.3)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create train user-item matrix
    train_user_item = sp.coo_matrix(
        (np.ones(len(train_df)), (train_df['user_id'].values, train_df['item_id'].values)),
        shape=(num_users, num_items)
    ).tocsr()
    
    # Initialize model with hyperparameters
    if model_name == "Scoreformer":
        model = model_class(
            num_layers=hyperparams.get('num_layers', 3),
            d_model=hyperparams.get('d_model', 128),
            num_heads=hyperparams.get('num_heads', 4),
            d_feedforward=hyperparams.get('d_feedforward', 256),
            input_dim=hyperparams.get('input_dim', 64),
            num_targets=1,
            num_users=num_users,
            num_items=num_items,
            dropout=hyperparams.get('dropout', 0.2),
            use_transformer=hyperparams.get('use_transformer', True),
            use_dng=hyperparams.get('use_dng', True),
            use_weights=hyperparams.get('use_weights', True)
        ).to(device)
    elif model_name == "CFUIcA":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            attention_dim=hyperparams.get('attention_dim', 64),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "NCF":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            layers=hyperparams.get('layers', [256, 128, 64]),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "NGCF":
        # Create a properly normalized adjacency matrix for NGCF
        ngcf_adj_matrix = NGCF.create_adj_matrix(train_user_item)  # Use train matrix only
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            adj_matrix=ngcf_adj_matrix,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            layers=hyperparams.get('layers', [128, 64]),
            node_dropout=hyperparams.get('node_dropout', 0.2),
            mess_dropout=hyperparams.get('mess_dropout', 0.2)
        ).to(device)
    elif model_name == "GraphSAGE":
        # Create adjacency lists from user-item matrix
        adj_lists = GraphSAGE.create_adj_lists(train_user_item)  # Use train matrix only
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            adj_lists=adj_lists,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            aggregator_type=hyperparams.get('agg_type', 'mean'),
            num_sample=hyperparams.get('num_samples', 10),
            num_layers=hyperparams.get('num_layers', 2),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "STGCN":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            adj_matrix=train_user_item,  # Use train matrix only
            embedding_dim=hyperparams.get('embedding_dim', 128),
            num_time_steps=hyperparams.get('num_time_steps', 3),
            num_layers=hyperparams.get('num_layers', 2),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "MFBias":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "AutoRec":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            hidden_dim=hyperparams.get('hidden_dim', 256),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
        
        # AutoRec requires the item rating matrix to be set
        # Convert sparse matrix to dense with safe indexing
        try:
            # Transpose the matrix to have items as rows (I-AutoRec approach)
            item_user_matrix = train_user_item.transpose().toarray()
            if item_user_matrix.shape[0] > 0 and item_user_matrix.shape[1] > 0:
                item_ratings = torch.FloatTensor(item_user_matrix).to(device)
                model.set_item_rating_matrix(item_ratings)
            else:
                print(f"Warning: Empty user-item matrix for AutoRec on {dataset_name}")
                return {"error": "Empty user-item matrix for AutoRec"}
        except Exception as e:
            print(f"Error setting up AutoRec item rating matrix: {e}")
            return {"error": f"Failed to initialize AutoRec: {e}"}
    elif model_name == "DMF":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            user_layers=hyperparams.get('user_layers', [256, 128, 64]),
            item_layers=hyperparams.get('item_layers', [256, 128, 64]),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
        
        # DMF requires the interaction matrices to be set
        model.set_interaction_matrices(train_user_item)
    else:
        print(f"Model {model_name} not implemented. Skipping.")
        return {"error": "Model not implemented"}
    
    # Create a simplified version of the dataset to make training easier
    class SimpleDataset(Dataset):
        def __init__(self, df):
            self.user_ids = df['user_id'].values
            self.item_ids = df['item_id'].values
            self.ratings = df['rating'].values
            
        def __len__(self):
            return len(self.user_ids)
        
        def __getitem__(self, idx):
            return {
                'user_id': self.user_ids[idx],
                'item_id': self.item_ids[idx],
                'rating': self.ratings[idx]
            }
    
    # Create training dataset using the simple approach
    train_dataset = SimpleDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation dataset 
    val_dataset = SimpleDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True
    )
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    bpr_criterion = nn.BCEWithLogitsLoss()  # For ranking loss
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training monitoring
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {model_name} with {num_epochs} epochs")
    
    # Track error rate to abort if too many consecutive errors
    consecutive_errors = 0
    max_consecutive_errors = 50  # Abort if this many consecutive errors
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            ratings = batch['rating'].float().to(device)
            
            # Forward pass - rating prediction loss
            try:
                # Compute MSE loss for rating prediction
                predictions = model.predict(user_ids, item_ids)
                rating_loss = mse_criterion(predictions.view(-1), ratings)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                rating_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                total_loss += rating_loss.item()
                batch_count += 1
                consecutive_errors = 0  # Reset error counter on success
            except Exception as e:
                consecutive_errors += 1
                print(f"Error during training batch {batch_idx}: {e}")
                
                # Abort if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Too many consecutive errors ({consecutive_errors}). Aborting training.")
                    # If we have at least some successful batches, continue to evaluation
                    if batch_count > 0:
                        print("Some batches were successful. Proceeding to evaluation.")
                        break
                    else:
                        print("No successful batches. Unable to train model.")
                        return {"error": f"Training failed: {e}"}
                        
                continue
        
        # Skip if no batches were processed successfully
        if batch_count == 0:
            print("No batches were processed in this epoch, skipping validation")
            continue
            
        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        consecutive_errors = 0  # Reset for validation
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")):
                user_ids = batch['user_id'].to(device)
                item_ids = batch['item_id'].to(device)
                ratings = batch['rating'].float().to(device)
                
                try:
                    predictions = model.predict(user_ids, item_ids)
                    batch_loss = mse_criterion(predictions.view(-1), ratings)
                    val_loss += batch_loss.item()
                    val_batch_count += 1
                    consecutive_errors = 0  # Reset on success
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Error during validation batch {batch_idx}: {e}")
                    
                    # Abort if too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Too many consecutive validation errors. Skipping remainder of validation.")
                        break
                    continue
        
        # Skip if no validation batches were processed
        if val_batch_count == 0:
            print("No validation batches were processed, using training loss for early stopping")
            avg_val_loss = avg_train_loss
        else:
            avg_val_loss = val_loss / val_batch_count
            
        val_losses.append(avg_val_loss)
            
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Update learning rate based on validation performance
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), os.path.join("models", f"{model_name}_{dataset_name}_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model for evaluation
    try:
        model.load_state_dict(torch.load(os.path.join("models", f"{model_name}_{dataset_name}_best.pt")))
    except Exception as e:
        print(f"Warning: Could not load best model, using current model instead. Error: {e}")
    
    # Save trained model
    torch.save(model.state_dict(), os.path.join("models", f"{model_name}_{dataset_name}.pt"))
    
    # Evaluate model with improved evaluation procedure
    model.eval()
    results = evaluate_model(
        model, 
        test_df, 
        user_item_matrix, 
        num_items, 
        model_name=model_name,
        max_items_per_user=500  # Increased from 200 to 500 for more thorough evaluation
    )
    
    # Save results
    result_path = os.path.join("results", f"{model_name}_{dataset_name}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    """Run the benchmark"""
    # Datasets to benchmark
    dataset_names = ["ml-100k", "ml-1m", "lastfm"]
    
    # For faster testing, you can use a subset of datasets
    # Uncomment the next line to test with fewer datasets
    # dataset_names = ["ml-100k", "rees46"]  # Use only these datasets for quicker testing
    
    # Models to benchmark - define a subset for faster running
    model_names = ["Scoreformer", "CFUIcA", "NCF", "MFBias", "AutoRec", "DMF"]
    # Uncomment the next line to test with fewer models
    # model_names = ["MFBias", "AutoRec"]  # Test just two models for quick results
    
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Run recommendation system benchmarks')
    parser.add_argument('--datasets', nargs='+', choices=dataset_names, help='Datasets to benchmark')
    parser.add_argument('--models', nargs='+', choices=model_names, help='Models to benchmark')
    parser.add_argument('--quick', action='store_true', help='Run a quick test (5 epochs)')
    args = parser.parse_args()
    
    # Use command line arguments if provided
    if args.datasets:
        dataset_names = args.datasets
    if args.models:
        model_names = args.models
    
    # Process all datasets
    for dataset_name in dataset_names:
        print(f"\n{'='*100}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*100}")
        preprocess_dataset(dataset_name)
    
    # Define model hyperparameters with improved settings
    hyperparams = {
        "Scoreformer": {
            'num_layers': 3,            # Increased from 2
            'd_model': 128,             # Increased from 64
            'num_heads': 4,
            'd_feedforward': 256,       # Increased from 128
            'input_dim': 64,            # Increased from 32
            'dropout': 0.2,             # Slightly increased for better regularization
            'use_transformer': True,
            'use_dng': True,
            'use_weights': True,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,           
            'learning_rate': 0.001,
            'weight_decay': 0.00001,    # L2 regularization
            'batch_size': 256,          # Increased batch size
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2               
        },
        "CFUIcA": {
            'embedding_dim': 128,       # Increased from 64
            'attention_dim': 64,        # Increased from 32
            'dropout': 0.2,             # Slightly increased
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "NCF": {
            'embedding_dim': 128,       # Increased from 64
            'layers': [256, 128, 64],   # Larger layers
            'dropout': 0.2,             # Slightly increased
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "MFBias": {
            'embedding_dim': 128,
            'dropout': 0.2,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5, 
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "AutoRec": {
            'hidden_dim': 256,
            'dropout': 0.2,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "DMF": {
            'user_layers': [256, 128, 64],
            'item_layers': [256, 128, 64],
            'dropout': 0.2,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        }
    }
    
    # Results storage
    all_results = {}
    
    # Train and evaluate on all datasets
    for dataset_name in dataset_names:
        print(f"\n{'='*100}")
        print(f"Benchmarking on dataset: {dataset_name}")
        print(f"{'='*100}")
        
        dataset_results = {}
        
        for model_name in model_names:
            print(f"\n{'-'*50}")
            print(f"Benchmarking {model_name} on {dataset_name}")
            print(f"{'-'*50}")
            
            if model_name not in MODELS:
                print(f"Model {model_name} not found in available models. Skipping.")
                dataset_results[model_name] = {"error": "Model not available"}
                continue
                
            model_class = MODELS[model_name]
            model_params = hyperparams.get(model_name, {})
            
            try:
                eval_results = train_and_evaluate(model_class, model_name, dataset_name, model_params)
                dataset_results[model_name] = eval_results
            except Exception as e:
                print(f"Error training {model_name} on {dataset_name}: {e}")
                dataset_results[model_name] = {"error": str(e)}
        
        all_results[dataset_name] = dataset_results
    
    # Save overall results
    with open("results/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Generate performance comparison table
    create_comparison_table(all_results)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    for dataset_name in dataset_names:
        print(f"\n{dataset_name}:")
        for model_name in model_names:
            if model_name in all_results.get(dataset_name, {}):
                metrics = all_results[dataset_name][model_name]
                if "error" in metrics:
                    print(f"  {model_name}: Error - {metrics['error']}")
                else:
                    print(f"  {model_name}: HR@10={metrics.get('HR@10', 'N/A'):.4f}, NDCG@10={metrics.get('NDCG@10', 'N/A'):.4f}")
            else:
                print(f"  {model_name}: Not evaluated")

def create_comparison_table(results):
    """Create a formatted comparison table from benchmark results"""
    import pandas as pd
    
    # Extract datasets and models
    datasets = list(results.keys())
    models = []
    
    for dataset in datasets:
        models.extend(list(results[dataset].keys()))
    models = list(set(models))  # Remove duplicates
    
    # Create table structure
    table_data = []
    
    # Metrics to include in the table
    metrics = ["HR@10", "NDCG@10"]
    
    # For each dataset
    for dataset in datasets:
        dataset_results = results[dataset]
        
        # For each metric
        for metric in metrics:
            # Row with dataset and metric
            row = {"Dataset": dataset, "Metric": metric}
            
            # Base model for improvement calculation
            base_model = "MFBias"  # Use MFBias as the baseline
            base_value = None
            
            if base_model in dataset_results and "error" not in dataset_results[base_model]:
                base_value = dataset_results[base_model].get(metric, 0.0)
            
            # Get values for each model
            for model in models:
                if model in dataset_results:
                    if "error" in dataset_results[model]:
                        row[model] = "Error"
                    else:
                        value = dataset_results[model].get(metric, 0.0)
                        row[model] = f"{value:.4f}"
                        
                        # Add improvement over baseline
                        if base_value is not None and base_value > 0 and model != base_model:
                            impr = ((value - base_value) / base_value) * 100
                            row[f"{model}_impr"] = f"{impr:.2f}%"
                else:
                    row[model] = "N/A"
            
            table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    df.to_csv("results/comparison_table.csv", index=False)
    print("Comparison table saved to results/comparison_table.csv")
    
    # Format for Markdown/HTML display
    md_table = "# Performance Comparison Table\n\n"
    
    for dataset in datasets:
        md_table += f"## {dataset}\n\n"
        
        # Filter for this dataset
        dataset_df = df[df["Dataset"] == dataset]
        
        # Create table header
        header = "| Metric | " + " | ".join(models) + " |\n"
        separator = "| --- | " + " | ".join(["---" for _ in models]) + " |\n"
        
        md_table += header + separator
        
        # Add rows
        for metric in metrics:
            metric_row = dataset_df[dataset_df["Metric"] == metric].iloc[0]
            row_str = f"| {metric} | "
            
            for model in models:
                value = metric_row.get(model, "N/A")
                improvement = metric_row.get(f"{model}_impr", "")
                
                if improvement and value != "Error" and value != "N/A":
                    row_str += f"{value}<br>({improvement}) | "
                else:
                    row_str += f"{value} | "
            
            md_table += row_str + "\n"
        
        md_table += "\n"
    
    # Save markdown table
    with open("results/comparison_table.md", "w") as f:
        f.write(md_table)
    print("Markdown comparison table saved to results/comparison_table.md")

if __name__ == "__main__":
    main() 


"""
Recommendation System Benchmark Tool

This script provides a comprehensive benchmarking framework for recommendation systems,
comparing multiple state-of-the-art algorithms on standard datasets. The following models 
are included in this benchmark:

1. Scoreformer:
   - A novel transformer-based recommendation model with Direct-Neighborhood-Graph (DNG) scoring
   - Combines transformer attention mechanisms with graph-based representations
   - Features configurable transformer depth, embedding dimensions, and attention heads
   - Modular architecture with options to enable/disable transformer and DNG components

2. CFUIcA (Collaborative Filtering with User-Item Context-aware Attention):
   - Leverages context-aware attention mechanisms to capture complex user-item interactions
   - Combines collaborative filtering with attention weights for improved recommendations
   - Particularly effective for capturing subtle preference patterns

3. STGCN (Spatial-Temporal Graph Convolutional Network):
   - Extends GCN by incorporating temporal dynamics in user-item interactions
   - Processes the evolving graph structure over multiple time steps
   - Applies temporal attention to weight the importance of different time periods
   - Well-suited for recommendation scenarios with time-varying preferences

4. NCF (Neural Collaborative Filtering):
   - Combines matrix factorization with multi-layer perceptrons for recommendation
   - Learns non-linear interactions between user and item latent features
   - Uses embeddings and deep neural networks for flexible modeling
   - Effective for both explicit and implicit feedback scenarios

5. NGCF (Neural Graph Collaborative Filtering):
   - Explicitly encodes the collaborative signal in user-item interactions
   - Exploits high-order connectivity in the user-item graph
   - Propagates embeddings through the interaction graph structure
   - Leverages both direct and higher-order user-item connections

6. GraphSAGE (Graph Sample and Aggregate):
   - Generates node embeddings by sampling and aggregating features from node neighborhoods
   - Scales to large graphs by using neighborhood sampling strategies
   - Supports different aggregation functions (mean, max, LSTM)
   - Effectively captures structural information for recommendation

7. MFBias (Matrix Factorization with Bias):
   - Classic matrix factorization approach with user and item biases
   - Decomposes the user-item interaction matrix into latent factors
   - Includes global, user and item bias terms to model rating deviations
   - Foundational approach that serves as a strong baseline

8. AutoRec (Autoencoder-based Recommendation):
   - Uses autoencoders to learn compact representations of user-item interactions
   - Reconstructs user-item vectors through a bottleneck architecture
   - Captures non-linear relationships without requiring negative sampling
   - Effective for handling sparse interaction data

9. DMF (Deep Matrix Factorization):
   - Extends matrix factorization with deep neural networks
   - Processes user and item interaction vectors with separate networks
   - Projects users and items into a common latent space
   - Captures complex non-linear relationships between users and items
   
This benchmark supports:
- Model training and evaluation on multiple datasets
- Comparison using standard metrics: HR@K and NDCG@K
- Detailed logging and result visualization
- Modular design for easy extension with new models
"""

# Create bench_models directory if it doesn't exist
os.makedirs("bench_models", exist_ok=True)

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Datasets to benchmark
DATASETS = {
    # MovieLens 100K: 100,000 ratings from 943 users on 1,682 movies
    # Contains demographic information for users (age, gender, occupation, zip)
    # Rating scale: 1-5 stars
    # Timestamp information available for temporal analysis
    # Source: GroupLens Research Project at the University of Minnesota
    "ml-100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "filename": "ml-100k.zip",
        "data_path": "data/ml-100k",
        "rating_file_path": "u.data",  # Path inside the extracted directory
        "sep": "\t",
        "header": None,
        "names": ["user_id", "item_id", "rating", "timestamp"],
        "description": "MovieLens 100K dataset with 100,000 ratings (1-5) from 943 users on 1,682 movies."
    },
    
    # MovieLens 1M: 1 million ratings from 6,000 users on 4,000 movies
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "filename": "ml-1m.zip",
        "data_path": "data/ml-1m",
        "rating_file_path": "ratings.dat",  # Path inside the extracted directory
        "sep": "::",
        "header": None,
        "names": ["user_id", "item_id", "rating", "timestamp"],
        "description": "MovieLens 1M dataset with 1 million ratings from 6,000 users on 4,000 movies."
    },
    
    # Last.FM: Music listening data
    "lastfm": {
        "url": "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
        "filename": "hetrec2011-lastfm-2k.zip",
        "data_path": "data/lastfm",
        "rating_file_path": "user_artists.dat",  # Path inside the extracted directory
        "sep": "\t",
        "header": 0,
        "names": ["user_id", "item_id", "weight"],
        "description": "Last.FM dataset with 92,834 artist listening records from 1,892 users."
    },
    
    # REES46 Ecommerce Behavior Dataset
    "rees46": {
        "url": None,  # Local file
        "filename": "events.csv",
        "data_path": "data/rees46",
        "rating_file_path": "REES46/events.csv",  # Path to the original file
        "local_file": True,  # Flag to indicate it's a local file
        "sep": ",",
        "header": 0,
        "names": ["event_time", "event_type", "product_id", "category_id", "category_code", "brand", "price", "user_id", "user_session"],
        "description": "REES46 e-commerce behavior dataset with user interactions with products."
    }
}

# Simple NCF model for comparison
class SimpleNCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, layers=[64, 32], dropout=0.1):
        super(SimpleNCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.ModuleList()
        input_size = 2 * embedding_dim
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size
        self.output_layer = nn.Linear(layers[-1], 1)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        x = torch.cat([user_emb, item_emb], dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        output = self.output_layer(x)
        return output.squeeze()
    
    def predict(self, user_indices, item_indices):
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
        
        # Process in smaller batches
        batch_size = 256
        num_samples = len(user_indices)
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch_users = user_indices[i:i+batch_size]
            batch_items = item_indices[i:i+batch_size]
            with torch.no_grad():
                batch_preds = self.forward(batch_users, batch_items)
            predictions.append(batch_preds)
        
        return torch.cat(predictions, dim=0)

# Models to compare - only using Scoreformer for now
MODELS = {
    # Scoreformer: Novel transformer-based model with Direct-Neighborhood-Graph scoring
    "Scoreformer": Scoreformer,
    
    # CFUIcA: Collaborative Filtering with User-Item Context-aware Attention
    "CFUIcA": CFUIcA,
    
    # STGCN: Spatial-Temporal Graph Convolutional Network for temporal dynamics
    "STGCN": STGCN,
    
    # NCF: Neural Collaborative Filtering with matrix factorization and MLPs
    "NCF": NCF,
    
    # NGCF: Neural Graph Collaborative Filtering using high-order connectivity
    "NGCF": NGCF,
    
    # GraphSAGE: Graph Sample and Aggregate for neighborhood embeddings
    "GraphSAGE": GraphSAGE,
    
    # MFBias: Matrix Factorization with bias terms
    "MFBias": MFBias,
    
    # AutoRec: Autoencoder-based recommendation model
    "AutoRec": AutoRec,
    
    # DMF: Deep Matrix Factorization with neural networks
    "DMF": DMF
}

# Metrics
def hit_ratio(ranked_list, ground_truth):
    """Calculate hit ratio"""
    return 1.0 if ground_truth in ranked_list else 0.0

def ndcg(ranked_list, ground_truth):
    """Calculate NDCG (Normalized Discounted Cumulative Gain)"""
    if ground_truth in ranked_list:
        index = ranked_list.index(ground_truth)
        return np.reciprocal(np.log2(index + 2))
    return 0.0

class RecommendationDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'rating': self.ratings[idx]
        }

def preprocess_dataset(dataset_name):
    """Download and preprocess a dataset"""
    dataset_info = DATASETS[dataset_name]
    
    # Create dataset directory
    os.makedirs(dataset_info["data_path"], exist_ok=True)
    
    # Handle local files differently
    if dataset_info.get("local_file", False):
        ratings_file = dataset_info["rating_file_path"]
        print(f"Using local file for {dataset_name} dataset: {ratings_file}")
    else:
        # Path to the downloaded file
        download_path = os.path.join(dataset_info["data_path"], dataset_info["filename"])
        
        # Download dataset if it doesn't exist and URL is provided
        if dataset_info["url"] and not os.path.exists(download_path):
            print(f"Downloading {dataset_name} dataset...")
            # For Yelp which requires Kaggle authentication, we'll assume manual download
            if dataset_name == "yelp":
                print("Please manually download the Yelp dataset from Kaggle and place it in data/yelp/")
                return None
            urllib.request.urlretrieve(dataset_info["url"], download_path)
        
        # Path to the actual ratings file
        ratings_file = dataset_info["rating_file_path"]
        
        # Extract if it's a zip file and we don't have the ratings file yet
        if download_path.endswith('.zip') and not os.path.exists(os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])):
            print(f"Extracting {dataset_name} dataset...")
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_info["data_path"])
            
            # For some datasets, get the correct path after extraction
            if dataset_name == "lastfm":
                ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
            elif dataset_name == "ml-100k":
                ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
            elif dataset_name == "ml-1m":
                ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
        elif download_path.endswith('.gz') and not os.path.exists(ratings_file):
            with gzip.open(download_path, 'rb') as f_in:
                with open(download_path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
    # Read ratings
    if dataset_name == "yelp":
        # Yelp requires special handling for JSON lines
        df = pd.read_json(os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"]), lines=True)
        df = df[['user_id', 'business_id', 'stars']]
        df.rename(columns={'business_id': 'item_id', 'stars': 'rating'}, inplace=True)
    elif dataset_name == "rees46":
        # Special handling for REES46 dataset
        df = pd.read_csv(ratings_file)
        # Filter for view and cart events as implicit positive feedback
        df = df[df['event_type'].isin(['view', 'cart'])]
        # Create a rating column (cart = 2, view = 1)
        df['rating'] = df['event_type'].apply(lambda x: 2 if x == 'cart' else 1)
        # Select relevant columns
        df = df[['user_id', 'product_id', 'rating']]
        # Rename columns to match our standard
        df.rename(columns={'product_id': 'item_id'}, inplace=True)
        # Take a random sample if the dataset is too large
        if len(df) > 500000:
            df = df.sample(n=500000, random_state=42)
    elif dataset_name == "lastfm":
        # Special handling for LastFM dataset
        lastfm_ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
        if not os.path.exists(lastfm_ratings_file):
            print(f"LastFM ratings file not found at expected path: {lastfm_ratings_file}")
            # Try finding it
            lastfm_files = [f for f in os.listdir(dataset_info["data_path"]) if f.endswith('.dat')]
            print(f"Available .dat files: {lastfm_files}")
            for file in lastfm_files:
                if 'user_artists' in file:
                    lastfm_ratings_file = os.path.join(dataset_info["data_path"], file)
                    print(f"Found LastFM ratings file: {lastfm_ratings_file}")
                    break
        
        df = pd.read_csv(
            lastfm_ratings_file, 
            sep=dataset_info["sep"], 
            header=dataset_info["header"],
            names=dataset_info["names"]
        )
        # Rename 'weight' to 'rating' for consistency
        df.rename(columns={'weight': 'rating'}, inplace=True)
        
        # Convert listen counts (weights) to implicit ratings
        # Normalize weights to a 1-5 scale for consistency with other datasets
        min_weight = df['rating'].min()
        max_weight = df['rating'].max()
        if max_weight > min_weight:  # Avoid division by zero
            # Use log scale for better distribution since listening counts can be highly skewed
            df['rating'] = 1 + 4 * (np.log1p(df['rating']) - np.log1p(min_weight)) / (np.log1p(max_weight) - np.log1p(min_weight))
        else:
            # If all weights are the same, set them to mid-range
            df['rating'] = 3.0
    elif dataset_name == "ml-100k" or dataset_name == "ml-1m":
        # Handle MovieLens datasets
        ml_ratings_file = os.path.join(dataset_info["data_path"], dataset_info["rating_file_path"])
        if not os.path.exists(ml_ratings_file):
            print(f"MovieLens ratings file not found at expected path: {ml_ratings_file}")
            # Try finding the file
            ml_files = [f for f in os.listdir(dataset_info["data_path"])]
            print(f"Files in dataset directory: {ml_files}")
            
            # For ML-100K, look for 'u.data' file
            if dataset_name == "ml-100k":
                subdirs = [d for d in ml_files if os.path.isdir(os.path.join(dataset_info["data_path"], d))]
                for subdir in subdirs:
                    subdir_path = os.path.join(dataset_info["data_path"], subdir)
                    subdir_files = os.listdir(subdir_path)
                    if "u.data" in subdir_files:
                        ml_ratings_file = os.path.join(subdir_path, "u.data")
                        print(f"Found ML-100K ratings file: {ml_ratings_file}")
                        break
            
            # For ML-1M, look for 'ratings.dat' file
            elif dataset_name == "ml-1m":
                subdirs = [d for d in ml_files if os.path.isdir(os.path.join(dataset_info["data_path"], d))]
                for subdir in subdirs:
                    subdir_path = os.path.join(dataset_info["data_path"], subdir)
                    subdir_files = os.listdir(subdir_path)
                    if "ratings.dat" in subdir_files:
                        ml_ratings_file = os.path.join(subdir_path, "ratings.dat")
                        print(f"Found ML-1M ratings file: {ml_ratings_file}")
                        break
        
        df = pd.read_csv(
            ml_ratings_file, 
            sep=dataset_info["sep"], 
            header=dataset_info["header"],
            names=dataset_info["names"]
        )
    else:
        # For other datasets, read from the correct path
        df = pd.read_csv(
            ratings_file, 
            sep=dataset_info["sep"], 
            header=dataset_info["header"],
            names=dataset_info["names"]
        )
    
    # Encode user_ids and item_ids
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    df['item_id'] = item_encoder.fit_transform(df['item_id'])
    
    # Save encoders for later use
    with open(os.path.join(dataset_info["data_path"], "user_encoder.pkl"), "wb") as f:
        pickle.dump(user_encoder, f)
    with open(os.path.join(dataset_info["data_path"], "item_encoder.pkl"), "wb") as f:
        pickle.dump(item_encoder, f)
    
    # Create adjacency matrix for graph algorithms
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    
    # Create user-item interaction matrix
    user_item_matrix = sp.coo_matrix(
        (np.ones(len(df)), (df['user_id'].values, df['item_id'].values)),
        shape=(num_users, num_items)
    ).tocsr()
    
    # Save processed data
    processed_data = {
        'df': df,
        'user_item_matrix': user_item_matrix,
        'num_users': num_users,
        'num_items': num_items
    }
    
    with open(os.path.join(dataset_info["data_path"], "processed_data.pkl"), "wb") as f:
        pickle.dump(processed_data, f)
    
    return processed_data

def get_train_test_data(df, test_size=0.2):
    """Split data into train and test sets"""
    # Group by user to ensure each user has test items
    user_groups = df.groupby('user_id')
    train_data = []
    test_data = []
    
    for user_id, group in user_groups:
        # For each user, sample test_size of their interactions for testing
        # Make sure each user has at least 2 interactions to allow for train/test split
        if len(group) < 2:
            # If user has only one interaction, keep it in training
            train_data.append(group)
            continue
            
        n_test = max(1, int(test_size * len(group)))
        test_indices = np.random.choice(group.index, n_test, replace=False)
        train_indices = list(set(group.index) - set(test_indices))
        
        train_data.append(df.loc[train_indices])
        test_data.append(df.loc[test_indices])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    return train_df, test_df

def evaluate_model(model, test_df, user_item_matrix, num_items, top_k=10, model_name="", max_items_per_user=500):
    """Evaluate a model using HR@K and NDCG@K metrics with more robust methodology"""
    print(f"Evaluating {model_name} with {top_k} recommendations per user")
    
    # Dictionary to store metrics at multiple cutoff points
    metrics = {}
    for k in [5, 10, 20]:
        metrics[f'HR@{k}'] = []
        metrics[f'NDCG@{k}'] = []
        metrics[f'Precision@{k}'] = []
        metrics[f'Recall@{k}'] = []
    
    # Group test data by user for more efficient evaluation
    test_users = test_df['user_id'].unique()
    
    # Use a reasonable sample for evaluation
    if len(test_users) > 200:
        print(f"Sampling 200 users from {len(test_users)} for evaluation")
        np.random.seed(42)  # For reproducibility
        test_users = np.random.choice(test_users, 200, replace=False)
    
    error_count = 0
    max_errors = 15  # Increased tolerance for errors
    
    # Process users in parallel or sequentially
    progress_bar = tqdm(test_users, desc=f"Evaluating {model_name}")
    
    for user_id in progress_bar:
        try:
            # Get ground truth items for this user
            user_test_items = test_df[test_df['user_id'] == user_id]['item_id'].values
            
            if len(user_test_items) == 0:
                continue  # Skip users with no test items
            
            # Get items that the user has not interacted with (in training set)
            user_interactions = user_item_matrix[user_id].indices
            all_items = np.arange(num_items)
            candidate_items = np.setdiff1d(all_items, user_interactions)
            
            # If there are too many candidate items, use a hybrid sampling approach
            if len(candidate_items) > max_items_per_user:
                # Create a candidate pool:
                # 1. Include all ground truth items
                # 2. Include some popular items (top 20%)
                # 3. Include some random items from the rest
                
                # Get item popularity
                item_popularity = np.array(user_item_matrix.sum(axis=0)).flatten()
                item_ranks = np.argsort(-item_popularity)  # Descending order
                
                # Top 20% popular items (excluding already interacted ones)
                num_popular = min(int(max_items_per_user * 0.2), num_items // 5)
                popular_candidates = [item for item in item_ranks[:num_popular] if item not in user_interactions]
                
                # Random items to fill the rest
                remaining_candidates = [item for item in candidate_items if item not in popular_candidates and item not in user_test_items]
                num_random = max_items_per_user - len(popular_candidates) - len(user_test_items)
                
                if num_random > 0 and len(remaining_candidates) > 0:
                    random_candidates = np.random.choice(
                        remaining_candidates, 
                        size=min(num_random, len(remaining_candidates)), 
                        replace=False
                    )
                else:
                    random_candidates = []
                
                # Combine pools
                candidate_items = np.concatenate([
                    popular_candidates,
                    random_candidates,
                    user_test_items
                ])
            
            # Make sure all ground truth items are in the candidates
            for gt_item in user_test_items:
                if gt_item not in candidate_items:
                    candidate_items = np.append(candidate_items, gt_item)
            
            # Convert to tensors
            user_tensor = torch.LongTensor([user_id]).to(device)  # Single user
            item_tensor = torch.LongTensor(candidate_items).to(device)  # All candidate items
            
            # Get predictions in smaller batches to prevent dimension issues
            all_scores = []
            batch_size = 100
            
            for i in range(0, len(item_tensor), batch_size):
                batch_items = item_tensor[i:i+batch_size]
                batch_users = user_tensor.repeat(len(batch_items))
                
                # Make sure both tensors have the same length
                assert len(batch_users) == len(batch_items), "Tensor length mismatch"
                
                with torch.no_grad():  # No gradients needed for evaluation
                    try:
                        batch_scores = model.predict(batch_users, batch_items)
                        all_scores.append(batch_scores)
                    except RuntimeError as e:
                        print(f"Batch error for user {user_id}, items {i}:{i+batch_size}: {e}")
                        # Use zeros as fallback scores for this batch
                        all_scores.append(torch.zeros(len(batch_items), device=device))
            
            # Combine all batches
            scores = torch.cat(all_scores)
            
            # Make sure scores are valid
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(f"NaN or Inf scores detected for user {user_id}, skipping")
                continue
            
            # Sort items by score
            _, indices = torch.sort(scores, descending=True)
            ranked_items = candidate_items[indices.cpu().numpy()]
            
            # Calculate metrics at multiple cutoff points
            for k in [5, 10, 20]:
                if k > len(ranked_items):
                    continue
                    
                # Get top-k items
                recommended_items = ranked_items[:k]
                
                # Hit Ratio - if any ground truth item is in recommendations
                hit = any(item in recommended_items for item in user_test_items)
                metrics[f'HR@{k}'].append(float(hit))
                
                # NDCG - normalized discounted cumulative gain
                dcg = 0
                idcg = 0
                
                # Calculate DCG
                for i, item in enumerate(recommended_items):
                    if item in user_test_items:
                        # Rank is 0-indexed, so +1 for the formula
                        dcg += 1 / np.log2(i + 2)
                
                # Calculate IDCG
                ideal_ranking = min(k, len(user_test_items))
                for i in range(ideal_ranking):
                    idcg += 1 / np.log2(i + 2)
                
                # Avoid division by zero
                if idcg > 0:
                    ndcg = dcg / idcg
                else:
                    ndcg = 0
                
                metrics[f'NDCG@{k}'].append(ndcg)
                
                # Precision - ratio of relevant items among recommendations
                relevant_count = sum(1 for item in recommended_items if item in user_test_items)
                precision = relevant_count / k
                metrics[f'Precision@{k}'].append(precision)
                
                # Recall - ratio of relevant items that were recommended
                recall = relevant_count / len(user_test_items) if len(user_test_items) > 0 else 0
                metrics[f'Recall@{k}'].append(recall)
                
        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
            error_count += 1
            if error_count >= max_errors:
                print(f"Too many errors encountered ({error_count}). Stopping evaluation.")
                if all(len(metrics[f'HR@{k}']) == 0 for k in [5, 10, 20]):
                    # Return zeros if no metrics were calculated
                    return {metric: 0.0 for metric in metrics.keys()}
                break
            continue
    
    # Calculate aggregate metrics
    results = {}
    for metric, values in metrics.items():
        if values:
            results[metric] = float(np.mean(values))
        else:
            results[metric] = 0.0
    
    # Print summary
    print(f"\nEvaluation results for {model_name}:")
    for k in [5, 10, 20]:
        print(f"  @{k}: HR={results.get(f'HR@{k}', 'N/A'):.4f}, NDCG={results.get(f'NDCG@{k}', 'N/A'):.4f}, " +
              f"Precision={results.get(f'Precision@{k}', 'N/A'):.4f}, Recall={results.get(f'Recall@{k}', 'N/A'):.4f}")
    
    return results

def train_and_evaluate(model_class, model_name, dataset_name, hyperparams):
    """Train and evaluate a model on a dataset"""
    print(f"Training {model_name} on {dataset_name}")
    
    # Load processed data
    dataset_info = DATASETS[dataset_name]
    with open(os.path.join(dataset_info["data_path"], "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    df = data['df']
    user_item_matrix = data['user_item_matrix']
    num_users = data['num_users']
    num_items = data['num_items']
    
    # Get hyperparameters
    num_epochs = hyperparams.get('num_epochs', 20)
    batch_size = hyperparams.get('batch_size', 256)
    lr = hyperparams.get('learning_rate', 0.001)
    weight_decay = hyperparams.get('weight_decay', 0.00001)
    patience = hyperparams.get('patience', 5)
    
    # Add timestamp-based features if available
    if 'timestamp' in df.columns:
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        # Extract time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
    
    # Split data into train, validation, and test sets - use stratified sampling by user
    # to ensure each user has items in each split, but handle cases where stratification fails
    try:
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, 
                                          stratify=df['user_id'] if len(df) > 10000 else None)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42,
                                        stratify=temp_df['user_id'] if len(temp_df) > 10000 else None)
    except ValueError as e:
        # If stratification fails, fall back to non-stratified split
        print(f"Stratification failed, using non-stratified split: {e}")
        train_df, temp_df = get_train_test_data(df, test_size=0.3)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create train user-item matrix
    train_user_item = sp.coo_matrix(
        (np.ones(len(train_df)), (train_df['user_id'].values, train_df['item_id'].values)),
        shape=(num_users, num_items)
    ).tocsr()
    
    # Initialize model with hyperparameters
    if model_name == "Scoreformer":
        model = model_class(
            num_layers=hyperparams.get('num_layers', 3),
            d_model=hyperparams.get('d_model', 128),
            num_heads=hyperparams.get('num_heads', 4),
            d_feedforward=hyperparams.get('d_feedforward', 256),
            input_dim=hyperparams.get('input_dim', 64),
            num_targets=1,
            num_users=num_users,
            num_items=num_items,
            dropout=hyperparams.get('dropout', 0.2),
            use_transformer=hyperparams.get('use_transformer', True),
            use_dng=hyperparams.get('use_dng', True),
            use_weights=hyperparams.get('use_weights', True)
        ).to(device)
    elif model_name == "CFUIcA":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            attention_dim=hyperparams.get('attention_dim', 64),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "NCF":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            layers=hyperparams.get('layers', [256, 128, 64]),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "NGCF":
        # Create a properly normalized adjacency matrix for NGCF
        ngcf_adj_matrix = NGCF.create_adj_matrix(train_user_item)  # Use train matrix only
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            adj_matrix=ngcf_adj_matrix,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            layers=hyperparams.get('layers', [128, 64]),
            node_dropout=hyperparams.get('node_dropout', 0.2),
            mess_dropout=hyperparams.get('mess_dropout', 0.2)
        ).to(device)
    elif model_name == "GraphSAGE":
        # Create adjacency lists from user-item matrix
        adj_lists = GraphSAGE.create_adj_lists(train_user_item)  # Use train matrix only
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            adj_lists=adj_lists,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            aggregator_type=hyperparams.get('agg_type', 'mean'),
            num_sample=hyperparams.get('num_samples', 10),
            num_layers=hyperparams.get('num_layers', 2),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "STGCN":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            adj_matrix=train_user_item,  # Use train matrix only
            embedding_dim=hyperparams.get('embedding_dim', 128),
            num_time_steps=hyperparams.get('num_time_steps', 3),
            num_layers=hyperparams.get('num_layers', 2),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "MFBias":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=hyperparams.get('embedding_dim', 128),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
    elif model_name == "AutoRec":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            hidden_dim=hyperparams.get('hidden_dim', 256),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
        
        # AutoRec requires the item rating matrix to be set
        # Convert sparse matrix to dense with safe indexing
        try:
            # Transpose the matrix to have items as rows (I-AutoRec approach)
            item_user_matrix = train_user_item.transpose().toarray()
            if item_user_matrix.shape[0] > 0 and item_user_matrix.shape[1] > 0:
                item_ratings = torch.FloatTensor(item_user_matrix).to(device)
                model.set_item_rating_matrix(item_ratings)
            else:
                print(f"Warning: Empty user-item matrix for AutoRec on {dataset_name}")
                return {"error": "Empty user-item matrix for AutoRec"}
        except Exception as e:
            print(f"Error setting up AutoRec item rating matrix: {e}")
            return {"error": f"Failed to initialize AutoRec: {e}"}
    elif model_name == "DMF":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            user_layers=hyperparams.get('user_layers', [256, 128, 64]),
            item_layers=hyperparams.get('item_layers', [256, 128, 64]),
            dropout=hyperparams.get('dropout', 0.2)
        ).to(device)
        
        # DMF requires the interaction matrices to be set
        model.set_interaction_matrices(train_user_item)
    else:
        print(f"Model {model_name} not implemented. Skipping.")
        return {"error": "Model not implemented"}
    
    # Create a simplified version of the dataset to make training easier
    class SimpleDataset(Dataset):
        def __init__(self, df):
            self.user_ids = df['user_id'].values
            self.item_ids = df['item_id'].values
            self.ratings = df['rating'].values
            
        def __len__(self):
            return len(self.user_ids)
        
        def __getitem__(self, idx):
            return {
                'user_id': self.user_ids[idx],
                'item_id': self.item_ids[idx],
                'rating': self.ratings[idx]
            }
    
    # Create training dataset using the simple approach
    train_dataset = SimpleDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation dataset 
    val_dataset = SimpleDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True
    )
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    bpr_criterion = nn.BCEWithLogitsLoss()  # For ranking loss
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training monitoring
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {model_name} with {num_epochs} epochs")
    
    # Track error rate to abort if too many consecutive errors
    consecutive_errors = 0
    max_consecutive_errors = 50  # Abort if this many consecutive errors
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            ratings = batch['rating'].float().to(device)
            
            # Forward pass - rating prediction loss
            try:
                # Compute MSE loss for rating prediction
                predictions = model.predict(user_ids, item_ids)
                rating_loss = mse_criterion(predictions.view(-1), ratings)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                rating_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                total_loss += rating_loss.item()
                batch_count += 1
                consecutive_errors = 0  # Reset error counter on success
            except Exception as e:
                consecutive_errors += 1
                print(f"Error during training batch {batch_idx}: {e}")
                
                # Abort if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Too many consecutive errors ({consecutive_errors}). Aborting training.")
                    # If we have at least some successful batches, continue to evaluation
                    if batch_count > 0:
                        print("Some batches were successful. Proceeding to evaluation.")
                        break
                    else:
                        print("No successful batches. Unable to train model.")
                        return {"error": f"Training failed: {e}"}
                        
                continue
        
        # Skip if no batches were processed successfully
        if batch_count == 0:
            print("No batches were processed in this epoch, skipping validation")
            continue
            
        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        consecutive_errors = 0  # Reset for validation
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")):
                user_ids = batch['user_id'].to(device)
                item_ids = batch['item_id'].to(device)
                ratings = batch['rating'].float().to(device)
                
                try:
                    predictions = model.predict(user_ids, item_ids)
                    batch_loss = mse_criterion(predictions.view(-1), ratings)
                    val_loss += batch_loss.item()
                    val_batch_count += 1
                    consecutive_errors = 0  # Reset on success
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Error during validation batch {batch_idx}: {e}")
                    
                    # Abort if too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Too many consecutive validation errors. Skipping remainder of validation.")
                        break
                    continue
        
        # Skip if no validation batches were processed
        if val_batch_count == 0:
            print("No validation batches were processed, using training loss for early stopping")
            avg_val_loss = avg_train_loss
        else:
            avg_val_loss = val_loss / val_batch_count
            
        val_losses.append(avg_val_loss)
            
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Update learning rate based on validation performance
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), os.path.join("models", f"{model_name}_{dataset_name}_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model for evaluation
    try:
        model.load_state_dict(torch.load(os.path.join("models", f"{model_name}_{dataset_name}_best.pt")))
    except Exception as e:
        print(f"Warning: Could not load best model, using current model instead. Error: {e}")
    
    # Save trained model
    torch.save(model.state_dict(), os.path.join("models", f"{model_name}_{dataset_name}.pt"))
    
    # Evaluate model with improved evaluation procedure
    model.eval()
    results = evaluate_model(
        model, 
        test_df, 
        user_item_matrix, 
        num_items, 
        model_name=model_name,
        max_items_per_user=500  # Increased from 200 to 500 for more thorough evaluation
    )
    
    # Save results
    result_path = os.path.join("results", f"{model_name}_{dataset_name}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    """Run the benchmark"""
    # Datasets to benchmark
    dataset_names = ["ml-100k", "ml-1m", "lastfm", "rees46"]
    
    # For faster testing, you can use a subset of datasets
    # Uncomment the next line to test with fewer datasets
    # dataset_names = ["ml-100k", "rees46"]  # Use only these datasets for quicker testing
    
    # Models to benchmark - define a subset for faster running
    model_names = ["Scoreformer", "CFUIcA", "NCF", "MFBias", "AutoRec", "DMF"]
    # Uncomment the next line to test with fewer models
    # model_names = ["MFBias", "AutoRec"]  # Test just two models for quick results
    
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Run recommendation system benchmarks')
    parser.add_argument('--datasets', nargs='+', choices=dataset_names, help='Datasets to benchmark')
    parser.add_argument('--models', nargs='+', choices=model_names, help='Models to benchmark')
    parser.add_argument('--quick', action='store_true', help='Run a quick test (5 epochs)')
    args = parser.parse_args()
    
    # Use command line arguments if provided
    if args.datasets:
        dataset_names = args.datasets
    if args.models:
        model_names = args.models
    
    # Process all datasets
    for dataset_name in dataset_names:
        print(f"\n{'='*100}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*100}")
        preprocess_dataset(dataset_name)
    
    # Define model hyperparameters with improved settings
    hyperparams = {
        "Scoreformer": {
            'num_layers': 3,            # Increased from 2
            'd_model': 128,             # Increased from 64
            'num_heads': 4,
            'd_feedforward': 256,       # Increased from 128
            'input_dim': 64,            # Increased from 32
            'dropout': 0.2,             # Slightly increased for better regularization
            'use_transformer': True,
            'use_dng': True,
            'use_weights': True,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,           
            'learning_rate': 0.001,
            'weight_decay': 0.00001,    # L2 regularization
            'batch_size': 256,          # Increased batch size
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2               
        },
        "CFUIcA": {
            'embedding_dim': 128,       # Increased from 64
            'attention_dim': 64,        # Increased from 32
            'dropout': 0.2,             # Slightly increased
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "NCF": {
            'embedding_dim': 128,       # Increased from 64
            'layers': [256, 128, 64],   # Larger layers
            'dropout': 0.2,             # Slightly increased
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "MFBias": {
            'embedding_dim': 128,
            'dropout': 0.2,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5, 
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "AutoRec": {
            'hidden_dim': 256,
            'dropout': 0.2,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        },
        "DMF": {
            'user_layers': [256, 128, 64],
            'item_layers': [256, 128, 64],
            'dropout': 0.2,
            'num_epochs': 20 if not (args.quick if hasattr(args, 'quick') else False) else 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'batch_size': 256,
            'patience': 5 if not (args.quick if hasattr(args, 'quick') else False) else 2
        }
    }
    
    # Results storage
    all_results = {}
    
    # Train and evaluate on all datasets
    for dataset_name in dataset_names:
        print(f"\n{'='*100}")
        print(f"Benchmarking on dataset: {dataset_name}")
        print(f"{'='*100}")
        
        dataset_results = {}
        
        for model_name in model_names:
            print(f"\n{'-'*50}")
            print(f"Benchmarking {model_name} on {dataset_name}")
            print(f"{'-'*50}")
            
            if model_name not in MODELS:
                print(f"Model {model_name} not found in available models. Skipping.")
                dataset_results[model_name] = {"error": "Model not available"}
                continue
                
            model_class = MODELS[model_name]
            model_params = hyperparams.get(model_name, {})
            
            try:
                eval_results = train_and_evaluate(model_class, model_name, dataset_name, model_params)
                dataset_results[model_name] = eval_results
            except Exception as e:
                print(f"Error training {model_name} on {dataset_name}: {e}")
                dataset_results[model_name] = {"error": str(e)}
        
        all_results[dataset_name] = dataset_results
    
    # Save overall results
    with open("results/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Generate performance comparison table
    create_comparison_table(all_results)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    for dataset_name in dataset_names:
        print(f"\n{dataset_name}:")
        for model_name in model_names:
            if model_name in all_results.get(dataset_name, {}):
                metrics = all_results[dataset_name][model_name]
                if "error" in metrics:
                    print(f"  {model_name}: Error - {metrics['error']}")
                else:
                    print(f"  {model_name}: HR@10={metrics.get('HR@10', 'N/A'):.4f}, NDCG@10={metrics.get('NDCG@10', 'N/A'):.4f}")
            else:
                print(f"  {model_name}: Not evaluated")

def create_comparison_table(results):
    """Create a formatted comparison table from benchmark results"""
    import pandas as pd
    
    # Extract datasets and models
    datasets = list(results.keys())
    models = []
    
    for dataset in datasets:
        models.extend(list(results[dataset].keys()))
    models = list(set(models))  # Remove duplicates
    
    # Create table structure
    table_data = []
    
    # Metrics to include in the table
    metrics = ["HR@10", "NDCG@10"]
    
    # For each dataset
    for dataset in datasets:
        dataset_results = results[dataset]
        
        # For each metric
        for metric in metrics:
            # Row with dataset and metric
            row = {"Dataset": dataset, "Metric": metric}
            
            # Base model for improvement calculation
            base_model = "MFBias"  # Use MFBias as the baseline
            base_value = None
            
            if base_model in dataset_results and "error" not in dataset_results[base_model]:
                base_value = dataset_results[base_model].get(metric, 0.0)
            
            # Get values for each model
            for model in models:
                if model in dataset_results:
                    if "error" in dataset_results[model]:
                        row[model] = "Error"
                    else:
                        value = dataset_results[model].get(metric, 0.0)
                        row[model] = f"{value:.4f}"
                        
                        # Add improvement over baseline
                        if base_value is not None and base_value > 0 and model != base_model:
                            impr = ((value - base_value) / base_value) * 100
                            row[f"{model}_impr"] = f"{impr:.2f}%"
                else:
                    row[model] = "N/A"
            
            table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    df.to_csv("results/comparison_table.csv", index=False)
    print("Comparison table saved to results/comparison_table.csv")
    
    # Format for Markdown/HTML display
    md_table = "# Performance Comparison Table\n\n"
    
    for dataset in datasets:
        md_table += f"## {dataset}\n\n"
        
        # Filter for this dataset
        dataset_df = df[df["Dataset"] == dataset]
        
        # Create table header
        header = "| Metric | " + " | ".join(models) + " |\n"
        separator = "| --- | " + " | ".join(["---" for _ in models]) + " |\n"
        
        md_table += header + separator
        
        # Add rows
        for metric in metrics:
            metric_row = dataset_df[dataset_df["Metric"] == metric].iloc[0]
            row_str = f"| {metric} | "
            
            for model in models:
                value = metric_row.get(model, "N/A")
                improvement = metric_row.get(f"{model}_impr", "")
                
                if improvement and value != "Error" and value != "N/A":
                    row_str += f"{value}<br>({improvement}) | "
                else:
                    row_str += f"{value} | "
            
            md_table += row_str + "\n"
        
        md_table += "\n"
    
    # Save markdown table
    with open("results/comparison_table.md", "w") as f:
        f.write(md_table)
    print("Markdown comparison table saved to results/comparison_table.md")

if __name__ == "__main__":
    main() 