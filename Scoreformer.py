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


'''

                +--------------------------------------+
                |      Scoreformer Model Architecture  |
                +--------------------------------------+
                                  |
            +---------------------+--------------------+
            |                                          |
      [User Indices]                             [Item Indices]
            |                                          |
  +---------+------------------------------------------+---------+
  |         |                                          |         |
  |         |                                          |         |
  | PATH A: DNG / GNN Pathway                          | PATH B: Matrix Factorization Pathway
  | ===========================                          | ==================================
  |                                                      |
  | [User/Item Embeddings]                             | [MF User/Item Embeddings]
  |         |                                          |         |
  | [Concat & Project] -> h_batch                      | [Dot Product] ---+
  |         |                                          |                  |
  | [Optional Transformer] -> h_dng                    | [User/Item Biases] -+
  |         |                                          |         |           |
  |         |                                          |   [Sum Biases]----+
  | +-------+--------------------------------------+   |                   |
  | |                                              |   |                   |
  | +->[Direct Score]--------------+               |   +-----------------> [output_mf]
  | |                              |               |
  | +->[Neighbour Score]------------+->[Dynamic    |
  | |  (from full graph GNN)       |  Attention   |
  | |                              |  Combiner]----+-->[h_final]->[Final Head]->[output_dng]
  | +->[Graph Structure Score]------+
  |    (from h_dng + metrics)      |
  |                                |
  +--------------------------------+

            |                                          |
            |                                          |
      [output_dng]                               [output_mf]
            |                                          |
            +--------------[Learnable                ]--+
                           [Ensemble Weighting]
                                  |
                                  |
                         +--------+---------+
                         | Final Prediction |
                         +------------------+
'''



# Helper GNN Layer for Neighbor Score Calculation
class GNNLayer(nn.Module):
    """
    A simple Graph Convolutional Layer for PROPAGATION ONLY (no linear transformation).
    This is used to calculate the Neighbour Score by aggregating features from neighbors.
    """
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        # This layer is now parameter-free for pure propagation, inspired by LightGCN.

    def forward(self, node_features, edge_index):
        """
        Performs message passing by summing neighbor features.
        """
        source_nodes, dest_nodes = edge_index
        neighbor_features = node_features[source_nodes]
        
        aggregated_features = torch.zeros_like(node_features)
        aggregated_features.index_add_(0, dest_nodes, neighbor_features)

        return aggregated_features

class SimpleSSMBlock(nn.Module):
    """
    A simplified Mamba-style State Space Model block.
    This uses a simplified SSM to model the combined feature vector.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_state = 16  # Internal state dimension
        self.d_conv = 4    # Convolution kernel size

        # Linear projections
        self.in_proj = nn.Linear(d_model, 2 * d_model) # Project for both conv and ssm
        self.out_proj = nn.Linear(d_model, d_model)
        self.ssm_out_proj = nn.Linear(self.d_state, d_model)

        # 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=self.d_conv, 
            padding=self.d_conv - 1,
            groups=d_model # Depthwise convolution
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32)).unsqueeze(1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, self.d_state) # Not used in simplified version but kept for structure
        self.C_proj = nn.Linear(d_model, self.d_state) # Not used in simplified version

    def forward(self, x):
        # x is [batch, d_model]
        # Treat d_model as the sequence length for the 1D conv
        x = x.unsqueeze(2) # [batch, d_model, 1]

        # Input projection
        x_conv, x_ssm = self.in_proj(x.squeeze(-1)).chunk(2, dim=1)

        # Convolutional path
        x_conv = x_conv.unsqueeze(-1)
        x_conv = self.conv1d(x_conv)[:, :, :1] # Causal padding
        x_conv = F.gelu(x_conv)

        # SSM path (simplified selective scan)
        A = -torch.exp(self.A_log.float()) # [d_state, 1]
        dt = torch.exp(self.dt_proj(x_ssm)) # [batch, d_model]

        # Discretize A. For a single step, this is simple.
        # This is a highly simplified scan for a non-sequential input.
        delta_A = torch.exp(dt.mean(dim=1).unsqueeze(-1).unsqueeze(-1) * A) # [batch, d_state, 1]
        
        # FIX: Replace random noise with a projection of the input (like the 'Bu' term in Mamba)
        # This makes the state dependent on the input features.
        projected_input = self.B_proj(x_ssm).unsqueeze(-1) # [batch, d_state, 1]
        h = delta_A * projected_input
        
        # FIX: Project the SSM state 'h' to the correct dimension before gating
        y_ssm = self.ssm_out_proj(h.squeeze(-1))
        
        # Combine paths
        y = x_conv.squeeze(-1) * y_ssm
        
        # Add residual connection and final projection
        y = y + x.squeeze(-1) * self.D
        y = self.out_proj(y)
        
        return y

class Scoreformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        input_dim,
        num_targets,
        num_users,
        num_items,
        dropout=0.1,
        use_dng=True
    ):
        super(Scoreformer, self).__init__()
        self.use_dng = use_dng
        self.num_users = num_users
        self.num_items = num_items
        self.input_dim = input_dim
        self.d_model = d_model
        
        # For recommendation tasks - embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, input_dim)
        self.item_embedding = nn.Embedding(num_items, input_dim)
        
        # Add second set of embeddings specifically for MF component
        self.user_embedding_mf = nn.Embedding(num_users, input_dim)
        self.item_embedding_mf = nn.Embedding(num_items, input_dim)
        
        # Add popularity-aware embeddings
        self.item_popularity = nn.Embedding(num_items, 1)
                # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Project input to the model dimension
        # Input is concatenation of user and item embeddings, so input_dim * 2
        self.initial_proj = nn.Sequential(
            nn.Linear(input_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # DNG (Direct, Neighbour, Graph) components
        if self.use_dng:
            # Direct scoring component (processes node's own features)
            self.direct_layer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            
            # Use a multi-layer GNN encoder for neighbor propagation
            self.gnn_layers = nn.ModuleList([GNNLayer(d_model, d_model) for _ in range(3)]) # 3-layer GNN
            
            # Graph structure scoring component
            # It will process the node's features PLUS its structural features
            self.graph_structural_feature_dim = 1 # Dimension of raw structural features (e.g., PageRank is 1-dim)
            self.graph_feature_projection_dim = 16  # Project structural features to this size
            self.graph_feature_proj = nn.Linear(self.graph_structural_feature_dim, self.graph_feature_projection_dim)
            
            self.graph_layer = nn.Sequential(
                nn.Linear(d_model + self.graph_feature_projection_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            
            # Combine the three scores with weighted attention
            self.dng_combine = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
            # Add the SSM block to process the combined scores
            self.ssm_block = SimpleSSMBlock(d_model=d_model)
            
            # Attention weights for dynamic component contribution
            self.dng_attention = None
        else:
            self.direct_layer = None
            self.gnn_layers = None
            self.graph_layer = None
            self.dng_combine = None
            self.ssm_block = None
        
        # Simplified prediction head
        self.final_linear = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_targets)
        )
        
        # Ensemble weighting - learnable weights for combining GNN and MF predictions
        self.ensemble_weight = nn.Parameter(torch.ones(2) / 2)
        
        # Initialize weights with improved strategies
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with improved initialization"""
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.uniform_(self.user_embedding_mf.weight, -0.05, 0.05)
        nn.init.uniform_(self.item_embedding_mf.weight, -0.05, 0.05)
        nn.init.normal_(self.item_popularity.weight, mean=0.1, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
            
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):
                for sub_m in m:
                    if isinstance(sub_m, nn.Linear) and any(isinstance(sibling, nn.GELU) for sibling in m):
                        nn.init.kaiming_normal_(sub_m.weight, nonlinearity='relu')
    
    def forward(self, user_indices, item_indices, edge_index, graph_structural_features):
        """
        Forward pass through Scoreformer.
        
        Args:
            user_indices (Tensor): [batch_size] - User indices for the batch.
            item_indices (Tensor): [batch_size] - Item indices for the batch.
            edge_index (Tensor): [2, num_edges] - The COO graph structure for the entire graph.
            graph_structural_features (Tensor): [num_nodes, structural_feature_dim] - Pre-computed structural features for all nodes.
        """
        
        # --- 1. Initial Embeddings and Projections ---
        
        # Get embeddings for the current batch
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        # Create the input for the DNG/Transformer part by concatenating user and item embeddings
        x_batch = torch.cat([user_emb, item_emb], dim=1)
        
        # Project batch features to d_model
        h_batch = self.initial_proj(x_batch)  # [batch_size, d_model]
        
        # --- 2. Matrix Factorization and Bias Path (Parallel to DNG) ---
        
        user_emb_mf = self.user_embedding_mf(user_indices)
        item_emb_mf = self.item_embedding_mf(item_indices)
        mf_output = torch.sum(user_emb_mf * item_emb_mf, dim=1, keepdim=True)
        
        user_b = self.user_bias(user_indices).squeeze(-1)
        item_b = self.item_bias(item_indices).squeeze(-1)
        global_b = self.global_bias.expand(user_indices.size(0))
        bias_term = (global_b + user_b + item_b).unsqueeze(1)
        
        # --- 3. Core DNG Block ---
        
        h_dng = h_batch
        
        if self.use_dng:
            # To compute scores, we need features for ALL nodes for message passing
            all_user_features = self.user_embedding.weight
            all_item_features = self.item_embedding.weight
            
            # This is a simplification. A real implementation might need a more complex initial projection
            # for all nodes, which can be memory intensive. For now, we assume input_dim == d_model
            # or we would need a way to project all_user/item_features to d_model.
            # Let's use the raw embeddings as the base for GNN.
            # A more robust model would project these all_user/item_features to d_model first.
            all_node_features = torch.cat([all_user_features, all_item_features], dim=0)

            # Direct score - based on node features directly (from the batch)
            d_score = self.direct_layer(h_dng)
            
            # Neighbour score - propagated through multiple GNN layers
            # This implements a simplified LightGCN-style layer combination
            all_embeddings = [all_node_features]
            current_features = all_node_features
            for layer in self.gnn_layers:
                # The GNN layers here are used for propagation.
                # The gradients will flow back to the initial `all_node_features` (i.e., user/item embeddings)
                current_features = layer(current_features, edge_index)
                all_embeddings.append(current_features)
            
            # Combine embeddings from all layers (mean pooling)
            all_n_scores = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
            
            # Select the neighbor scores for the current batch
            # Item indices need to be offset by the number of users
            user_n_scores = all_n_scores[user_indices]
            item_n_scores = all_n_scores[item_indices + self.num_users]
            n_score = (user_n_scores + item_n_scores) / 2 # Simple averaging

            # Graph structure score
            # Get structural features for the batch of users and items
            user_struct_feats = graph_structural_features[user_indices]
            item_struct_feats = graph_structural_features[item_indices + self.num_users]
            struct_feats_raw = (user_struct_feats + item_struct_feats) / 2 # Simple averaging, shape: [batch, 1]
            struct_feats_proj = self.graph_feature_proj(struct_feats_raw) # Project to a learnable space
            
            g_input = torch.cat([h_dng, struct_feats_proj], dim=1)
            g_score = self.graph_layer(g_input)
            
            # Combine the three scores by simple concatenation before the final projection.
            combined_scores = torch.cat([d_score, n_score, g_score], dim=1)
            h_combined = self.dng_combine(combined_scores)
            h_final = h_combined + 0.3 * h_dng # Final residual

            # Pass through the SSM block for final feature refinement
            h_final = self.ssm_block(h_final)
        else:
            h_final = h_dng
            
        # --- 4. Final Prediction ---
        
        # DNG-based prediction
        output_dng = self.final_linear(h_final)
        
        # MF-based prediction
        output_mf = mf_output + bias_term
        
        # Weighted ensemble with learnable weights
        ensemble_weights = F.softmax(self.ensemble_weight, dim=0)
        output = output_dng * ensemble_weights[0] + output_mf * ensemble_weights[1]
            
        # FIX: Return embeddings for contrastive loss alongside the final score
        return output, user_n_scores, item_n_scores
    
    def predict(self, user_indices, item_indices, edge_index, graph_structural_features):
        """
        Make predictions for recommendation tasks.
        This function now requires the full graph context.
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            # For prediction, we only need the score, not the embeddings
            predictions, _, _ = self.forward(user_indices, item_indices, edge_index, graph_structural_features)
        
        return predictions.squeeze(-1)





# class Scoreformer(nn.Module):
#     def __init__(self, num_layers=2, d_model=64, num_heads=4, d_feedforward=256, 
#                  input_dim=None, categorical_features=None, numerical_features=None, 
#                  embedding_dims=None, num_weights=10, use_weights=True, dropout=0.1,
#                  normalization='batch', attention_type='dot_product'):
#         """
#         Enhanced Scoreformer with support for mixed data types and flexible graph handling
        
#         Args:
#             num_layers: Number of transformer layers
#             d_model: Dimension of the model
#             num_heads: Number of attention heads
#             d_feedforward: Dimension of the feedforward network
#             input_dim: Legacy input dimension (for backward compatibility)
#             categorical_features: List of categorical feature names and their cardinalities
#                                  e.g., [('gender', 2), ('occupation', 10)]
#             numerical_features: List of numerical feature names
#             embedding_dims: Dictionary mapping feature names to embedding dimensions
#             num_weights: Number of feature weights
#             use_weights: Whether to use feature weighting
#             dropout: Dropout rate
#             normalization: Type of normalization ('batch', 'layer', or 'none')
#             attention_type: Type of attention mechanism ('dot_product' or 'additive')
#         """
#         super(Scoreformer, self).__init__()
        
#         self.num_weights = num_weights
#         self.use_weights = use_weights
#         self.d_model = d_model
#         self.normalization = normalization
#         self.attention_type = attention_type
        
#         # Set up feature processing
#         self.categorical_features = categorical_features or []
#         self.numerical_features = numerical_features or []
#         self.embedding_dims = embedding_dims or {}
        
#         # Determine input dimension if not explicitly provided.
#         # For numerical features we use their count; for categorical features we use their embedding dimensions.
#         if input_dim is None:
#             numeric_dim = len(self.numerical_features) if self.numerical_features is not None else 0
#             cat_dim = 0
#             for feature_name, cardinality in (self.categorical_features or []):
#                 # Use the provided embedding dimension or default to min(50, (cardinality+1)//2)
#                 embedding_dim = self.embedding_dims.get(feature_name, min(50, (cardinality+1)//2))
#                 cat_dim += embedding_dim
#             self.input_dim = numeric_dim + cat_dim
#         else:
#             self.input_dim = input_dim
        
#         # Feature embeddings for categorical variables
#         self.embeddings = nn.ModuleDict()
#         for feature_name, cardinality in self.categorical_features:
#             embedding_dim = self.embedding_dims.get(feature_name, min(50, (cardinality + 1) // 2))
#             self.embeddings[feature_name] = Embedding(cardinality, embedding_dim)
        
#         # Input projection
#         self.input_linear = Linear(self.input_dim, d_model)
#         nn.init.xavier_uniform_(self.input_linear.weight, gain=1.0)
        
#         # Add this line - restore the separate projection for DNG scores
#         self.dng_projection = Linear(self.input_dim, d_model)
#         nn.init.xavier_uniform_(self.dng_projection.weight, gain=1.0)
        
#         # Transformer encoder
#         self.encoder_layer = TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=num_heads, 
#             dim_feedforward=d_feedforward, 
#             dropout=dropout, 
#             batch_first=True
#         )
#         self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
#         # Output layers
#         self.pre_output = Linear(d_model, d_model)
#         nn.init.xavier_uniform_(self.pre_output.weight, gain=1.0)
        
#         self.output_linear = Linear(d_model, 1)
#         nn.init.xavier_uniform_(self.output_linear.weight, gain=1.0)
        
#         # Regularization and normalization
#         self.dropout = Dropout(dropout)
        
#         if normalization == 'batch':
#             self.norm = nn.BatchNorm1d(d_model)
#         elif normalization == 'layer':
#             self.norm = LayerNorm(d_model)
#         else:
#             self.norm = nn.Identity()
        
#         # Feature weighting
#         if self.use_weights:
#             self.weight_linears = ModuleList([Linear(self.input_dim, d_model) for _ in range(num_weights)])
#             for layer in self.weight_linears:
#                 nn.init.xavier_uniform_(layer.weight, gain=1.0)

#     def process_features(self, features):
#         """
#         Process mixed-type feature input
        
#         Args:
#             features: Dictionary with 'categorical' and 'numerical' keys
#                      or tensor for backward compatibility
        
#         Returns:
#             Processed feature tensor
#         """
#         if isinstance(features, dict):
#             numerical_values = features.get('numerical', torch.tensor([]))
            
#             # Process categorical features
#             categorical_embeddings = []
#             for feature_name, _ in self.categorical_features:
#                 if feature_name in features.get('categorical', {}):
#                     embedded = self.embeddings[feature_name](features['categorical'][feature_name])
#                     categorical_embeddings.append(embedded)
            
#             # Combine all features
#             all_features = []
#             if len(categorical_embeddings) > 0:
#                 all_features.extend(categorical_embeddings)
#             if numerical_values.numel() > 0:
#                 all_features.append(numerical_values)
            
#             # Concatenate everything along the feature dimension
#             if len(all_features) > 1:
#                 return torch.cat(all_features, dim=-1)
#             elif len(all_features) == 1:
#                 return all_features[0]
#             else:
#                 return torch.zeros((features.get('batch_size', 1), self.input_dim), 
#                                   device=next(self.parameters()).device)
#         else:
#             # Legacy mode: if a tensor is passed and its feature dimension does not match,
#             # assume the tensor contains only numerical features and pad with zeros.
#             if features.shape[1] != self.input_dim:
#                 diff = self.input_dim - features.shape[1]
#                 if diff > 0:
#                     pad = torch.zeros(features.shape[0], diff, device=features.device, dtype=features.dtype)
#                     features = torch.cat([features, pad], dim=-1)
#                 else:
#                     features = features[:, :self.input_dim]
#             return features

#     def compute_neighborhood_similarity(self, adjacency_matrix, x, alpha=0.5, normalization='symmetric'):
#         """
#         Enhanced neighborhood similarity with configurable parameters
        
#         Args:
#             adjacency_matrix: Graph adjacency matrix
#             x: Node features
#             alpha: Teleport probability for PageRank-like weighting
#             normalization: Type of normalization ('symmetric', 'row', or 'column')
        
#         Returns:
#             Neighborhood similarity scores
#         """
#         # Handle empty or invalid adjacency matrix
#         if adjacency_matrix.numel() == 0 or adjacency_matrix.sum() == 0:
#             return torch.zeros_like(x)
            
#         # Convert to binary for similarity calculation
#         binary_adj = (adjacency_matrix > 0).float()
        
#         # Compute intersection of neighborhoods
#         intersection = binary_adj @ binary_adj.T
        
#         # Get node degrees
#         row_sums = binary_adj.sum(dim=1, keepdim=True)
#         col_sums = binary_adj.sum(dim=0, keepdim=True)
        
#         # Compute union for Jaccard similarity
#         union = row_sums + col_sums.T - intersection
        
#         # Avoid division by zero with better epsilon handling
#         epsilon = 1e-6
#         safe_union = union.clone()
#         safe_union[safe_union < epsilon] = epsilon
        
#         # Compute similarity with selected normalization
#         if normalization == 'symmetric':
#             similarity = intersection / safe_union
#         elif normalization == 'row':
#             # Row-normalized similarity (each row sums to 1 except for isolated nodes)
#             safe_row_sums = row_sums.clone()
#             safe_row_sums[safe_row_sums < epsilon] = epsilon
#             similarity = intersection / safe_row_sums
#         elif normalization == 'column':
#             # Column-normalized similarity
#             safe_col_sums = col_sums.T.clone()
#             safe_col_sums[safe_col_sums < epsilon] = epsilon
#             similarity = intersection / safe_col_sums
#         else:
#             similarity = intersection / safe_union
        
#         # Apply teleport probability (PageRank-like)
#         identity_matrix = torch.eye(similarity.size(0), device=similarity.device)
#         similarity = (1 - alpha) * similarity + alpha * identity_matrix
        
#         # Propagate features through similarity matrix
#         return similarity @ x

#     def project_graph_metrics(self, graph_metrics, target_dim):
#         """
#         Improved projection of graph metrics to target dimension
        
#         Args:
#             graph_metrics: Graph metric features
#             target_dim: Target dimension
            
#         Returns:
#             Projected graph metrics
#         """
#         # Handle empty metrics
#         if graph_metrics.numel() == 0:
#             return torch.zeros((graph_metrics.size(0), target_dim), 
#                               device=graph_metrics.device)
        
#         # Use adaptive pooling for flexible dimension handling
#         if graph_metrics.size(1) == target_dim:
#             # No change needed
#             return graph_metrics
#         elif graph_metrics.size(1) < target_dim:
#             # Use a learned projection to expand dimensions
#             if not hasattr(self, 'graph_metric_expander'):
#                 self.graph_metric_expander = Linear(
#                     graph_metrics.size(1), target_dim, 
#                     device=graph_metrics.device
#                 )
#                 nn.init.xavier_uniform_(self.graph_metric_expander.weight, gain=1.0)
#             return self.graph_metric_expander(graph_metrics)
#         else:
#             # Use a learned projection to reduce dimensions
#             if not hasattr(self, 'graph_metric_reducer'):
#                 self.graph_metric_reducer = Linear(
#                     graph_metrics.size(1), target_dim,
#                     device=graph_metrics.device
#                 )
#                 nn.init.xavier_uniform_(self.graph_metric_reducer.weight, gain=1.0)
#             return self.graph_metric_reducer(graph_metrics)

#     def forward(self, x, adjacency_matrix=None, graph_metrics=None, weights=None):
#         """
#         Forward pass with enhanced flexibility
        
#         Args:
#             x: Input features (tensor or dictionary)
#             adjacency_matrix: Graph adjacency matrix (optional)
#             graph_metrics: Graph metrics (optional)
#             weights: Feature weights (optional)
            
#         Returns:
#             Recommendation scores
#         """
#         # Process inputs with potential mixed types
#         x = self.process_features(x)
#         batch_size, input_dim = x.shape
        
#         # Handle missing adjacency matrix
#         if adjacency_matrix is None or adjacency_matrix.numel() == 0:
#             adjacency_matrix = torch.zeros((batch_size, batch_size), device=x.device)
#         adjacency_matrix = adjacency_matrix.float()
        
#         # Handle missing graph metrics
#         if graph_metrics is None or graph_metrics.numel() == 0:
#             graph_metrics = torch.zeros((batch_size, input_dim), device=x.device)
#         graph_metrics = graph_metrics.float()
        
#         # Calculate direct connections score
#         direct_scores = adjacency_matrix @ x
        
#         # Calculate neighborhood similarity score
#         neighborhood_similarity = self.compute_neighborhood_similarity(
#             adjacency_matrix, x, alpha=0.5, normalization='symmetric'
#         )
        
#         # Calculate graph structure score
#         graph_metrics_projected = self.project_graph_metrics(graph_metrics, input_dim)
#         graph_structure_scores = graph_metrics_projected * x

#         # Combine DNG scores and project
#         dng_scores = direct_scores + neighborhood_similarity + graph_structure_scores
#         dng_scores = self.dng_projection(dng_scores)
        
#         # Process input through transformer with optional weighting
#         if self.use_weights and weights is not None:
#             weighted_x = torch.zeros(batch_size, self.d_model, device=x.device)
#             # Handle potential weight dimension mismatch
#             weight_cols = min(weights.shape[1], self.num_weights)
#             for i in range(weight_cols):
#                 weight_index = min(i, len(self.weight_linears) - 1)
#                 projected_x = self.weight_linears[weight_index](x)
#                 weighted_x += projected_x * weights[:, i:i+1]
#             transformer_input = weighted_x
#         else:
#             transformer_input = self.input_linear(x)

#         # Apply normalization
#         transformer_input = self.norm(transformer_input)
        
#         # Pass through transformer
#         transformer_output = self.transformer_encoder(transformer_input.unsqueeze(1)).squeeze(1)
        
#         # Combine and produce final output
#         combined = transformer_output + dng_scores
#         combined = self.dropout(combined)
#         output = self.pre_output(combined)
#         output = F.relu(output)
#         output = self.output_linear(output)
#         output = torch.sigmoid(output)
        
#         return output.squeeze(-1)
    
#     def explain(self, x, adjacency_matrix=None, graph_metrics=None, weights=None, top_k=5):
#         """
#         Generate explanations for model predictions
        
#         Args:
#             x: Input features
#             adjacency_matrix: Graph adjacency matrix
#             graph_metrics: Graph metrics
#             weights: Feature weights
#             top_k: Number of top explanatory factors to return
            
#         Returns:
#             Predictions and explanations
#         """
#         # Store intermediate representations
#         with torch.no_grad():
#             # Process inputs
#             proc_x = self.process_features(x)
#             batch_size, input_dim = proc_x.shape
            
#             # Handle missing adjacency matrix
#             if adjacency_matrix is None or adjacency_matrix.numel() == 0:
#                 adjacency_matrix = torch.zeros((batch_size, batch_size), device=proc_x.device)
#             adjacency_matrix = adjacency_matrix.float()
            
#             # Handle missing graph metrics
#             if graph_metrics is None or graph_metrics.numel() == 0:
#                 graph_metrics = torch.zeros((batch_size, input_dim), device=proc_x.device)
#             graph_metrics = graph_metrics.float()
            
#             # Calculate component scores
#             direct_scores = adjacency_matrix @ proc_x
#             neighborhood_similarity = self.compute_neighborhood_similarity(
#                 adjacency_matrix, proc_x, alpha=0.5, normalization='symmetric'
#             )
#             graph_metrics_projected = self.project_graph_metrics(graph_metrics, input_dim)
#             graph_structure_scores = graph_metrics_projected * proc_x
            
#             # Get final predictions
#             predictions = self.forward(x, adjacency_matrix, graph_metrics, weights)
            
#             # Calculate contribution scores for explanation
#             contributions = {
#                 'direct_connections': (direct_scores.abs().mean(dim=1) / (direct_scores.abs().mean(dim=1).sum() + 1e-8)).tolist(),
#                 'neighborhood_similarity': (neighborhood_similarity.abs().mean(dim=1) / (neighborhood_similarity.abs().mean(dim=1).sum() + 1e-8)).tolist(),
#                 'graph_metrics': (graph_structure_scores.abs().mean(dim=1) / (graph_structure_scores.abs().mean(dim=1).sum() + 1e-8)).tolist()
#             }
            
#             return {
#                 'predictions': predictions.tolist(),
#                 'contributions': contributions
#             }
