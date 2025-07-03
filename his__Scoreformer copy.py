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
        
        # Add second set of embeddings specifically for MF component
        self.user_embedding_mf = nn.Embedding(num_users, input_dim) if num_users else None
        self.item_embedding_mf = nn.Embedding(num_items, input_dim) if num_items else None
        
        # Add popularity-aware embeddings
        self.item_popularity = nn.Embedding(num_items, 1) if num_items else None
        
        # Enhanced bias terms with factorization capabilities
        self.user_bias = nn.Embedding(num_users, 1) if num_users else None
        self.item_bias = nn.Embedding(num_items, 1) if num_users else None
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Add higher-order bias modeling
        self.user_bias_factors = nn.Embedding(num_users, 4) if num_users else None
        self.item_bias_factors = nn.Embedding(num_items, 4) if num_items else None
        self.bias_interaction = nn.Linear(8, 1)
        
        # Add direct user-item similarity scoring
        self.direct_similarity = nn.Bilinear(input_dim, input_dim, 1)
        
        # Project input to the model dimension
        self.initial_proj = nn.Sequential(
            nn.Linear(input_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Improved transformer encoder branch
        if self.use_transformer:
            # Add layer normalization for better gradient flow
            self.norm_layer = LayerNorm(d_model)
            
            # Use more advanced transformer encoder with improved attention
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=num_heads, 
                dropout=dropout,
                dim_feedforward=d_feedforward,
                activation=F.gelu,  # Switch to GELU activation (better performance)
                batch_first=True    # Add batch_first for better performance
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Add attention pooling
            self.attention_pooling = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softmax(dim=1)
            )
        else:
            self.encoder = None
            self.norm_layer = None
            self.attention_pooling = None
        
        # Enhanced DNG components with residual connections
        if self.use_dng:
            # Direct scoring component with layer normalization
            self.direct_layer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            
            # More powerful neighborhood modeling with cosine similarity
            self.neighborhood_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.neighborhood_layer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            
            # Add cosine similarity attention
            self.cosine_sim_layer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.Tanh()  # Bounded activation for stable similarity
            )
            
            # Graph structure scoring component with multi-head attention
            self.graph_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.graph_layer = nn.Sequential(
                nn.Linear(d_model, d_model),
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
            
            # Add attention weights for dynamic component contribution
            self.dng_attention = nn.Sequential(
                nn.Linear(d_model * 3, 3),
                nn.Softmax(dim=1)
            )
            
            # Add an additional refinement layer
            self.refinement = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
        else:
            self.direct_layer = None
            self.neighborhood_attn = None
            self.neighborhood_layer = None
            self.cosine_sim_layer = None
            self.graph_attn = None
            self.graph_layer = None
            self.dng_combine = None
            self.dng_attention = None
            self.refinement = None
        
        # Improved weight layer with residual connection
        if self.use_weights:
            self.weight_layer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)  # Lower dropout for this layer
            )
        else:
            self.weight_layer = None
        
        # Add autoencoder component (inspired by AutoRec)
        self.autoencoder_enabled = True
        if self.autoencoder_enabled:
            # Enhanced encoder with bottleneck
            self.encoder_ae = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d_model // 2, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.GELU()
            )
            
            # Enhanced decoder with progressive expansion
            self.decoder_ae = nn.Sequential(
                nn.Linear(d_model // 4, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
        
        # Multiple prediction heads with different architectures
        # Standard MLP head
        self.final_linear1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_targets)
        )
        
        # Direct head with simple linear mapping
        self.final_linear2 = nn.Linear(d_model, num_targets)
        
        # Matrix factorization head - improved with deeper architecture
        self.final_mf = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, num_targets)
        )
        
        # Popularity-aware score boost
        self.popularity_boost = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Ensemble weighting - learnable weights for combining predictions
        self.ensemble_weight = nn.Parameter(torch.ones(4) / 4)  # Now 4 components including popularity
        
        # Add feature interaction matrix
        self.feature_interaction = nn.Bilinear(d_model, d_model, d_model)
        
        # Initialize weights with improved strategies
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with improved initialization"""
        # User and item embeddings - normal distribution
        if self.user_embedding is not None:
            nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        if self.item_embedding is not None:
            nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
            
        # MF-specific embeddings - uniform distribution for diversity
        if self.user_embedding_mf is not None:
            nn.init.uniform_(self.user_embedding_mf.weight, -0.05, 0.05)
        if self.item_embedding_mf is not None:
            nn.init.uniform_(self.item_embedding_mf.weight, -0.05, 0.05)
        
        # Popularity embedding - positive initialization for better Hit Ratio
        if self.item_popularity is not None:
            nn.init.normal_(self.item_popularity.weight, mean=0.1, std=0.01)
            
        # Bias initialization
        if self.user_bias is not None:
            nn.init.zeros_(self.user_bias.weight)
        if self.item_bias is not None:
            nn.init.zeros_(self.item_bias.weight)
            
        # Higher-order bias factors - normal with small std
        if self.user_bias_factors is not None:
            nn.init.normal_(self.user_bias_factors.weight, std=0.005)
        if self.item_bias_factors is not None:
            nn.init.normal_(self.item_bias_factors.weight, std=0.005)
            
        # Apply Xavier/Glorot initialization to linear layers for better gradient flow
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Kaiming initialization for GELU activations
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for sub_m in m:
                    if isinstance(sub_m, nn.Linear) and any(isinstance(sibling, nn.GELU) for sibling in m):
                        nn.init.kaiming_normal_(sub_m.weight, nonlinearity='relu')
    
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
        bias_term = 0
        mf_output = 0
        direct_dot_product = 0
        popularity_score = 0
        
        if user_indices is not None and item_indices is not None and self.user_embedding is not None and self.item_embedding is not None:
            user_emb = self.user_embedding(user_indices)
            item_emb = self.item_embedding(item_indices)
            
            # Direct dot product for simple, fast similarity - important for Hit Ratio
            direct_dot_product = torch.sum(user_emb * item_emb, dim=1, keepdim=True)
            
            # Direct bilinear similarity - captures more complex interactions
            direct_similarity = self.direct_similarity(user_emb, item_emb)
            
            # Get MF-specific embeddings
            user_emb_mf = self.user_embedding_mf(user_indices)
            item_emb_mf = self.item_embedding_mf(item_indices)
            
            # Calculate matrix factorization component
            mf_output = torch.sum(user_emb_mf * item_emb_mf, dim=1, keepdim=True)
            
            # Calculate popularity score - important for hit ratio
            if self.item_popularity is not None:
                item_pop = self.item_popularity(item_indices)
                popularity_score = self.popularity_boost(item_pop)
            
            # Enhanced bias calculation
            if self.user_bias is not None and self.item_bias is not None:
                user_b = self.user_bias(user_indices).squeeze(-1)
                item_b = self.item_bias(item_indices).squeeze(-1)
                global_b = self.global_bias.expand(user_indices.size(0))
                
                # Higher-order bias modeling
                user_bias_factors = self.user_bias_factors(user_indices)
                item_bias_factors = self.item_bias_factors(item_indices)
                bias_factors = torch.cat([user_bias_factors, item_bias_factors], dim=1)
                interaction_bias = self.bias_interaction(bias_factors).squeeze(-1)
                
                # Store bias term - will be added to final prediction
                bias_term = global_b + user_b + item_b + interaction_bias + direct_similarity.squeeze(-1)
                
                # If bias_term is 0-dimensional, expand it
                if bias_term.dim() == 0:
                    bias_term = bias_term.expand(1)
                    
            # Combine user and item embeddings with element-wise product component
            x = torch.cat([user_emb, item_emb], dim=1) if x is None else x
            
        # Improved input projection with layer normalization
        h = self.initial_proj(x)  # [batch_size, d_model]
        
        # Transformer encoder with residual connection and attention pooling
        if self.use_transformer and self.encoder is not None:
            # Apply layer normalization before transformer
            h_normed = self.norm_layer(h) if self.norm_layer is not None else h
                
            # Transformer now uses batch_first=True for better performance
            h_trans = h_normed.unsqueeze(1)  # add sequence dimension: [batch_size, 1, d_model]
            h_trans = self.encoder(h_trans)
            h_trans = h_trans.squeeze(1)    # [batch_size, d_model]
            
            # Residual connection with scaling
            h_trans = 0.8 * h_trans + 0.2 * h
        else:
            h_trans = h
            
        # Enhanced DNG scoring mechanism with attention and residual connections
        if self.use_dng:
            # Direct score - based on node features directly
            d_score = self.direct_layer(h_trans)
            
            # Neighborhood score with enhanced attention
            if self.neighborhood_attn is not None:
                # Reshape for attention operation (now using batch_first=True)
                h_attn = h_trans.unsqueeze(1)  # [batch_size, 1, d_model]
                n_score_attn, _ = self.neighborhood_attn(h_attn, h_attn, h_attn)
                n_score_attn = n_score_attn.squeeze(1)  # [batch_size, d_model]
                
                # Cosine similarity modeling
                h_cosine = self.cosine_sim_layer(h_trans)
                norm_h = F.normalize(h_cosine, p=2, dim=1)
                cosine_sim = torch.mm(norm_h, norm_h.transpose(0, 1))
                sim_weighted = torch.matmul(cosine_sim, h_trans) / (torch.sum(cosine_sim, dim=1, keepdim=True) + 1e-8)
                
                # Combine attention and similarity
                n_score = self.neighborhood_layer(n_score_attn + 0.2 * sim_weighted)
            else:
                n_score = self.neighborhood_layer(h_trans)
            
            # Graph structure score with attention
            if self.graph_attn is not None:
                h_graph = h_trans.unsqueeze(1)  # [batch_size, 1, d_model]
                g_score_attn, _ = self.graph_attn(h_graph, h_graph, h_graph)
                g_score_attn = g_score_attn.squeeze(1)
                g_score = self.graph_layer(g_score_attn)
            else:
                g_score = self.graph_layer(h_trans)
            
            # Dynamic weighting of components based on input
            if self.dng_attention is not None:
                combined_feats = torch.cat([d_score, n_score, g_score], dim=1)
                component_weights = self.dng_attention(combined_feats)
                weighted_d_score = d_score * component_weights[:, 0].unsqueeze(1)
                weighted_n_score = n_score * component_weights[:, 1].unsqueeze(1)
                weighted_g_score = g_score * component_weights[:, 2].unsqueeze(1)
                
                # Increased direct score weight for better Hit Ratio
                h_combined = self.dng_combine(torch.cat([
                    weighted_d_score * 1.2,  # Boost direct scores
                    weighted_n_score, 
                    weighted_g_score
                ], dim=1))
            else:
                # Combine the three scores
                combined_scores = torch.cat([d_score, n_score, g_score], dim=1)
                h_combined = self.dng_combine(combined_scores)
            
            # Apply feature interaction for cross-feature learning
            h_interaction = self.feature_interaction(h_combined, h_trans)
            
            # Apply refinement layer with residual learning
            if self.refinement is not None:
                h_refined = self.refinement(h_combined + 0.2 * h_interaction)
                h = h_refined + 0.7 * h_combined + 0.3 * h_trans  # Weighted residual
            else:
                h = h_combined + 0.3 * h_trans  # Simple residual
        else:
            h = h_trans
            
        # Apply weights if enabled (with residual connection)
        if self.use_weights and self.weight_layer is not None:
            h_weighted = self.weight_layer(h)
            h = 0.7 * h_weighted + 0.3 * h  # Weighted residual connection
        
        # Apply autoencoder component for representation refinement
        if self.autoencoder_enabled:
            h_encoded = self.encoder_ae(h)
            h_decoded = self.decoder_ae(h_encoded)
            
            # Residual connection with original representation
            h = 0.8 * h_decoded + 0.2 * h  # Weighted residual to preserve original info

        # Multiple prediction heads for ensemble effect
        # Head 1: MLP-based prediction
        output1 = self.final_linear1(h)
        
        # Head 2: Linear prediction
        output2 = self.final_linear2(h)
        
        # Head 3: Matrix factorization prediction
        if isinstance(mf_output, torch.Tensor) and mf_output.numel() > 0:
            output3 = mf_output
        else:
            output3 = self.final_mf(user_emb_mf * item_emb_mf)
            
        # Head 4: Direct dot product - crucial for Hit Ratio
        output4 = direct_dot_product * 1.5  # Boost direct similarity for better Hit Ratio
        
        # Normalize ensemble weights with softmax
        ensemble_weights = F.softmax(self.ensemble_weight, dim=0)
        
        # Weighted ensemble with learnable weights - now with 4 components
        output = (output1 * ensemble_weights[0] + 
                 output2 * ensemble_weights[1] + 
                 output3 * ensemble_weights[2] +
                 output4 * ensemble_weights[3])
        
        # Add bias term if available
        if isinstance(bias_term, torch.Tensor) and bias_term.numel() > 0:
            if output.dim() > 1 and bias_term.dim() == 1:
                # Handle dimensions for addition
                bias_term = bias_term.unsqueeze(1)
            output = output + bias_term
        
        # Apply popularity boost - critical for Hit Ratio
        if isinstance(popularity_score, torch.Tensor) and popularity_score.numel() > 0:
            output = output + popularity_score * 0.15  # Add scaled popularity boost
            
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
            
            # Get MF embeddings
            user_emb_mf = self.user_embedding_mf(batch_users)
            item_emb_mf = self.item_embedding_mf(batch_items)
            
            # Concatenate embeddings
            x = torch.cat([user_emb, item_emb], dim=1)
            
            # Forward pass with all components
            batch_predictions = self.forward(
                x=x, 
                user_indices=batch_users, 
                item_indices=batch_items
            )
            
            predictions.append(batch_predictions)
        
        # Combine predictions from all batches
        predictions = torch.cat(predictions, dim=0)
        return predictions.squeeze(-1)
