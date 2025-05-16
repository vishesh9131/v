import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

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
        
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # L2 regularization
        reg_loss = lambda_val * (torch.norm(u_embeddings) ** 2 + 
                                 torch.norm(pos_i_embeddings) ** 2 + 
                                 torch.norm(neg_i_embeddings) ** 2) / 2
        
        return loss + reg_loss

    @staticmethod
    def create_adj_matrix(user_item_matrix):
        """
        Create an adjacency matrix for the user-item graph
        
        Args:
            user_item_matrix: Sparse user-item interaction matrix
            
        Returns:
            Normalized adjacency matrix with self-connections
        """
        # Get shape
        n_users, n_items = user_item_matrix.shape
        
        # Create adjacency matrix with shape [(n_users + n_items), (n_users + n_items)]
        adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        
        # User-item interactions
        adj_mat[:n_users, n_users:] = user_item_matrix
        adj_mat[n_users:, :n_users] = user_item_matrix.T
        
        # Convert to COO format for efficient calculations
        adj_mat = adj_mat.tocoo()
        
        # Add self-connections
        row_idx = np.arange(0, n_users + n_items)
        col_idx = np.arange(0, n_users + n_items)
        self_connections = sp.coo_matrix((np.ones(n_users + n_items), 
                                         (row_idx, col_idx)), 
                                         shape=(n_users + n_items, n_users + n_items),
                                         dtype=np.float32)
        adj_mat = adj_mat + self_connections
        
        # Get degrees for normalization
        rowsum = np.array(adj_mat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        # Create diagonal degree matrix
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # Normalized adjacency: D^(-1/2) * A * D^(-1/2)
        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        
        return norm_adj.tocsr() 