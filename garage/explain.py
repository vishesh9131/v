import torch
import torch.nn as nn
import torch.nn.functional as F

# Open file for writing
with open('scoreformer_outputs.txt', 'w') as f:
    # Sample input data
    X = torch.randn(32, 100)  # Feature matrix: 32 items with 100 features each
    
    # Create a more meaningful adjacency matrix
    A = torch.zeros(32, 32)
    # Add some random connections (1s) to represent relationships between items
    mask = torch.rand(32, 32) > 0.7  # 30% of connections will be active
    A[mask] = 1.0
    # Make it symmetric (if i is connected to j, j is connected to i)
    A = torch.maximum(A, A.t())
    # Remove self-loops
    A.fill_diagonal_(0)
    
    G = torch.randn(32, 10)   # Graph metrics matrix: 10 different graph metrics per item
    W = torch.randn(32, 32)   # Weight matrix: importance weights between items

    f.write("Input Matrices:\n")
    f.write(f"X (Feature Matrix):\n{X}\n\n")
    f.write(f"A (Adjacency Matrix):\n{A}\n\n")
    f.write(f"G (Graph Metrics):\n{G}\n\n")
    f.write(f"W (Weight Matrix):\n{W}\n\n")

    # Define model parameters
    d_model = 64
    num_targets = 10
    num_layers = 2
    num_heads = 4
    dropout = 0.1

    # Initialize model components
    initial_proj = nn.Linear(X.shape[1], d_model)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    final_linear = nn.Linear(d_model, num_targets)

    # 1. Scoring Phase
    f.write("\n=== SCORING PHASE ===\n\n")

    # A. Direct Score
    direct_score = torch.matmul(A, X)
    f.write("1.A Direct Score:\n")
    f.write(f"{direct_score}\n\n")

    # B. Neighbourhood Similarity
    def compute_jaccard_similarity(A):
        intersection = torch.matmul(A, A.T)
        row_sums = A.sum(dim=1)
        union = row_sums.unsqueeze(0) + row_sums.unsqueeze(1) - intersection
        jaccard = intersection / (union + 1e-8)
        return jaccard

    similarity_matrix = compute_jaccard_similarity(A)
    f.write("1.B Neighbourhood Similarity (Jaccard):\n")
    f.write(f"{similarity_matrix}\n\n")

    # C. Graph Metrics
    f.write("1.C Graph Metrics:\n")
    f.write(f"{G}\n\n")

    # D. Final DNG Scores
    dng_scores = torch.cat([
        direct_score,
        torch.matmul(similarity_matrix, X),
        G
    ], dim=1)
    f.write("1.D Final DNG Scores:\n")
    f.write(f"{dng_scores}\n\n")

    # Initialize DNG projection after computing dng_scores
    dng_proj = nn.Linear(dng_scores.shape[1], d_model)

    # 2. Transformer Phase
    f.write("\n=== TRANSFORMER PHASE ===\n\n")

    # A. Linear Projections
    weighted_projection = initial_proj(torch.matmul(W, X))
    original_projection = initial_proj(X)
    combined_proj = weighted_projection + original_projection
    f.write("2.A Linear Projection [Combined Projection] :\n")
    f.write(f"{combined_proj}\n\n")

    # B. Transformer Encoder
    transformer_input = combined_proj.unsqueeze(0)
    transformer_output = encoder(transformer_input).squeeze(0)
    f.write("2.B Transformer Encoder via Transformer Output:\n")
    f.write(f"{transformer_output}\n\n")

    # C. Element-wise Addition
    dng_scores_projected = dng_proj(dng_scores)
    final_representation = transformer_output + dng_scores_projected
    f.write(" 2.C Element-wise Addition via Final Combined Representation:\n")
    f.write(f"{final_representation}\n\n")

    # D. Final Output
    final_combined = F.relu(final_representation)
    output = final_linear(final_combined)
    f.write("2.D Final Output:\n")
    f.write(f"{output}\n")

print("Results have been saved to 'scoreformer_outputs.txt'")