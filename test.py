print("hello world 1")
import os
print("hello world 1.1")

import torch
print("hello world 2")
def pairwise_distance_matrix(x):
    """
    Calculate the pairwise distance matrix for a batch of vectors.
    
    Args:
        x (torch.Tensor): Input tensor of shape (n, m), where n is the number of points and m is the dimensionality.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape (n, n).
    """
    # Compute the squared norms of each row (shape: [n, 1])
    squared_norms = torch.sum(x**2, dim=1, keepdim=True)
    
    # Compute the pairwise squared distances (broadcasting)
    pairwise_squared_distances = squared_norms - 2 * torch.matmul(x, x.T) + squared_norms.T
    
    # Avoid negative distances due to numerical issues (clip to 0)
    pairwise_squared_distances = torch.clamp(pairwise_squared_distances, min=0.0)
    
    # Take the square root to get pairwise distances
    pairwise_distances = torch.sqrt(pairwise_squared_distances)
    
    return pairwise_distances
x_flat = torch.randn(4, 2, requires_grad=True)      
print("hello world 3")    
distances1 = torch.norm(x_flat[:, None] - x_flat, dim=2, p=2)
distances2 = pairwise_distance_matrix(x_flat)
print(x_flat)
print(distances1)
print(distances2)