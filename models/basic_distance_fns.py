import torch

def pairwise_euclidean_distance(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise Euclidean distance matrix of a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (n, ..., ...).

    Returns:
        torch.Tensor: Distance matrix of shape (n, n).
    """
    # n = x.shape[0]
    # # x_flat = x.view(n, -1)  # Flatten all but the first dimension
    # # x_squared = torch.sum(x_flat ** 2, dim=1, keepdim=True)  # (n, 1)
    
    # # # Compute pairwise Euclidean distance using broadcasting
    # # dist_matrix = x_squared - 2 * (x_flat @ x_flat.T) + x_squared.T
    # # dist_matrix = torch.sqrt(torch.clamp(dist_matrix, min=0))  # Ensure non-negative sqrt
    # # has_nan_per_row = torch.isnan(x).any()

    # # print("Contains NaN:", has_nan_per_row)
    x_flat = x.view(x.size(0), -1)
    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=2)
    return distances

def pairwise_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise Cosine similarity matrix of a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (n, ..., ...).

    Returns:
        torch.Tensor: Cosine similarity matrix of shape (n, n).
    """
    n = x.shape[0]
    x_flat = x.view(n, -1)  # Flatten all but the first dimension
    
    # Normalize each row to unit length
    x_norm = x_flat / (torch.norm(x_flat, dim=1, keepdim=True) + 1e-8)  # Avoid division by zero
    
    # Compute cosine similarity as dot product between normalized vectors
    sim_matrix = x_norm @ x_norm.T
    
    return sim_matrix
