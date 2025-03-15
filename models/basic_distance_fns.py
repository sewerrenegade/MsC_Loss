import torch

def pairwise_euclidean_distance(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise Euclidean distance matrix of a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (n, ..., ...).

    Returns:
        torch.Tensor: Distance matrix of shape (n, n).
    """
    x_flat = x.view(x.size(0), -1)
    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=2)
    return distances


def pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise Cosine distance matrix of a tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape (n, ..., ...).

    Returns:
        torch.Tensor: Cosine distance matrix of shape (n, n).
    """
    x_flat = x.view(x.size(0), -1)  # Flatten all but the first dimension

    # Normalize each row to unit length
    x_norm = x_flat / (torch.norm(x_flat, dim=1, keepdim=True) + 1e-8)  # Avoid division by zero

    # Compute cosine similarity
    sim_matrix = x_norm @ x_norm.T

    # Convert similarity to distance
    dist_matrix = 1 - sim_matrix

    return dist_matrix

