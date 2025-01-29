
import torch.nn as nn
import torch

class MDSLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, distances1, distances2):
        triangular_indices = torch.triu_indices(distances1.size(0), distances2.size(1), offset=0)

        # Get the upper triangular elements from both matrices
        upper_triangular1 = distances1[triangular_indices[0], triangular_indices[1]]
        upper_triangular2 = distances2[triangular_indices[0], triangular_indices[1]]

        # Compute the L2 (Euclidean) distance
        l2_distance = torch.norm(upper_triangular1 - upper_triangular2)

        # Normalize the distance by the number of upper triangular elements (excluding the diagonal)
        num_elements = upper_triangular1.numel()

        normalized_l2_distance = l2_distance / torch.sqrt(torch.tensor(num_elements, dtype=torch.float32))
        
        return normalized_l2_distance,{}