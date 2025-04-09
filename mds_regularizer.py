import torch
import torch.nn as nn
import torch.nn.functional as F

class MDSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "mds_loss"

    def forward(self, distances1, distances2):
        # Create a mask to ignore the diagonal
        mask = ~torch.eye(distances1.size(0), device=distances1.device).bool()

        # Apply the mask to get only off-diagonal elements
        d1_offdiag = distances1[mask]
        d2_offdiag = distances2[mask]

        # Compute MSE loss only on off-diagonal elements
        loss = F.mse_loss(d1_offdiag, d2_offdiag)
        return loss, {}


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MDSLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.name = "mds_loss"

#     def forward(self, distances1, distances2):
#         # Use simple mean squared error (L2 loss)
#         loss = F.mse_loss(distances1, distances2)
#         return loss, {}