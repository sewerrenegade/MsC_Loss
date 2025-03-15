import torch.nn as nn
import torch

from dect.directions import generate_uniform_directions
from dect.ect import compute_ect
from dect.ect_fn import scaled_sigmoid

class ECT_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "ect_loss"
        self.final_loss = torch.nn.MSELoss()
        
    @staticmethod
    def _flatten(x, transform=None):
        if transform:
            x = transform(x.squeeze(dim=0))
        x_flat = x.view(x.size(0), -1)
        return x_flat
    
    def forward(self, space1, space2):
        input_v = generate_uniform_directions(num_thetas=space1.shape[0], d=space1.shape[1], seed=42, device=space1.device).to(space1.device)
        input_ect = compute_ect(space1, v=input_v, radius=1, resolution=space1.shape[0], scale=500, ect_fn=scaled_sigmoid)

        latent_v = generate_uniform_directions(num_thetas=space2.shape[0], d=space2.shape[1], seed=42, device=space2.device).to(space2.device)
        latent_ect = compute_ect(space2, v=latent_v, radius=1, resolution=space2.shape[0], scale=500, ect_fn=scaled_sigmoid)
        loss = self.final_loss(input_ect,latent_ect)
        return loss,{}